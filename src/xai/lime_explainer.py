"""LIME explainer for the Phase 1 TF-IDF + Logistic Regression sentiment model.

Uses lime.lime_text.LimeTextExplainer which perturbs the original raw text (word removal),
runs the full preprocessing → TF-IDF → model pipeline on each perturbation, and fits a
local linear surrogate to approximate the model decision boundary around the sample.

LIME is model-agnostic: it works for any black-box classifier that exposes a predict_proba
(or decision_function) interface.  It is the right choice when the model is non-linear or
when you want a human-readable "which words pushed this decision" explanation.
"""

import logging
import math
from typing import List, Dict, Any

import numpy as np

from src.nlp.preprocessor import clean_text

logger = logging.getLogger("xai.lime")

_TOP_N_DEFAULT = 10
_NUM_SAMPLES = 500   # More samples → more stable explanations, but slower


class LIMEExplainer:
    """Wraps lime.LimeTextExplainer around the production pipeline.

    Attributes
    ----------
    model      : The fitted scikit-learn classifier
    vectorizer : The fitted TfidfVectorizer
    class_names: Human-readable class labels in model.classes_ order
    """

    def __init__(self, model, vectorizer, class_names: List[str] = None):
        self.model = model
        self.vectorizer = vectorizer
        # Determine class order from model if possible
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model, "classes_"):
            self.class_names = [str(c).lower() for c in model.classes_]
        else:
            self.class_names = ["negative", "positive"]
        self._explainer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_explainer(self):
        if self._explainer is None:
            try:
                from lime.lime_text import LimeTextExplainer  # noqa: PLC0415
                self._explainer = LimeTextExplainer(class_names=self.class_names)
                logger.info("LIME LimeTextExplainer initialised with classes: %s", self.class_names)
            except ImportError:
                logger.warning(
                    "lime package not found — using word-removal perturbation fallback."
                )
                self._explainer = "_fallback_"
        return self._explainer

    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        """Prediction function passed to LIME — mirrors the /predict pipeline exactly.

        LIME calls this with many perturbed variants of the original text.
        Each variant is preprocessed with clean_text, vectorized, then classified.

        Returns a (n_samples, n_classes) probability matrix.
        """
        cleaned = [clean_text(t) for t in texts]
        features = self.vectorizer.transform(cleaned)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)
        else:
            # LinearSVC fallback: use sigmoid of decision_function
            scores = self.model.decision_function(features)
            if scores.ndim == 1:
                pos_prob = 1 / (1 + np.exp(-scores))
                return np.column_stack([1 - pos_prob, pos_prob])
            return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _fallback_explain(self, text: str, top_n: int) -> List[Dict[str, Any]]:
        """Word-removal perturbation fallback (no lime package required).

        For each unique word in the text, computes the change in the positive-class
        probability when that word is removed, which gives an additive local
        attribution similar to LIME's linear surrogate.
        """
        pos_idx = 1
        for i, cls in enumerate(self.class_names):
            if cls in ("positive", "pos", "1"):
                pos_idx = i
                break

        words = text.split()
        if not words:
            return []

        # Baseline probability with all words present
        baseline_proba = self._predict_fn([text])[0][pos_idx]

        unique_words = list(dict.fromkeys(words))  # preserve order, deduplicate
        results = []
        for word in unique_words:
            # Build perturbed text with this word removed
            perturbed = " ".join(w for w in words if w != word)
            if not perturbed.strip():
                perturbed = " "  # avoid empty string edge-case
            perturbed_proba = self._predict_fn([perturbed])[0][pos_idx]
            # Attribution = drop in positive-class probability when word is removed.
            # Positive value → word supports positive class.
            importance = float(baseline_proba - perturbed_proba)
            results.append({"feature": word, "importance": round(importance, 6)})

        # Sort by absolute importance, take top_n
        results.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return results[:top_n]

    def explain(
        self,
        text: str,
        top_n: int = _TOP_N_DEFAULT,
        num_samples: int = _NUM_SAMPLES,
    ) -> List[Dict[str, Any]]:
        """Return top_n LIME word-level contributions for the given text.

        Parameters
        ----------
        text        : Raw input text
        top_n       : Number of top features to return
        num_samples : Number of LIME perturbation samples

        Returns
        -------
        List of dicts with keys: feature, importance
            importance > 0 → word supports *positive* class
            importance < 0 → word supports *negative* class
        """
        if not text or not text.strip():
            return []

        # Determine the positive class index
        pos_idx = 1
        for i, cls in enumerate(self.class_names):
            if cls in ("positive", "pos", "1"):
                pos_idx = i
                break

        explainer = self._get_explainer()

        if explainer == "_fallback_":
            return self._fallback_explain(text, top_n)

        try:
            explanation = explainer.explain_instance(
                text,
                self._predict_fn,
                num_features=top_n,
                num_samples=num_samples,
                labels=[pos_idx],
            )
        except Exception as exc:
            logger.error("LIME explanation failed: %s", exc, exc_info=True)
            return []

        raw = explanation.as_list(label=pos_idx)
        return [
            {
                "feature": str(word),
                "importance": round(float(score), 6),
            }
            for word, score in raw
        ]
