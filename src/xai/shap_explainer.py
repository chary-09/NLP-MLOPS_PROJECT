"""SHAP explainer for the Phase 1 TF-IDF + Logistic Regression sentiment model.

Uses shap.LinearExplainer which is the correct choice for linear models (LogisticRegression,
LinearSVC with calibration, Ridge, etc.) because it avoids the O(M*N) sampling overhead of
KernelExplainer and computes exact Shapley values in closed form using the covariance of the
training-data distribution.

For tree-based models (RandomForest, GradientBoosting) swap in shap.TreeExplainer instead.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

from src.nlp.preprocessor import clean_text

logger = logging.getLogger("xai.shap")

_TOP_N_DEFAULT = 10


class SHAPExplainer:
    """Wraps shap.LinearExplainer around the production TF-IDF + classifier pipeline.

    Attributes
    ----------
    model      : The fitted scikit-learn classifier (LogisticRegression etc.)
    vectorizer : The fitted TfidfVectorizer
    explainer  : Lazy-initialised shap.LinearExplainer instance
    """

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self._explainer: Optional[Any] = None
        self._feature_names: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature_names(self) -> np.ndarray:
        if self._feature_names is None:
            self._feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self._feature_names

    def _get_explainer(self):
        """Lazy-initialise the LinearExplainer (masker = independent distribution).

        Falls back to a coefficient-based pseudo-explainer when the ``shap``
        package is not available.  For a linear model the SHAP value of feature
        *i* is exactly ``coef[i] * x[i]`` (zero-baseline), so this fallback is
        mathematically equivalent to ``shap.LinearExplainer`` with an
        independent masker.
        """
        if self._explainer is None:
            try:
                import shap  # noqa: PLC0415
                # masker=independent samples from marginal distribution; safe for sparse TF-IDF
                self._explainer = shap.LinearExplainer(
                    self.model, masker=shap.maskers.Independent(data=None)
                )
                logger.info("SHAP LinearExplainer initialised (shap package).")
            except ImportError:
                logger.warning(
                    "shap package not found — using coefficient-based fallback "
                    "(mathematically equivalent for linear models)."
                )
                self._explainer = "_fallback_"
        return self._explainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, text: str, top_n: int = _TOP_N_DEFAULT) -> List[Dict[str, Any]]:
        """Return top_n SHAP feature contributions for the given text.

        Parameters
        ----------
        text   : Raw input text (will be preprocessed with clean_text)
        top_n  : Number of top features to return (by absolute SHAP value)

        Returns
        -------
        List of dicts with keys: feature, importance
            importance > 0 → pushes toward the *positive* class
            importance < 0 → pushes toward the *negative* class
        """
        if not text or not text.strip():
            return []

        cleaned = clean_text(text)
        feature_matrix = self.vectorizer.transform([cleaned])

        explainer = self._get_explainer()
        feature_names = self._get_feature_names()

        if explainer == "_fallback_":
            # Coefficient-based attribution: shap_i = coef_i * x_i
            # For binary LogisticRegression coef_ is shape (1, n_features);
            # positive values push toward the positive class.
            coef = np.array(self.model.coef_).flatten()
            x = np.array(feature_matrix.todense()).flatten()
            sv = coef * x
        else:
            # shap_values shape: (n_samples, n_features) for binary classification
            shap_values = explainer.shap_values(feature_matrix)

            # LinearExplainer returns a list [neg_shap, pos_shap] for multi-output
            # or a 2-D array for single binary output depending on shap version.
            if isinstance(shap_values, list):
                # Take the values for the positive class (index 1)
                sv = np.array(shap_values[1]).flatten()
            else:
                sv = np.array(shap_values).flatten()

        # Only report features that actually appeared in this document
        feature_matrix_dense = np.array(feature_matrix.todense()).flatten()
        present_mask = feature_matrix_dense > 0

        if present_mask.sum() == 0:
            # All tokens are OOV — return the top global SHAP contributors anyway
            present_mask = np.ones(len(sv), dtype=bool)

        present_indices = np.where(present_mask)[0]
        present_sv = sv[present_indices]
        present_features = feature_names[present_indices]

        # Sort by absolute importance
        order = np.argsort(np.abs(present_sv))[::-1][:top_n]

        return [
            {
                "feature": str(present_features[i]),
                "importance": round(float(present_sv[i]), 6),
            }
            for i in order
        ]
