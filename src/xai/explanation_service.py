"""Explanation service — orchestrates SHAP and LIME explainers.

Designed as a singleton (one per process) that holds the SHAP and LIME
explainer instances so they are initialised only once (LinearExplainer
fitting and LimeTextExplainer construction are not expensive, but it is
good practice to reuse them).

The service also normalises the outputs of both explainers into a unified
ExplanationResult schema so the FastAPI route and the dashboard both
receive a consistent data structure.
"""

import logging
import math
from typing import List, Dict, Any, Literal, Optional

from src.model.predictor import SentimentPredictor
from src.xai.shap_explainer import SHAPExplainer
from src.xai.lime_explainer import LIMEExplainer
from src.nlp.preprocessor import clean_text

logger = logging.getLogger("xai.service")

ExplainMethod = Literal["shap", "lime", "both"]


class ExplanationService:
    """Reusable explanation service wrapping SHAP and LIME for the Phase 1 pipeline.

    Parameters
    ----------
    predictor : SentimentPredictor — the same singleton used by /predict
    """

    def __init__(self, predictor: SentimentPredictor):
        self._predictor = predictor
        self._shap: Optional[SHAPExplainer] = None
        self._lime: Optional[LIMEExplainer] = None

    # ------------------------------------------------------------------
    # Lazy explainer init
    # ------------------------------------------------------------------

    def _get_shap(self) -> SHAPExplainer:
        if self._shap is None:
            self._shap = SHAPExplainer(
                model=self._predictor.model,
                vectorizer=self._predictor.vectorizer,
            )
        return self._shap

    def _get_lime(self) -> LIMEExplainer:
        if self._lime is None:
            classes = ["negative", "positive"]
            if hasattr(self._predictor.model, "classes_"):
                classes = [str(c).lower() for c in self._predictor.model.classes_]
            self._lime = LIMEExplainer(
                model=self._predictor.model,
                vectorizer=self._predictor.vectorizer,
                class_names=classes,
            )
        return self._lime

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_predict(self, text: str) -> Dict[str, Any]:
        """Run inference via the same SentimentPredictor used by /predict."""
        result = self._predictor.predict(text)
        return result

    @staticmethod
    def _split_contributions(
        features: List[Dict[str, Any]]
    ) -> tuple[List[Dict], List[Dict]]:
        """Split feature list into positive-contributing and negative-contributing tokens."""
        positive = [f for f in features if f["importance"] > 0]
        negative = [f for f in features if f["importance"] < 0]
        positive.sort(key=lambda x: x["importance"], reverse=True)
        negative.sort(key=lambda x: x["importance"])
        return positive, negative

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        text: str,
        method: ExplainMethod = "lime",
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Generate an explanation for the given text.

        Parameters
        ----------
        text   : Raw input text (same as sent to POST /predict)
        method : "shap", "lime", or "both"
        top_n  : Number of top features per explainer

        Returns
        -------
        Dict with keys:
            prediction, confidence, model_version, method,
            explanation (list of {feature, importance}),
            positive_words, negative_words
        """
        if not text or not text.strip():
            return {
                "error": "Input text must not be empty or blank.",
                "prediction": None,
                "confidence": None,
                "method": method,
                "explanation": [],
                "positive_words": [],
                "negative_words": [],
            }

        # 1. Run inference (ensures consistency with /predict)
        pred = self._run_predict(text)
        sentiment = pred["sentiment"].lower()
        confidence = round(pred["confidence"], 4)
        model_version = pred.get("model_version", "0.1.0")

        # 2. Generate feature explanations
        explanation: List[Dict[str, Any]] = []

        if method in ("shap", "both"):
            try:
                shap_features = self._get_shap().explain(text, top_n=top_n)
                if method == "shap":
                    explanation = shap_features
                else:
                    # Tag them when returning both
                    for f in shap_features:
                        f["method"] = "shap"
                    explanation.extend(shap_features)
            except Exception as exc:
                logger.error("SHAP explain error: %s", exc, exc_info=True)

        if method in ("lime", "both"):
            try:
                lime_features = self._get_lime().explain(text, top_n=top_n)
                if method == "lime":
                    explanation = lime_features
                else:
                    for f in lime_features:
                        f["method"] = "lime"
                    explanation.extend(lime_features)
            except Exception as exc:
                logger.error("LIME explain error: %s", exc, exc_info=True)

        # 3. Split into positive/negative word lists (human-readable)
        positive_words, negative_words = self._split_contributions(
            [f for f in explanation if "method" not in f or f.get("method") == method]
            if method != "both"
            else explanation
        )

        return {
            "prediction": sentiment,
            "confidence": confidence,
            "model_version": model_version,
            "method": method,
            "explanation": explanation,
            "positive_words": [f["feature"] for f in positive_words[:top_n]],
            "negative_words": [f["feature"] for f in negative_words[:top_n]],
        }
