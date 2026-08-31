import json
import math
from pathlib import Path
from src.config import MODEL_DIR
from .model_utils import load_artifact
from src.nlp.preprocessor import clean_text


class SentimentPredictor:
    """Production inference predictor reusing Phase 1 preprocessor, TF-IDF vectorizer, and ML model."""

    def __init__(
        self,
        model_path=MODEL_DIR / "sentiment_model.pkl",
        vectorizer_path=MODEL_DIR / "tfidf_vectorizer.pkl",
        metadata_path=MODEL_DIR / "model_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.metadata_path = Path(metadata_path)
        self.model = self.vectorizer = None
        self.metadata = {}

    def load(self):
        """Load model, vectorizer, and metadata artifacts."""
        self.model = load_artifact(self.model_path)
        self.vectorizer = load_artifact(self.vectorizer_path)
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        return self

    @property
    def model_version(self) -> str:
        """Return model version from loaded metadata."""
        return self.metadata.get("model_version", "0.1.0")

    def predict(self, text: str) -> dict:
        """Classify input text sentiment and compute confidence score."""
        if self.model is None:
            self.load()
        features = self.vectorizer.transform([clean_text(text)])
        label = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            confidence = float(max(self.model.predict_proba(features)[0]))
        else:
            scores = self.model.decision_function(features)[0]
            confidence = float(1 / (1 + math.exp(-abs(float(scores)))))
        return {
            "text": text,
            "sentiment": str(label).upper(),
            "confidence": confidence,
            "model_version": self.model_version,
        }
