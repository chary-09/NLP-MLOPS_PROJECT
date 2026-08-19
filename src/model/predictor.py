from pathlib import Path
from src.config import MODEL_DIR
from .model_utils import load_artifact
from src.nlp.preprocessor import clean_text


class SentimentPredictor:
    def __init__(self, model_path=MODEL_DIR / "sentiment_model.pkl", vectorizer_path=MODEL_DIR / "tfidf_vectorizer.pkl"):
        self.model_path, self.vectorizer_path = Path(model_path), Path(vectorizer_path)
        self.model = self.vectorizer = None

    def load(self):
        self.model, self.vectorizer = load_artifact(self.model_path), load_artifact(self.vectorizer_path)
        return self

    def predict(self, text: str) -> dict:
        if self.model is None:
            self.load()
        features = self.vectorizer.transform([clean_text(text)])
        label = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            confidence = float(max(self.model.predict_proba(features)[0]))
        else:
            scores = self.model.decision_function(features)[0]
            confidence = float(1 / (1 + __import__("math").exp(-abs(float(scores)))))
        return {"text": text, "sentiment": str(label).upper(), "confidence": confidence}
