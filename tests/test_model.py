from src.model.evaluator import evaluate_model
from src.model.predictor import SentimentPredictor
from src.model.trainer import train_model


def test_train_model():
    model, vectorizer = train_model(["great", "bad", "okay"], ["positive", "negative", "neutral"])
    assert model.predict(vectorizer.transform(["great"]))[0] == "positive"


def test_evaluation_contains_required_metrics():
    texts = ["great", "bad", "okay"]
    labels = ["positive", "negative", "neutral"]
    model, vectorizer = train_model(texts, labels)
    metrics = evaluate_model(model, vectorizer, texts, labels)
    assert {"accuracy", "precision", "recall", "f1_score", "confusion_matrix"} <= metrics.keys()


def test_predictor_returns_uppercase_sentiment(tmp_path):
    model, vectorizer = train_model(["great", "bad"], ["positive", "negative"])
    from src.model.model_utils import save_artifact

    save_artifact(model, tmp_path / "model.pkl")
    save_artifact(vectorizer, tmp_path / "vectorizer.pkl")
    result = SentimentPredictor(tmp_path / "model.pkl", tmp_path / "vectorizer.pkl").predict("great")
    assert result["sentiment"] == "POSITIVE"
    assert 0 <= result["confidence"] <= 1
