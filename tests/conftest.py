import json
from pathlib import Path
import pytest
import joblib

from src.config import DATA_DIR, MODEL_DIR
from src.database.migrations import create_tables
from src.model.trainer import train_models
from src.api.dependencies import get_predictor


@pytest.fixture
def sample_text():
    return "This project is excellent."


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure artifacts and database tables exist for tests."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "sentiment_model.pkl"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    metrics_path = MODEL_DIR / "metrics.json"
    metadata_path = MODEL_DIR / "model_metadata.json"

    if not model_path.exists() or not vectorizer_path.exists():
        sample_texts = [
            "The product was amazing and exceeded all my expectations!",
            "I absolutely loved this movie, it was fantastic and thrilling!",
            "Outstanding work and brilliant performance!",
            "A truly remarkable and brilliant film with outstanding acting.",
            "This was the worst purchase ever, completely defective and awful.",
            "This was the worst experience of my life, completely terrible and boring.",
            "Horrible plot, waste of time and money, totally disappointed.",
            "Terrible service and awful quality.",
        ]
        sample_labels = [
            "positive",
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
            "negative",
        ]

        models, vectorizer = train_models(sample_texts, sample_labels, max_features=5000)
        joblib.dump(models["logistic_regression"], model_path)
        joblib.dump(vectorizer, vectorizer_path)

    if not metrics_path.exists():
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_model": "logistic_regression",
                    "models": {
                        "logistic_regression": {
                            "accuracy": 0.8904,
                            "precision": 0.8906,
                            "recall": 0.8904,
                            "f1_score": 0.8904,
                        }
                    },
                },
                f,
                indent=2,
            )

    if not metadata_path.exists():
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"model_version": "0.1.0"}, f, indent=2)

    # Initialize database tables
    create_tables()

    # Preload predictor
    get_predictor()
