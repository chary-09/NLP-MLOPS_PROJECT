import json
import os
from pathlib import Path
import pytest
import joblib

from src.core.config import settings
from src.api.services.model_service import model_service
from src.model.trainer import train_models


@pytest.fixture
def sample_text():
    return "This project is excellent."


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts_available():
    """Ensure trained model and vectorizer artifacts exist and are loaded for test execution."""
    settings.MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # If artifacts are missing (e.g. in fresh CI environment), generate lightweight test artifacts
    if not settings.MODEL_PATH.exists() or not settings.VECTORIZER_PATH.exists():
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
        joblib.dump(models["logistic_regression"], settings.MODEL_PATH)
        joblib.dump(vectorizer, settings.VECTORIZER_PATH)

    if not settings.METRICS_PATH.exists():
        with open(settings.METRICS_PATH, "w", encoding="utf-8") as f:
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

    if not settings.METADATA_PATH.exists():
        with open(settings.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"model_version": "0.1.0"}, f, indent=2)

    # Load artifacts into model_service singleton
    model_service.load_artifacts()
