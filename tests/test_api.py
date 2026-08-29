import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.model.predictor import SentimentPredictor


@pytest.fixture
def client():
    """TestClient fixture with lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client


# 1. Health check tests
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api"] is True
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data


# 2. Normal text predictions (positive & negative)
def test_predict_positive_sentiment(client):
    payload = {"text": "The product was amazing and exceeded all my expectations!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["text"] == payload["text"]
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["prediction_id"]) == 36  # UUID length
    assert "model_version" in data
    assert "timestamp" in data


def test_predict_negative_sentiment(client):
    payload = {"text": "This was the worst purchase ever, completely defective and awful."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["text"] == payload["text"]
    assert data["sentiment"] == "negative"
    assert data["confidence"] >= 0.5


# 3. Empty text validation
def test_predict_empty_string(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_predict_whitespace_only(client):
    response = client.post("/predict", json={"text": "   \n\t  "})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


# 4. Very long text handling
def test_predict_very_long_valid_text(client):
    long_review = "This movie is fantastic and wonderfully crafted. " * 100
    assert len(long_review) > 4000
    response = client.post("/predict", json={"text": long_review})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert "confidence" in data


def test_predict_text_exceeding_max_limit(client):
    too_long = "bad " * 3000
    assert len(too_long) > 10000
    response = client.post("/predict", json={"text": too_long})
    assert response.status_code == 422


# 5. Invalid request format tests
def test_predict_missing_text_field(client):
    response = client.post("/predict", json={"query": "Hello world"})
    assert response.status_code == 422


def test_predict_invalid_data_type(client):
    response = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# 6. Model info endpoint tests
def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "logistic_regression"
    assert "model_version" in data
    assert "vectorizer" in data
    assert data["vectorizer"]["max_features"] == 5000
    assert data["classes"] == ["negative", "positive"]
    assert "training_metrics" in data


# 7. Metrics and prediction history tests
def test_predictions_and_metrics_endpoints(client):
    client.post("/predict", json={"text": "Outstanding work!"})
    client.post("/predict", json={"text": "Terrible service."})

    # History endpoint
    history_res = client.get("/predictions?limit=5")
    assert history_res.status_code == 200
    history = history_res.json()
    assert history["total"] >= 2
    assert len(history["predictions"]) <= 5

    # Metrics endpoint
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert metrics["total_predictions"] >= 2
    assert metrics["total_requests"] >= 2
    assert metrics["average_confidence"] > 0.0
    assert "positive" in metrics["sentiment_distribution"]
    assert "negative" in metrics["sentiment_distribution"]


# 8. Training-to-Inference consistency check
def test_phase1_vs_phase2_consistency(client):
    test_cases = [
        "I absolutely loved this movie, it was fantastic and thrilling!",
        "This was the worst experience of my life, completely terrible and boring.",
        "A truly remarkable and brilliant film with outstanding acting.",
        "Horrible plot, waste of time and money, totally disappointed.",
    ]

    phase1_predictor = SentimentPredictor().load()

    for text in test_cases:
        phase1_result = phase1_predictor.predict(text)
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        phase2_result = response.json()

        assert phase1_result["sentiment"].lower() == phase2_result["sentiment"].lower()
        assert abs(phase1_result["confidence"] - phase2_result["confidence"]) < 1e-3
