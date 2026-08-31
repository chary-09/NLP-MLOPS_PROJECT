import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.connection import SessionLocal
from src.database.repository import PredictionRepository
from src.model.predictor import SentimentPredictor


@pytest.fixture
def client():
    """TestClient fixture managing application lifecycle."""
    with TestClient(app) as test_client:
        yield test_client


# 1. Health check endpoint (verifies API, model, and database connection)
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api"] is True
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True
    assert data["database_connected"] is True
    assert "model_version" in data
    assert "timestamp" in data


# 2. Prediction generation and database persistence (POST /predict)
def test_predict_and_database_persistence(client):
    payload = {"text": "This product was fantastic and worked wonderfully!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Prediction is generated
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "positive"
    # 7. Confidence is correct (numerical float)
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    # 6. Model version is correct
    assert data["model_version"] == "0.1.0"
    # 5. Timestamp is present and formatted
    assert "T" in data["timestamp"]
    assert len(data["prediction_id"]) == 36

    # 2. Prediction is saved in the database
    db = SessionLocal()
    try:
        repo = PredictionRepository(db)
        record = repo.get_by_id(data["prediction_id"])
        assert record is not None
        assert record.input_text == payload["text"]
        assert record.sentiment == "positive"
        assert record.model_version == "0.1.0"
        assert abs(record.confidence - data["confidence"]) < 1e-4
    finally:
        db.close()


# 3. GET /predictions returns the stored prediction
def test_get_predictions_returns_saved_records(client):
    review = "A completely terrible and horrible experience."
    post_res = client.post("/predict", json={"text": review})
    assert post_res.status_code == 200
    created_id = post_res.json()["prediction_id"]

    # 3. GET /predictions returns it
    get_res = client.get("/predictions?limit=10")
    assert get_res.status_code == 200
    history = get_res.json()
    assert history["total"] >= 1
    assert any(p["prediction_id"] == created_id for p in history["predictions"])


# 4. Multiple predictions are stored and paginated
def test_multiple_predictions_and_pagination(client):
    texts = [
        "First review is great!",
        "Second review is bad.",
        "Third review is awesome!",
        "Fourth review is poor.",
    ]
    for t in texts:
        client.post("/predict", json={"text": t})

    # Test limit=2, offset=0
    res_page1 = client.get("/predictions?limit=2&offset=0")
    assert res_page1.status_code == 200
    p1 = res_page1.json()
    assert p1["total"] >= 4
    assert len(p1["predictions"]) == 2

    # Test limit=2, offset=2
    res_page2 = client.get("/predictions?limit=2&offset=2")
    assert res_page2.status_code == 200
    p2 = res_page2.json()
    assert len(p2["predictions"]) == 2
    assert p1["predictions"][0]["prediction_id"] != p2["predictions"][0]["prediction_id"]


# 8. Database survives API restart
def test_database_survives_api_restart():
    # Session 1: Post prediction
    with TestClient(app) as client1:
        res1 = client1.post("/predict", json={"text": "Survives restart test."})
        pred_id = res1.json()["prediction_id"]

    # Session 2: New fresh TestClient simulates restart
    with TestClient(app) as client2:
        res2 = client2.get("/predictions?limit=50")
        assert res2.status_code == 200
        found = any(p["prediction_id"] == pred_id for p in res2.json()["predictions"])
        assert found is True


# 9. Verify GET /predictions does NOT run the NLP model
def test_get_predictions_does_not_call_model(client):
    with patch.object(SentimentPredictor, "predict") as mock_predict:
        response = client.get("/predictions?limit=5")
        assert response.status_code == 200
        # The model's predict method should NEVER be invoked during GET /predictions
        mock_predict.assert_not_called()


# 10. Validation error testing
def test_predict_validation_errors(client):
    # Empty string
    assert client.post("/predict", json={"text": ""}).status_code == 422
    # Whitespace only
    assert client.post("/predict", json={"text": "   \n\t  "}).status_code == 422
    # Missing field
    assert client.post("/predict", json={"query": "Hello"}).status_code == 422


# 11. Model Info and Metrics endpoints
def test_model_info_and_metrics_endpoints(client):
    # Model Info
    info_res = client.get("/model-info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert info_data["model_name"] == "logistic_regression"
    assert "vectorizer" in info_data
    assert info_data["classes"] == ["negative", "positive"]

    # Metrics
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    metrics_data = metrics_res.json()
    assert metrics_data["total_predictions"] >= 1
    assert "sentiment_distribution" in metrics_data
    assert isinstance(metrics_data["average_confidence"], float)


# 12. Verification of Phase 1 parity
def test_phase1_prediction_parity(client):
    test_sentence = "The cinematography was brilliant and thrilling."
    phase1_result = SentimentPredictor().load().predict(test_sentence)

    response = client.post("/predict", json={"text": test_sentence})
    assert response.status_code == 200
    phase2_result = response.json()

    assert phase1_result["sentiment"].lower() == phase2_result["sentiment"].lower()
    assert abs(phase1_result["confidence"] - phase2_result["confidence"]) < 1e-3
