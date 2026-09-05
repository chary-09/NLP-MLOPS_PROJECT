"""Day 5 End-to-End Integration, Consistency, and Readiness Tests."""

import json
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import MODEL_DIR
from src.database.connection import SessionLocal
from src.database.repository import PredictionRepository
from src.model.model_utils import load_artifact
from src.model.predictor import SentimentPredictor
from src.monitoring.metrics import MonitoringService
from src.nlp.preprocessor import clean_text

CONSISTENCY_TEXT = "The product was amazing but delivery was terrible."


def test_unified_inference_pipeline_consistency():
    """Verify 100% parity across Phase 1 direct inference, SentimentPredictor, FastAPI, SHAP, and LIME."""
    # 1. Phase 1 direct calculation
    vec = load_artifact(MODEL_DIR / "tfidf_vectorizer.pkl")
    model = load_artifact(MODEL_DIR / "sentiment_model.pkl")
    cleaned_p1 = clean_text(CONSISTENCY_TEXT)
    features_p1 = vec.transform([cleaned_p1])
    p1_pred = str(model.predict(features_p1)[0]).upper()
    p1_conf = float(max(model.predict_proba(features_p1)[0])) if hasattr(model, "predict_proba") else 0.0

    # 2. CLI / SentimentPredictor
    predictor = SentimentPredictor().load()
    cli_result = predictor.predict(CONSISTENCY_TEXT)

    # 3. FastAPI /predict
    client = TestClient(app)
    api_resp = client.post("/predict", json={"text": CONSISTENCY_TEXT})
    assert api_resp.status_code == 200
    api_result = api_resp.json()

    # 4. XAI (SHAP & LIME)
    shap_resp = client.post("/explain", json={"text": CONSISTENCY_TEXT, "method": "shap", "top_n": 5})
    assert shap_resp.status_code == 200
    shap_result = shap_resp.json()

    lime_resp = client.post("/explain", json={"text": CONSISTENCY_TEXT, "method": "lime", "top_n": 5})
    assert lime_resp.status_code == 200
    lime_result = lime_resp.json()

    # Parity verification
    assert p1_pred == cli_result["sentiment"] == api_result["sentiment"].upper() == shap_result["prediction"].upper() == lime_result["prediction"].upper()
    assert abs(p1_conf - cli_result["confidence"]) < 1e-4
    assert abs(p1_conf - api_result["confidence"]) < 1e-4
    assert abs(p1_conf - shap_result["confidence"]) < 1e-4
    assert abs(p1_conf - lime_result["confidence"]) < 1e-4


def test_api_edge_cases_comprehensive():
    """Verify input validation, edge cases, and proper HTTP response codes."""
    client = TestClient(app)

    # Valid edge cases
    assert client.post("/predict", json={"text": "Super good!"}).status_code == 200
    assert client.post("/predict", json={"text": "Very bad!"}).status_code == 200
    assert client.post("/predict", json={"text": "Good but bad."}).status_code == 200
    assert client.post("/predict", json={"text": "@#$%^&*()!"}).status_code == 200
    assert client.post("/predict", json={"text": "great " * 500}).status_code == 200

    # Invalid edge cases
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "    "}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": 12345}).status_code == 422
    assert client.post("/predict", content="not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_database_persistence_and_pagination_contract():
    """Verify database persistence and pagination integrity."""
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "Day 5 contract persistence verification."})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_id" in data
    assert "timestamp" in data

    hist_resp = client.get("/predictions?limit=3&offset=0")
    assert hist_resp.status_code == 200
    hist = hist_resp.json()
    assert "total" in hist
    assert "predictions" in hist
    assert len(hist["predictions"]) <= 3


def test_xai_both_method_contract():
    """Verify method='both' generates combined SHAP and LIME explanations."""
    client = TestClient(app)
    resp = client.post("/explain", json={"text": CONSISTENCY_TEXT, "method": "both", "top_n": 3})
    assert resp.status_code == 200
    data = resp.json()
    methods = {item.get("method") for item in data["explanation"]}
    assert "shap" in methods
    assert "lime" in methods


def test_monitoring_metrics_honesty():
    """Verify that production metrics are flagged UNAVAILABLE without real ground truth."""
    service = MonitoringService()
    db = SessionLocal()
    try:
        repo = PredictionRepository(db)
        predictor = SentimentPredictor().load()
        summary = service.get_full_summary(repo=repo, predictor=predictor)
    finally:
        db.close()

    assert summary["model_performance"]["production_metrics"]["status"] == "UNAVAILABLE"
    assert summary["model_performance"]["production_metrics"]["accuracy"] is None
    assert summary["model_performance"]["baseline_metrics"]["accuracy"] >= 0.85
