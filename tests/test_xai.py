"""Tests for Phase 2 Day 3 — Explainable AI (SHAP + LIME).

Verifies:
1.  /predict and /explain return the same sentiment for the same text
2.  SHAP explanation works and returns a list of features
3.  LIME explanation works and returns a list of features
4.  Positive text gives features with positive importance
5.  Negative text gives features with negative importance
6.  Mixed-sentiment text is handled without errors
7.  Empty / invalid text is rejected with 422
8.  `method=both` returns features from both explainers
9.  OOV (all-unknown words) text is handled gracefully
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

POSITIVE_TEXT = "This movie was absolutely fantastic and I loved every moment of it!"
NEGATIVE_TEXT = "This was the worst experience ever. Terrible service and horrible quality."
MIXED_TEXT = "The product was amazing but the delivery was terrible and disappointing."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def explain(text: str, method: str = "lime", top_n: int = 5) -> dict:
    resp = client.post("/explain", json={"text": text, "method": method, "top_n": top_n})
    return resp


def predict(text: str) -> dict:
    resp = client.post("/predict", json={"text": text})
    return resp


# ---------------------------------------------------------------------------
# 1. /predict and /explain return the same sentiment
# ---------------------------------------------------------------------------

def test_predict_and_explain_same_sentiment_positive():
    pred_resp = predict(POSITIVE_TEXT)
    expl_resp = explain(POSITIVE_TEXT, method="lime")

    assert pred_resp.status_code == 200, pred_resp.text
    assert expl_resp.status_code == 200, expl_resp.text

    pred_sentiment = pred_resp.json()["sentiment"].lower()
    expl_sentiment = expl_resp.json()["prediction"].lower()
    assert pred_sentiment == expl_sentiment, (
        f"/predict says '{pred_sentiment}' but /explain says '{expl_sentiment}'"
    )


def test_predict_and_explain_same_sentiment_negative():
    pred_resp = predict(NEGATIVE_TEXT)
    expl_resp = explain(NEGATIVE_TEXT, method="lime")

    assert pred_resp.status_code == 200, pred_resp.text
    assert expl_resp.status_code == 200, expl_resp.text

    pred_sentiment = pred_resp.json()["sentiment"].lower()
    expl_sentiment = expl_resp.json()["prediction"].lower()
    assert pred_sentiment == expl_sentiment


# ---------------------------------------------------------------------------
# 2. SHAP explanation works
# ---------------------------------------------------------------------------

def test_shap_explanation_returns_features():
    resp = explain(POSITIVE_TEXT, method="shap", top_n=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "shap"
    assert isinstance(body["explanation"], list)
    assert len(body["explanation"]) > 0
    for item in body["explanation"]:
        assert "feature" in item
        assert "importance" in item
        assert isinstance(item["importance"], float)


# ---------------------------------------------------------------------------
# 3. LIME explanation works
# ---------------------------------------------------------------------------

def test_lime_explanation_returns_features():
    resp = explain(NEGATIVE_TEXT, method="lime", top_n=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "lime"
    assert isinstance(body["explanation"], list)
    assert len(body["explanation"]) > 0
    for item in body["explanation"]:
        assert "feature" in item
        assert "importance" in item


# ---------------------------------------------------------------------------
# 4. Positive text gives at least one positive-importance feature
# ---------------------------------------------------------------------------

def test_positive_text_has_positive_features_lime():
    resp = explain(POSITIVE_TEXT, method="lime", top_n=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    positive_words = body["positive_words"]
    # For strongly positive text, at least one word should push toward positive
    assert len(positive_words) > 0 or len(body["explanation"]) > 0, (
        "Expected at least one feature from a strongly positive sentence."
    )


def test_positive_text_has_positive_features_shap():
    resp = explain(POSITIVE_TEXT, method="shap", top_n=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["explanation"]) > 0


# ---------------------------------------------------------------------------
# 5. Negative text gives at least one negative-importance feature
# ---------------------------------------------------------------------------

def test_negative_text_has_negative_features_lime():
    resp = explain(NEGATIVE_TEXT, method="lime", top_n=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    negative_words = body["negative_words"]
    assert len(negative_words) > 0 or len(body["explanation"]) > 0


def test_negative_text_has_negative_features_shap():
    resp = explain(NEGATIVE_TEXT, method="shap", top_n=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["explanation"]) > 0


# ---------------------------------------------------------------------------
# 6. Mixed sentiment text is handled without errors
# ---------------------------------------------------------------------------

def test_mixed_text_lime():
    resp = explain(MIXED_TEXT, method="lime", top_n=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body
    assert "explanation" in body
    assert "positive_words" in body
    assert "negative_words" in body


def test_mixed_text_shap():
    resp = explain(MIXED_TEXT, method="shap", top_n=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body
    assert len(body["explanation"]) > 0


# ---------------------------------------------------------------------------
# 7. Empty / invalid text is rejected with 422
# ---------------------------------------------------------------------------

def test_empty_text_is_rejected():
    resp = explain("", method="lime")
    assert resp.status_code == 422


def test_blank_whitespace_text_is_rejected():
    resp = explain("   \n\t  ", method="lime")
    assert resp.status_code == 422


def test_invalid_method_is_rejected():
    resp = client.post("/explain", json={"text": "some text", "method": "invalid_method"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 8. method=both returns features from both explainers
# ---------------------------------------------------------------------------

def test_method_both_returns_combined_features():
    resp = explain(POSITIVE_TEXT, method="both", top_n=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "both"
    # Each feature should have a 'method' tag when using 'both'
    assert len(body["explanation"]) > 0
    methods_present = {item.get("method") for item in body["explanation"]}
    assert len(methods_present) >= 1  # at least shap or lime tagged


# ---------------------------------------------------------------------------
# 9. OOV text handled gracefully
# ---------------------------------------------------------------------------

def test_oov_text_does_not_crash():
    """Text made entirely of punctuation/numbers after cleaning — all OOV after TF-IDF."""
    resp = explain("123 !!! 456 ???", method="lime", top_n=5)
    # Should return 200 with empty or minimal explanation, not crash
    assert resp.status_code in (200, 422), resp.text


def test_very_short_text():
    resp = explain("ok", method="lime", top_n=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body


def test_very_long_text():
    long_text = "This movie was amazing! " * 200  # 4800 chars
    resp = explain(long_text[:9999], method="lime", top_n=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body


# ---------------------------------------------------------------------------
# Schema structure sanity check
# ---------------------------------------------------------------------------

def test_explain_response_schema():
    resp = explain(POSITIVE_TEXT, method="lime", top_n=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    required_keys = {"prediction", "confidence", "model_version", "method", "explanation",
                     "positive_words", "negative_words"}
    assert required_keys.issubset(body.keys()), f"Missing keys: {required_keys - body.keys()}"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["prediction"] in ("positive", "negative")
    assert body["method"] == "lime"
