"""POST /explain route — Explainable AI endpoint (Day 3).

Accepts the same raw text as POST /predict, runs the same Phase 1
inference pipeline, then generates SHAP or LIME feature-level explanations
that describe *why* the model made that prediction.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_explanation_service
from src.api.schemas import ExplainRequest, ExplainResponse, FeatureContribution
from src.xai.explanation_service import ExplanationService

logger = logging.getLogger("explain_route")
router = APIRouter(tags=["Explainable AI"])


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain Sentiment Prediction (SHAP / LIME)",
    description=(
        "Runs the same TF-IDF → Logistic Regression inference as POST /predict, "
        "then produces word-level feature contributions using SHAP (LinearExplainer) "
        "or LIME (LimeTextExplainer).  "
        "Set `method` to `'shap'`, `'lime'`, or `'both'`."
    ),
    responses={
        200: {
            "description": "Successful explanation",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "negative",
                        "confidence": 0.89,
                        "model_version": "0.1.0",
                        "method": "lime",
                        "explanation": [
                            {"feature": "terrible", "importance": -0.51},
                            {"feature": "amazing", "importance": 0.25},
                        ],
                        "positive_words": ["amazing"],
                        "negative_words": ["terrible"],
                    }
                }
            },
        },
        422: {"description": "Validation error (e.g. empty text, invalid method)"},
        503: {"description": "Model or explainer not available"},
    },
)
def explain_sentiment(
    request: ExplainRequest,
    svc: ExplanationService = Depends(get_explanation_service),
) -> ExplainResponse:
    """Generate an XAI explanation for the given text."""
    try:
        result = svc.explain(
            text=request.text,
            method=request.method,
            top_n=request.top_n,
        )
    except FileNotFoundError as exc:
        logger.error("Model artifacts missing during explanation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run training scripts first.",
        ) from exc
    except Exception as exc:
        logger.error("Explanation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(exc)}",
        ) from exc

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    contributions = [
        FeatureContribution(
            feature=f["feature"],
            importance=f["importance"],
            method=f.get("method"),
        )
        for f in result["explanation"]
    ]

    return ExplainResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        method=result["method"],
        explanation=contributions,
        positive_words=result["positive_words"],
        negative_words=result["negative_words"],
    )
