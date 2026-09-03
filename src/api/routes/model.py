import json
from fastapi import APIRouter, Depends, HTTPException, status

from src.config import MODEL_DIR
from src.api.dependencies import get_predictor
from src.api.schemas.model import ModelInfoResponse
from src.model.predictor import SentimentPredictor

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model & Vectorizer Metadata",
    description="Returns metadata about the active ML model, TF-IDF vectorizer configuration, evaluation metrics, and supported classes.",
)
def get_model_info(
    predictor: SentimentPredictor = Depends(get_predictor),
) -> ModelInfoResponse:
    """Retrieve model, vectorizer, and evaluation metadata."""
    try:
        metrics_path = MODEL_DIR / "metrics.json"
        metrics_data = {}
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except Exception:
                metrics_data = {}

        vectorizer = predictor.vectorizer
        vectorizer_info = {
            "type": vectorizer.__class__.__name__ if vectorizer else "NotLoaded",
            "vectorizer_type": vectorizer.__class__.__name__ if vectorizer else "NotLoaded",
            "max_features": getattr(vectorizer, "max_features", 5000),
            "ngram_range": list(getattr(vectorizer, "ngram_range", (1, 2))),
            "vocabulary_size": len(getattr(vectorizer, "vocabulary_", {})),
        }

        classes = ["negative", "positive"]
        if predictor.model and hasattr(predictor.model, "classes_"):
            classes = [str(c).lower() for c in predictor.model.classes_]

        best_model_name = metrics_data.get("best_model", "logistic_regression")
        training_metrics = metrics_data.get("models", {}).get(best_model_name)

        return ModelInfoResponse(
            model_name=best_model_name,
            model_version=predictor.model_version,
            vectorizer=vectorizer_info,
            classes=classes,
            training_metrics=training_metrics,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to retrieve model information: {str(exc)}",
        ) from exc
