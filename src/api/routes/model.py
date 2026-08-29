from fastapi import APIRouter, Depends, HTTPException, status
from src.api.dependencies import get_model_service
from src.api.schemas.model import ModelInfoResponse
from src.api.services.model_service import ModelService

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model & Vectorizer Metadata",
    description="Returns metadata about the active ML model, TF-IDF vectorizer configuration, evaluation metrics, and supported classes.",
)
def get_model_info(
    model_svc: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Retrieve model, vectorizer, and evaluation metadata."""
    try:
        return model_svc.get_model_info()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to retrieve model information: {str(exc)}",
        ) from exc
