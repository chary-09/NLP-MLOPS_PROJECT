from fastapi import APIRouter, Depends, HTTPException
from .dependencies import get_predictor
from .schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, predictor=Depends(get_predictor)):
    try:
        return predictor.predict(request.text)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Model artifacts are not trained yet.") from error
