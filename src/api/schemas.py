from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
