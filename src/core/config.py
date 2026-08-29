from pathlib import Path
import os
from pydantic import BaseModel, Field

# Base directories
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = DATA_DIR / "models"


class Settings(BaseModel):
    """Central application and ML configuration."""

    API_TITLE: str = "Production Sentiment Analysis API"
    API_VERSION: str = "0.1.0"
    API_DESCRIPTION: str = (
        "High-performance FastAPI production service for real-time sentiment analysis. "
        "Reuses Phase 1 preprocessing, TF-IDF vectorizer, and trained classifier artifacts."
    )
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # Artifact paths (configurable via environment variables)
    ROOT_PATH: Path = ROOT_DIR
    DATA_PATH: Path = DATA_DIR
    MODEL_DIR_PATH: Path = MODEL_DIR
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "sentiment_model.pkl")))
    VECTORIZER_PATH: Path = Path(os.getenv("VECTORIZER_PATH", str(MODEL_DIR / "tfidf_vectorizer.pkl")))
    METRICS_PATH: Path = Path(os.getenv("METRICS_PATH", str(MODEL_DIR / "metrics.json")))
    METADATA_PATH: Path = Path(os.getenv("METADATA_PATH", str(MODEL_DIR / "model_metadata.json")))

    # Validation thresholds
    MIN_TEXT_LENGTH: int = 1
    MAX_TEXT_LENGTH: int = 10000

    # In-memory history buffer size for Day 1
    MAX_HISTORY_RECORDS: int = 500


settings = Settings()
