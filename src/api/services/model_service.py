from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from src.core.config import settings
from src.core.logging import get_logger
from src.api.schemas.model import ModelInfoResponse, VectorizerInfo

logger = get_logger("model_service")


class ModelService:
    """Manages lifecycle, artifact loading, and metadata for NLP models."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.vectorizer_path = Path(vectorizer_path or settings.VECTORIZER_PATH)
        self.metadata_path = Path(metadata_path or settings.METADATA_PATH)
        self.metrics_path = Path(metrics_path or settings.METRICS_PATH)

        self.model = None
        self.vectorizer = None
        self.metadata: Dict[str, Any] = {}
        self.metrics_data: Dict[str, Any] = {}
        self.startup_time = datetime.now(timezone.utc)
        self._is_loaded = False

    def load_artifacts(self) -> None:
        """Load trained model and TF-IDF vectorizer into memory."""
        logger.info(f"Loading model artifact from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info(f"Loading TF-IDF vectorizer artifact from {self.vectorizer_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {self.vectorizer_path}")
        self.vectorizer = joblib.load(self.vectorizer_path)

        # Load metadata if exists
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to parse metadata: {exc}")
                self.metadata = {}

        # Load metrics if exists
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics_data = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to parse metrics: {exc}")
                self.metrics_data = {}

        self._is_loaded = True
        logger.info("Model and vectorizer artifacts successfully loaded into memory.")

    @property
    def is_loaded(self) -> bool:
        """Check if artifacts are loaded and ready."""
        return self._is_loaded and self.model is not None and self.vectorizer is not None

    def get_version(self) -> str:
        """Return model version from metadata or default."""
        return self.metadata.get("model_version", settings.API_VERSION)

    def get_model_name(self) -> str:
        """Return model name / identifier."""
        if self.metrics_data and "best_model" in self.metrics_data:
            return self.metrics_data["best_model"]
        return self.model.__class__.__name__.lower() if self.model else "unknown"

    def get_classes(self) -> List[str]:
        """Return list of supported sentiment class labels."""
        if self.model is not None and hasattr(self.model, "classes_"):
            return [str(cls_label).lower() for cls_label in self.model.classes_]
        return ["negative", "positive"]

    def get_vectorizer_info(self) -> VectorizerInfo:
        """Return vectorizer details."""
        if self.vectorizer is None:
            return VectorizerInfo(vectorizer_type="NotLoaded")

        ngram_range = list(getattr(self.vectorizer, "ngram_range", (1, 1)))
        max_features = getattr(self.vectorizer, "max_features", None)
        vocab_size = len(getattr(self.vectorizer, "vocabulary_", {}))
        lowercase = getattr(self.vectorizer, "lowercase", True)

        return VectorizerInfo(
            vectorizer_type=self.vectorizer.__class__.__name__,
            max_features=max_features,
            ngram_range=ngram_range,
            vocabulary_size=vocab_size,
            lowercase=lowercase,
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return complete model metadata response."""
        best_name = self.get_model_name()
        training_metrics = None
        if self.metrics_data and "models" in self.metrics_data:
            training_metrics = self.metrics_data["models"].get(best_name)

        return ModelInfoResponse(
            model_name=best_name,
            model_type=self.model.__class__.__name__ if self.model else "Unknown",
            model_version=self.get_version(),
            vectorizer=self.get_vectorizer_info(),
            classes=self.get_classes(),
            training_metrics=training_metrics,
            artifact_paths={
                "model_path": str(self.model_path),
                "vectorizer_path": str(self.vectorizer_path),
                "metrics_path": str(self.metrics_path),
                "metadata_path": str(self.metadata_path),
            },
        )

    def get_uptime_seconds(self) -> float:
        """Return elapsed uptime in seconds."""
        delta = datetime.now(timezone.utc) - self.startup_time
        return round(delta.total_seconds(), 2)


# Global singleton instance
model_service = ModelService()
