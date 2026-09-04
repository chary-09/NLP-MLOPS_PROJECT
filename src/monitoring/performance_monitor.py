"""Model performance monitoring comparing baseline evaluation against production ground-truth."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import MODEL_DIR


def performance_status(accuracy: float, minimum: float = 0.75) -> str:
    """Evaluate if model accuracy meets production criteria (backward compatibility)."""
    return "healthy" if accuracy >= minimum else "degraded"


class ModelPerformanceMonitor:
    """Tracks baseline training evaluation metrics and computes real production performance when ground-truth labels are available."""

    def __init__(self, metrics_file: Optional[Path] = None) -> None:
        self.metrics_file = metrics_file or (MODEL_DIR / "metrics.json")
        self._cached_baseline: Optional[Dict[str, Any]] = None
        self._latest_production_eval: Optional[Dict[str, Any]] = None

    def get_baseline_metrics(self) -> Dict[str, Any]:
        """Load Phase 1 baseline evaluation metrics."""
        if self._cached_baseline is not None:
            return self._cached_baseline

        default_baseline = {
            "dataset": "Phase 1 Test Set (7,500 samples)",
            "model_name": "logistic_regression",
            "accuracy": 0.8904,
            "precision": 0.8906,
            "recall": 0.8904,
            "f1_score": 0.8904,
        }

        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                best_model = data.get("best_model", "logistic_regression")
                model_metrics = data.get("models", {}).get(best_model, {})
                if model_metrics:
                    default_baseline = {
                        "dataset": "Phase 1 Test Set (IMDb)",
                        "model_name": best_model,
                        "accuracy": round(float(model_metrics.get("accuracy", 0.8904)), 4),
                        "precision": round(float(model_metrics.get("precision", 0.8906)), 4),
                        "recall": round(float(model_metrics.get("recall", 0.8904)), 4),
                        "f1_score": round(float(model_metrics.get("f1_score", 0.8904)), 4),
                    }
            except Exception:
                pass

        self._cached_baseline = default_baseline
        return self._cached_baseline

    def evaluate_ground_truth(
        self,
        predictions: List[str],
        ground_truth: List[str],
    ) -> Dict[str, Any]:
        """Calculate genuine production performance metrics from actual ground-truth labels.
        
        Never fakes metrics when ground-truth is absent.
        """
        if not predictions or not ground_truth:
            raise ValueError("Predictions and ground_truth lists must not be empty.")
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Mismatch in counts: {len(predictions)} predictions vs {len(ground_truth)} ground-truth labels."
            )

        # Standardize strings to lower case
        preds_clean = [str(p).strip().lower() for p in predictions]
        gt_clean = [str(g).strip().lower() for g in ground_truth]

        acc = float(accuracy_score(gt_clean, preds_clean))
        prec = float(
            precision_score(
                gt_clean, preds_clean, average="weighted", zero_division=0
            )
        )
        rec = float(
            recall_score(
                gt_clean, preds_clean, average="weighted", zero_division=0
            )
        )
        f1 = float(
            f1_score(
                gt_clean, preds_clean, average="weighted", zero_division=0
            )
        )

        eval_result = {
            "status": "AVAILABLE",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "sample_count": len(gt_clean),
            "evaluated_at": datetime.now(timezone.utc).isoformat() + "Z",
        }
        self._latest_production_eval = eval_result
        return eval_result

    def get_production_metrics(self) -> Dict[str, Any]:
        """Return production performance status.
        
        Clearly reports UNAVAILABLE if no real ground truth has been observed,
        preventing any false calculation from unverified predictions.
        """
        if self._latest_production_eval is not None:
            return self._latest_production_eval

        return {
            "status": "UNAVAILABLE",
            "reason": "Production ground-truth labels not yet observed",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "sample_count": 0,
        }

    def get_summary(self, model_version: str = "0.1.0") -> Dict[str, Any]:
        """Return combined model performance monitoring report."""
        baseline = self.get_baseline_metrics()
        production = self.get_production_metrics()

        return {
            "model_version": model_version,
            "baseline_metrics": baseline,
            "production_metrics": production,
            "can_calculate_production_metrics": production["status"] == "AVAILABLE",
        }


# Global singleton performance monitor
model_performance_monitor = ModelPerformanceMonitor()
