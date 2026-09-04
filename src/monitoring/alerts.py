"""Alert and threshold logic for production MLOps monitoring."""

from typing import Any, Dict, List, Optional


def should_alert(status: str) -> bool:
    """Check if status requires attention (backward compatibility)."""
    return status.upper() not in ("HEALTHY", "NORMAL")


class MonitoringAlertManager:
    """Evaluates threshold-based alerts across system, prediction, model, and drift metrics."""

    def __init__(
        self,
        low_confidence_warning_pct: float = 20.0,
        low_confidence_critical_pct: float = 40.0,
        error_rate_warning_pct: float = 5.0,
        error_rate_critical_pct: float = 15.0,
        latency_warning_seconds: float = 0.50,
        latency_critical_seconds: float = 1.00,
        drift_threshold: float = 0.20,
    ) -> None:
        self.low_confidence_warning_pct = low_confidence_warning_pct
        self.low_confidence_critical_pct = low_confidence_critical_pct
        self.error_rate_warning_pct = error_rate_warning_pct
        self.error_rate_critical_pct = error_rate_critical_pct
        self.latency_warning_seconds = latency_warning_seconds
        self.latency_critical_seconds = latency_critical_seconds
        self.drift_threshold = drift_threshold

    def get_thresholds(self) -> Dict[str, Any]:
        """Return configured threshold parameters."""
        return {
            "low_confidence_warning_pct": self.low_confidence_warning_pct,
            "low_confidence_critical_pct": self.low_confidence_critical_pct,
            "error_rate_warning_pct": self.error_rate_warning_pct,
            "error_rate_critical_pct": self.error_rate_critical_pct,
            "latency_warning_seconds": self.latency_warning_seconds,
            "latency_critical_seconds": self.latency_critical_seconds,
            "drift_threshold": self.drift_threshold,
        }

    def evaluate(
        self,
        prediction_metrics: Optional[Dict[str, Any]] = None,
        system_metrics: Optional[Dict[str, Any]] = None,
        drift_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate current telemetry against thresholds and compute overall status.
        
        Returns one of: NORMAL, WARNING, CRITICAL, DRIFT_DETECTED.
        """
        alerts: List[str] = []
        has_critical = False
        has_warning = False
        drift_detected = False

        # 1. Evaluate Prediction Metrics (Low Confidence & Distribution)
        if prediction_metrics:
            total_preds = prediction_metrics.get("total_predictions", 0)
            low_conf_pct = prediction_metrics.get("low_confidence_percentage", 0.0)

            if total_preds >= 5:
                if low_conf_pct >= self.low_confidence_critical_pct:
                    alerts.append(
                        f"CRITICAL: Low confidence predictions at {low_conf_pct}% "
                        f"(>= {self.low_confidence_critical_pct}% critical threshold)"
                    )
                    has_critical = True
                elif low_conf_pct >= self.low_confidence_warning_pct:
                    alerts.append(
                        f"WARNING: Low confidence predictions elevated at {low_conf_pct}% "
                        f"(>= {self.low_confidence_warning_pct}% warning threshold)"
                    )
                    has_warning = True

            # Check extreme sentiment distribution imbalance if sufficient volume
            if total_preds >= 20:
                pos_pct = prediction_metrics.get("positive_percentage", 0.0)
                neg_pct = prediction_metrics.get("negative_percentage", 0.0)
                if pos_pct >= 85.0 or neg_pct >= 85.0:
                    dominant = "positive" if pos_pct >= 85.0 else "negative"
                    dom_pct = max(pos_pct, neg_pct)
                    alerts.append(
                        f"WARNING: Severe sentiment distribution imbalance: {dominant} is {dom_pct}%"
                    )
                    has_warning = True

        # 2. Evaluate System Metrics (API Errors & Latencies)
        if system_metrics:
            error_rate = system_metrics.get("error_rate_percentage", 0.0)
            avg_latency = system_metrics.get("average_latency_seconds", 0.0)
            total_reqs = system_metrics.get("total_requests", 0)

            if total_reqs >= 5:
                if error_rate >= self.error_rate_critical_pct:
                    alerts.append(
                        f"CRITICAL: High API error rate at {error_rate}% "
                        f"(>= {self.error_rate_critical_pct}% critical threshold)"
                    )
                    has_critical = True
                elif error_rate >= self.error_rate_warning_pct:
                    alerts.append(
                        f"WARNING: Elevated API error rate at {error_rate}% "
                        f"(>= {self.error_rate_warning_pct}% warning threshold)"
                    )
                    has_warning = True

            if avg_latency >= self.latency_critical_seconds:
                alerts.append(
                    f"CRITICAL: High average latency of {avg_latency}s "
                    f"(>= {self.latency_critical_seconds}s critical threshold)"
                )
                has_critical = True
            elif avg_latency >= self.latency_warning_seconds:
                alerts.append(
                    f"WARNING: Elevated average latency of {avg_latency}s "
                    f"(>= {self.latency_warning_seconds}s warning threshold)"
                )
                has_warning = True

        # 3. Evaluate Data Drift
        if drift_metrics and drift_metrics.get("drift_detected", False):
            drift_detected = True
            score = drift_metrics.get("score", 0.0)
            thresh = drift_metrics.get("threshold", self.drift_threshold)
            alerts.append(
                f"DRIFT_DETECTED: NLP input data drift score is {score} (>= {thresh} threshold)"
            )

        # 4. Overall Status Determination
        if has_critical:
            system_status = "CRITICAL"
        elif drift_detected:
            system_status = "DRIFT_DETECTED"
        elif has_warning:
            system_status = "WARNING"
        else:
            system_status = "NORMAL"

        return {
            "system_status": system_status,
            "alerts": alerts,
            "thresholds": self.get_thresholds(),
        }


# Global singleton alert manager
monitoring_alert_manager = MonitoringAlertManager()
