"""System latency and API metrics monitoring."""

import threading
from time import perf_counter
from typing import Any, Dict, List


class LatencyMonitor:
    """Utility class to measure latency of individual code blocks or requests."""

    def __init__(self) -> None:
        self._started_at: float = perf_counter()

    def start(self) -> None:
        """Start or reset latency timing."""
        self._started_at = perf_counter()

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return (perf_counter() - self._started_at) * 1000.0

    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds."""
        return perf_counter() - self._started_at


class SystemMetricsTracker:
    """Thread-safe collector for FastAPI HTTP request metrics and latencies."""

    def __init__(self, max_latency_history: int = 1000) -> None:
        self._lock = threading.Lock()
        self.max_latency_history = max_latency_history
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics (useful for testing and window resets)."""
        with self._lock:
            self._total_requests: int = 0
            self._prediction_requests: int = 0
            self._successful_requests: int = 0
            self._failed_requests: int = 0
            self._api_error_count: int = 0
            self._endpoint_counts: Dict[str, int] = {}
            self._latencies: List[float] = []

    def record_request(self, path: str, status_code: int, latency_seconds: float) -> None:
        """Record a completed request with its endpoint, status code, and latency in seconds."""
        with self._lock:
            self._total_requests += 1

            # Track prediction-specific requests
            clean_path = path.rstrip("/") if path != "/" else "/"
            if clean_path in ("/predict", "/api/v1/predict"):
                self._prediction_requests += 1

            # Track success vs failure
            if status_code < 400:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
                self._api_error_count += 1

            # Track endpoint counts
            self._endpoint_counts[clean_path] = self._endpoint_counts.get(clean_path, 0) + 1

            # Track latency
            self._latencies.append(latency_seconds)
            if len(self._latencies) > self.max_latency_history:
                self._latencies.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve aggregated system performance and request metrics."""
        with self._lock:
            total = self._total_requests
            latencies = list(self._latencies)
            endpoints = dict(self._endpoint_counts)
            prediction_reqs = self._prediction_requests
            success = self._successful_requests
            failed = self._failed_requests
            errors = self._api_error_count

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = 0.0
            min_latency = 0.0
            max_latency = 0.0

        error_rate_pct = round((failed / total * 100.0), 2) if total > 0 else 0.0

        return {
            "total_requests": total,
            "prediction_requests": prediction_reqs,
            "successful_requests": success,
            "failed_requests": failed,
            "api_error_count": errors,
            "error_rate_percentage": error_rate_pct,
            "average_latency_seconds": round(avg_latency, 4),
            "min_latency_seconds": round(min_latency, 4),
            "max_latency_seconds": round(max_latency, 4),
            "average_latency_ms": round(avg_latency * 1000.0, 2),
            "min_latency_ms": round(min_latency * 1000.0, 2),
            "max_latency_ms": round(max_latency * 1000.0, 2),
            "endpoint_request_counts": endpoints,
        }


# Global singleton tracker instance
system_metrics_tracker = SystemMetricsTracker()
