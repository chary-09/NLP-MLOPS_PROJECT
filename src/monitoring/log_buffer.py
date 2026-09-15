"""Recent records emitted by the application's existing logging system."""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


class LogBufferHandler(logging.Handler):
    """Keep a bounded copy of real log records for read-only API inspection."""

    def __init__(self, max_records: int = 500) -> None:
        super().__init__()
        self._records: Deque[Dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        with self._lock:
            self._records.append(entry)

    def get_records(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._records)
        return [
            record
            for record in records
            if (not level or record["level"] == level.upper())
            and (not component or record["component"] == component)
            and (not date or record["timestamp"].startswith(date))
        ]


log_buffer_handler = LogBufferHandler()


def install_log_buffer_handler() -> None:
    """Attach the capture handler without replacing configured log handlers."""
    root_logger = logging.getLogger()
    if log_buffer_handler not in root_logger.handlers:
        root_logger.addHandler(log_buffer_handler)