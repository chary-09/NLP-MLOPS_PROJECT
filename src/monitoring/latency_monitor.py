from time import perf_counter


class LatencyMonitor:
    def start(self) -> None:
        self._started_at = perf_counter()

    def elapsed_ms(self) -> float:
        return (perf_counter() - self._started_at) * 1000
