def performance_status(accuracy: float, minimum: float = 0.75) -> str:
    return "healthy" if accuracy >= minimum else "degraded"
