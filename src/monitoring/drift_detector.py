def detect_drift(reference_mean: float, current_mean: float, threshold: float = 0.1) -> bool:
    return abs(reference_mean - current_mean) > threshold
