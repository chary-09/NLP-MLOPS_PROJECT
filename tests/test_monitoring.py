from src.monitoring.drift_detector import detect_drift


def test_detect_drift():
    assert detect_drift(0.1, 0.3)
