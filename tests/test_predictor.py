def test_predictor_module_imports():
    from src.model.predictor import SentimentPredictor
    assert SentimentPredictor is not None
