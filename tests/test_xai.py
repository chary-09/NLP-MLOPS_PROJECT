def test_xai_module_imports():
    from src.xai.feature_importance import top_features
    assert callable(top_features)
