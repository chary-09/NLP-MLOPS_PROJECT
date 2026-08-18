def test_dashboard_module_imports():
    from src.dashboard import theme
    assert theme.PRIMARY_COLOR
