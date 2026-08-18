def test_database_module_imports():
    from src.database.connection import get_engine
    assert callable(get_engine)
