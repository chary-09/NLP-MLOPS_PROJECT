from src.nlp.preprocessor import clean_text


def test_clean_text():
    assert clean_text("  HELLO   world ") == "hello world"
