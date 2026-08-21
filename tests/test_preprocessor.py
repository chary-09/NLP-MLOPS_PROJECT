from src.nlp.preprocessor import clean_text


def test_clean_text():
    assert clean_text("  HELLO   world ") == "hello world"


def test_clean_text_removes_html_and_preserves_negations():
    cleaned = clean_text("<br /> I do NOT like this, no way! Never again.")
    assert cleaned == "i do not like this no way never again"
