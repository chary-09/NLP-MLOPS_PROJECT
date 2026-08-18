from src.nlp.tokenizer import tokenize


def test_tokenize(sample_text):
    assert tokenize(sample_text) == ["this", "project", "is", "excellent."]
