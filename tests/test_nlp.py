from src.nlp.feature_extraction import build_vectorizer
from src.nlp.preprocessor import clean_text
from src.nlp.tokenizer import tokenize


def test_tokenize(sample_text):
    assert tokenize(sample_text) == ["this", "project", "is", "excellent"]


def test_clean_text_removes_urls_and_punctuation():
    assert clean_text("Great! Visit https://example.com") == "great visit"


def test_tfidf_vectorizer_creates_features():
    vectorizer = build_vectorizer()
    features = vectorizer.fit_transform(["great product", "bad product"])
    assert features.shape[0] == 2
    assert features.shape[1] > 0
