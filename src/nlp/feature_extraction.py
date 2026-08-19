from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used by training and prediction."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def fit_transform(texts, max_features: int = 5000):
    """Fit a TF-IDF vectorizer and transform cleaned text."""
    vectorizer = build_vectorizer(max_features)
    return vectorizer.fit_transform(texts), vectorizer
