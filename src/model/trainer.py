from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.nlp.feature_extraction import build_vectorizer
from src.nlp.preprocessor import clean_text


def train_models(texts, labels, max_features: int = 5000):
    """Train the three Phase 1 candidate classifiers."""
    vectorizer = build_vectorizer(max_features)
    features = vectorizer.fit_transform([clean_text(text) for text in texts])
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(random_state=42),
    }
    models = {}
    for name, model in candidates.items():
        model.fit(features, labels)
        models[name] = model
    return models, vectorizer


def train_model(texts, labels, max_features: int = 5000):
    """Backward-compatible helper that returns the logistic model."""
    models, vectorizer = train_models(texts, labels, max_features)
    return models["logistic_regression"], vectorizer
