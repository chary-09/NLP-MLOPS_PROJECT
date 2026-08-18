from sklearn.linear_model import LogisticRegression
from src.nlp.feature_extraction import build_vectorizer


def train_model(texts, labels, max_features: int = 5000):
    vectorizer = build_vectorizer(max_features)
    features = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)
    return model, vectorizer
