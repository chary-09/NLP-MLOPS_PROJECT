from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.nlp.preprocessor import clean_text


def evaluate_model(model, vectorizer, texts, labels) -> dict:
    cleaned_texts = [clean_text(text) for text in texts]
    predictions = model.predict(vectorizer.transform(cleaned_texts))
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
