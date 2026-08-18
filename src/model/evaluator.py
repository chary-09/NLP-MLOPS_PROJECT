from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, vectorizer, texts, labels) -> dict:
    predictions = model.predict(vectorizer.transform(texts))
    return {"accuracy": float(accuracy_score(labels, predictions)), "report": classification_report(labels, predictions, output_dict=True)}
