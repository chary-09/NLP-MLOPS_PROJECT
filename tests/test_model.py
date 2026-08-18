from src.model.trainer import train_model


def test_train_model():
    model, vectorizer = train_model(["great", "bad", "okay"], ["positive", "negative", "neutral"])
    assert model.predict(vectorizer.transform(["great"]))[0] == "positive"
