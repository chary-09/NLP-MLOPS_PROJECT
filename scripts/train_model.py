import pandas as pd
from src.config import DATA_DIR, MODEL_DIR
from src.model.model_utils import save_artifact
from src.model.trainer import train_model

data = pd.read_csv(DATA_DIR / "processed" / "train.csv")
model, vectorizer = train_model(data["text"], data["sentiment"])
save_artifact(model, MODEL_DIR / "sentiment_model.pkl")
save_artifact(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
print("Model artifacts saved.")
