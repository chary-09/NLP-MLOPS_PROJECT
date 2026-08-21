"""Prepare, train, compare, evaluate, and save the best sentiment model."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, MODEL_DIR
from src.model.evaluator import evaluate_model
from src.model.model_utils import save_artifact
from src.model.trainer import train_models
from src.nlp.preprocessor import clean_text


RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
	"""Load the generated IMDb CSV, with the small legacy CSV as a fallback."""
	dataset_path = DATA_DIR / "raw" / "imdb_reviews.csv"
	if not dataset_path.exists():
		dataset_path = DATA_DIR / "raw" / "sentiment.csv"
	data = pd.read_csv(dataset_path)
	required_columns = {"text", "sentiment"}
	if not required_columns.issubset(data.columns):
		raise ValueError(f"{dataset_path} must contain 'text' and 'sentiment' columns")
	return data.dropna(subset=["text", "sentiment"]).reset_index(drop=True)


def split_dataset(data: pd.DataFrame):
	"""Return stratified 70% train, 15% validation, and 15% test partitions."""
	train, remainder = train_test_split(
		data, test_size=0.30, random_state=RANDOM_STATE, stratify=data["sentiment"]
	)
	validation, test = train_test_split(
		remainder, test_size=0.50, random_state=RANDOM_STATE, stratify=remainder["sentiment"]
	)
	return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def save_processed_splits(train, validation, test) -> None:
	processed_dir = DATA_DIR / "processed"
	processed_dir.mkdir(parents=True, exist_ok=True)
	for name, split in (("train", train), ("validation", validation), ("test", test)):
		cleaned = split.copy()
		cleaned["text"] = cleaned["text"].map(clean_text)
		cleaned.to_csv(processed_dir / f"{name}.csv", index=False)


def main() -> None:
	data = load_dataset()
	train, validation, test = split_dataset(data)
	save_processed_splits(train, validation, test)
	models, vectorizer = train_models(train["text"], train["sentiment"])
	evaluations = {
		name: evaluate_model(model, vectorizer, test["text"], test["sentiment"])
		for name, model in models.items()
	}
	best_name = max(evaluations, key=lambda name: evaluations[name]["f1_score"])
	save_artifact(models[best_name], MODEL_DIR / "sentiment_model.pkl")
	save_artifact(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
		json.dump({"best_model": best_name, "models": evaluations}, file, indent=2)

	print(f"Best model: {best_name}")
	for name, metrics in evaluations.items():
		print(
			f"{name}: accuracy={metrics['accuracy']:.3f}, "
			f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
			f"f1={metrics['f1_score']:.3f}"
		)
	print(f"Artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
	main()
