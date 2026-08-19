"""Train, compare, evaluate, and save the best sentiment model."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, MODEL_DIR
from src.model.evaluator import evaluate_model
from src.model.model_utils import save_artifact
from src.model.trainer import train_models


def main() -> None:
	data = pd.read_csv(DATA_DIR / "raw" / "sentiment.csv")
	required_columns = {"text", "sentiment"}
	if not required_columns.issubset(data.columns):
		raise ValueError("sentiment.csv must contain 'text' and 'sentiment' columns")

	models, vectorizer = train_models(data["text"], data["sentiment"])
	evaluations = {
		name: evaluate_model(model, vectorizer, data["text"], data["sentiment"])
		for name, model in models.items()
	}
	best_name = max(evaluations, key=lambda name: evaluations[name]["f1_score"])
	save_artifact(models[best_name], MODEL_DIR / "sentiment_model.pkl")
	save_artifact(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
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
