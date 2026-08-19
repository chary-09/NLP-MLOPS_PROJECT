import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.predictor import SentimentPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for text")
    parser.add_argument("text", nargs="?", default="I really love this product!")
    args = parser.parse_args()
    result = SentimentPredictor().predict(args.text)
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.0%}")


if __name__ == "__main__":
    main()
