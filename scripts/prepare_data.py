"""Convert the downloaded IMDb folder into the Phase 1 raw CSV."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, ROOT_DIR


def load_imdb_reviews(dataset_dir: Path) -> pd.DataFrame:
    """Read labeled IMDb reviews, ignoring the unlabeled ``train/unsup`` data."""
    records = []
    for split in ("train", "test"):
        for label in ("pos", "neg"):
            review_dir = dataset_dir / split / label
            if not review_dir.is_dir():
                raise FileNotFoundError(f"Missing IMDb directory: {review_dir}")
            for review_path in sorted(review_dir.glob("*.txt")):
                records.append(
                    {
                        "text": review_path.read_text(encoding="utf-8", errors="replace"),
                        "sentiment": "positive" if label == "pos" else "negative",
                    }
                )
    return pd.DataFrame(records, columns=["text", "sentiment"])


def main() -> None:
    dataset_dir = ROOT_DIR / "storage" / "aclImdb"
    output_path = DATA_DIR / "raw" / "imdb_reviews.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviews = load_imdb_reviews(dataset_dir)
    reviews.to_csv(output_path, index=False)
    print(f"Loaded {len(reviews):,} labeled reviews", flush=True)
    print(f"Raw dataset saved to {output_path}")


if __name__ == "__main__":
    main()