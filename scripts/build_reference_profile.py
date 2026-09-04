import json
from pathlib import Path
import pandas as pd

def build_profile():
    train_path = Path("data/processed/train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path, nrows=2000)
        texts = df["text"].dropna().astype(str).tolist()
    else:
        texts = ["This is a good movie review with typical length." * 10] * 50

    lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    stats = {
        "dataset_name": "IMDb Training Corpus (train.csv)",
        "sample_count": len(texts),
        "mean_char_length": round(sum(lengths) / len(lengths), 2),
        "mean_word_count": round(sum(word_counts) / len(word_counts), 2),
        "sample_char_lengths": lengths[:500],
        "sample_word_counts": word_counts[:500],
    }
    out_dir = Path("data/reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reference_profile.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Generated reference_profile.json with {len(texts)} samples.")

if __name__ == "__main__":
    build_profile()
