from pathlib import Path
import joblib


def save_artifact(artifact: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> object:
    return joblib.load(path)
