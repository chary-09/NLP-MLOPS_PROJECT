from pathlib import Path
import json


def get_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
