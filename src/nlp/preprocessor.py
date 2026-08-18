import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove URLs from text."""
    text = re.sub(r"https?://\S+", "", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()
