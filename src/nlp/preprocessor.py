"""Beginner-friendly text cleaning utilities."""

import re


def clean_text(text: str) -> str:
    """Lowercase text, remove HTML and symbols, and normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
