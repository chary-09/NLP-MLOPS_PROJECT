from .preprocessor import clean_text


def tokenize(text: str) -> list[str]:
    """Return whitespace-separated tokens from cleaned text."""
    return clean_text(text).split()
