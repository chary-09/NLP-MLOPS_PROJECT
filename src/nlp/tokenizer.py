from .preprocessor import clean_text


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()
