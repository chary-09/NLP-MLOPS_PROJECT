from .tokenizer import tokenize


def text_statistics(text: str) -> dict[str, int]:
    tokens = tokenize(text)
    return {"characters": len(text), "words": len(tokens), "unique_words": len(set(tokens))}
