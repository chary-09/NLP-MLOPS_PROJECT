def detect_language(text: str) -> str:
    """Minimal safe default; replace with a language-ID model if required."""
    return "en" if text and text.isascii() else "unknown"
