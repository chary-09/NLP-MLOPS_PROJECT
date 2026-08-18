def model_version(metadata: dict) -> str:
    return metadata.get("model_version", "unknown")
