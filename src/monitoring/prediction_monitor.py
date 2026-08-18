def monitor_prediction(result: dict) -> dict:
    return {"sentiment": result["sentiment"], "confidence": result["confidence"]}
