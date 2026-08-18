def should_alert(status: str) -> bool:
    return status != "healthy"
