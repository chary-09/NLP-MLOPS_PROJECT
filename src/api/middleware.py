from fastapi import FastAPI


def configure_middleware(app: FastAPI) -> None:
    """Register cross-cutting middleware in one place."""
