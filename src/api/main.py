from fastapi import FastAPI
from .health import healthcheck
from .middleware import configure_middleware
from .routes import router

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")
configure_middleware(app)
app.include_router(router, prefix="/api/v1", tags=["prediction"])
app.get("/health")(healthcheck)
