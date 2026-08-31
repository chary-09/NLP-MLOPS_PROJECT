from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import DATABASE_URL, DATA_DIR


def get_engine(url: str = DATABASE_URL):
    """Create SQLAlchemy engine with appropriate dialect settings."""
    if url.startswith("sqlite"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return create_engine(url, echo=False)


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI database session dependency yielding a managed session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
