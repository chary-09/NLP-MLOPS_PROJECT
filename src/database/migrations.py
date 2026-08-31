import logging
from sqlalchemy import Engine
from .connection import engine, get_engine
from .models import Base

logger = logging.getLogger(__name__)


def create_tables(bind_engine: Engine = None) -> None:
    """Create all configured database tables if they do not exist."""
    target_engine = bind_engine or engine
    logger.info(f"Creating database tables with engine: {target_engine.url}")
    Base.metadata.create_all(bind=target_engine)
    logger.info("Database tables verified / created successfully.")


def drop_tables(bind_engine: Engine = None) -> None:
    """Drop all tables (primarily used for test environment teardown)."""
    target_engine = bind_engine or engine
    logger.info(f"Dropping database tables with engine: {target_engine.url}")
    Base.metadata.drop_all(bind=target_engine)
