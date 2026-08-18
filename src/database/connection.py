from sqlalchemy import create_engine
from src.config import DATABASE_URL


def get_engine(url: str = DATABASE_URL):
    return create_engine(url)
