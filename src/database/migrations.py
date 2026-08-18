from .connection import get_engine
from .models import Base


def create_tables() -> None:
    Base.metadata.create_all(get_engine())
