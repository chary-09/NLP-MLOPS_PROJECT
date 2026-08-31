import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.migrations import create_tables
from src.database.connection import engine
from src.config import DATABASE_URL


def main():
    print(f"Initializing database tables at: {DATABASE_URL}")
    create_tables(engine)
    print("Database setup complete: 'predictions' table is ready.")


if __name__ == "__main__":
    main()
