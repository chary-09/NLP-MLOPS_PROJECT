import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, PredictionRecord
from src.database.repository import PredictionRepository
from src.database.migrations import create_tables, drop_tables


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_database_table_creation():
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    assert "predictions" in Base.metadata.tables
    drop_tables(engine)


def test_prediction_repository_create_and_get(test_db_session):
    repo = PredictionRepository(test_db_session)
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    record = repo.create(
        prediction_id=pred_id,
        input_text="The product is outstanding!",
        sentiment="positive",
        confidence=0.9523,
        model_version="0.1.0",
        timestamp=now,
    )

    assert record.prediction_id == pred_id
    assert record.input_text == "The product is outstanding!"
    assert record.sentiment == "positive"
    assert record.confidence == 0.9523
    assert record.model_version == "0.1.0"

    fetched = repo.get_by_id(pred_id)
    assert fetched is not None
    assert fetched.prediction_id == pred_id
    assert fetched.input_text == "The product is outstanding!"
    assert fetched.sentiment == "positive"


def test_prediction_repository_list_recent_pagination(test_db_session):
    repo = PredictionRepository(test_db_session)

    # Insert 5 records
    for i in range(5):
        repo.create(
            prediction_id=f"id-{i}",
            input_text=f"Review text {i}",
            sentiment="positive" if i % 2 == 0 else "negative",
            confidence=0.85 + (i * 0.02),
            model_version="0.1.0",
        )

    # Fetch with limit=2, offset=0
    records, total = repo.list_recent(limit=2, offset=0)
    assert total == 5
    assert len(records) == 2

    # Fetch offset=2, limit=2
    records_p2, total_p2 = repo.list_recent(limit=2, offset=2)
    assert total_p2 == 5
    assert len(records_p2) == 2
    assert records_p2[0].prediction_id != records[0].prediction_id


def test_prediction_repository_metrics_summary(test_db_session):
    repo = PredictionRepository(test_db_session)

    # Empty DB
    summary = repo.get_metrics_summary()
    assert summary["total_predictions"] == 0
    assert summary["average_confidence"] == 0.0

    # Add records
    repo.create("p1", "good", "positive", 0.90, "0.1.0")
    repo.create("p2", "great", "positive", 0.80, "0.1.0")
    repo.create("p3", "bad", "negative", 0.70, "0.1.0")

    summary = repo.get_metrics_summary()
    assert summary["total_predictions"] == 3
    assert summary["sentiment_distribution"]["positive"] == 2
    assert summary["sentiment_distribution"]["negative"] == 1
    assert summary["average_confidence"] == 0.80


def test_database_persistence_across_sessions(tmp_path):
    db_file = tmp_path / "test_persist.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    # Session 1: Write
    session1 = Session()
    repo1 = PredictionRepository(session1)
    repo1.create("persist-123", "Persistent review text", "positive", 0.92, "0.1.0")
    session1.close()

    # Session 2: Read
    session2 = Session()
    repo2 = PredictionRepository(session2)
    record = repo2.get_by_id("persist-123")
    assert record is not None
    assert record.input_text == "Persistent review text"
    assert record.sentiment == "positive"
    session2.close()
