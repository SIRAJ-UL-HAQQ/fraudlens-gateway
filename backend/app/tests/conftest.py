import pytest

from app.db.session import Base, engine


@pytest.fixture(autouse=True, scope="session")
def setup_test_database():
    """Ensure database tables exist for integration and contract tests."""
    Base.metadata.create_all(bind=engine)
    yield
