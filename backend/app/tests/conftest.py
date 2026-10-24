import pytest

from app.db.session import Base, engine
from app.services.scoring import scoring_service


@pytest.fixture(autouse=True, scope="session")
def setup_test_database():
    """Ensure database tables exist for integration and contract tests."""
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(autouse=True)
def load_test_stub_model(request):
    """Load stub model for tests unless opted out with @pytest.mark.unloaded_model.

    Teardown always restores the explicit unloaded state.
    """
    if "unloaded_model" in request.keywords:
        scoring_service.reset()
        yield
        scoring_service.reset()
    else:
        scoring_service.load_artifact(None)
        yield
        scoring_service.reset()
