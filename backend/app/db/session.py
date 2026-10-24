import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def check_db_connection() -> bool:
    """Verify live database connectivity by running a lightweight ping query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database connectivity check failed: {e}")
        return False


def get_db():
    """FastAPI dependency yielding managed DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
