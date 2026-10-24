import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for FraudLens Gateway via pydantic-settings."""

    PROJECT_NAME: str = "FraudLens Gateway"
    API_V1_STR: str = "/api/v1"
    APP_ENV: str = Field(default_factory=lambda: os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "dev")).lower())
    
    # Database
    DATABASE_URL: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./fraudlens.db"))
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"
    ]
    
    # Authentication & Secrets (No default values in production)
    JWT_SECRET_KEY: str | None = None
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRES_IN_SECONDS: int = 3600
    ANALYST_USERNAME: str = "analyst_1"
    ANALYST_PASSWORD: str | None = None
    
    # Model Artifact Configuration
    MODEL_PATH: str | None = Field(default_factory=lambda: os.getenv("MODEL_PATH", None))
    METRICS_PATH: str = "ml/reports/metrics.json"
    
    # ML Scoring Thresholds (Defaults for stub only; manifest.json overrides when present)
    FRAUD_THRESHOLD_HIGH: float = 0.70
    FRAUD_THRESHOLD_REVIEW: float = 0.40
    
    # Provisional heuristic rule thresholds (to be finalized by data teammate from train split)
    RULE_LARGE_TRANSFER_THRESHOLD: float = 200_000.0  # PROVISIONAL
    RULE_DRAIN_BALANCE_RATIO: float = 0.90           # PROVISIONAL
    RULE_HIGH_AMOUNT_BY_TYPE: dict[str, float] = {   # PROVISIONAL
        "TRANSFER": 500_000.0,
        "CASH_OUT": 300_000.0,
        "PAYMENT": 50_000.0,
        "DEBIT": 10_000.0,
        "CASH_IN": 500_000.0,
    }
    
    # Export Limits
    EXPORT_MAX_ROWS: int = 5000

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """In prod mode, require secure secrets and reject placeholders."""
        env = self.APP_ENV.lower()
        if env in ["prod", "production"]:
            # Validate JWT_SECRET_KEY
            if not self.JWT_SECRET_KEY:
                raise ValueError("JWT_SECRET_KEY is required in production environment.")
            key_lower = self.JWT_SECRET_KEY.lower()
            if any(p in key_lower for p in ["change_me", "placeholder", "secret", "default", "example"]):
                raise ValueError("JWT_SECRET_KEY cannot use placeholder or default values in production.")

            # Validate ANALYST_PASSWORD
            if not self.ANALYST_PASSWORD:
                raise ValueError("ANALYST_PASSWORD is required in production environment.")
            pwd_lower = self.ANALYST_PASSWORD.lower()
            if any(p in pwd_lower for p in ["change_me", "placeholder", "password", "default", "example"]):
                raise ValueError("ANALYST_PASSWORD cannot use placeholder or default values in production.")
        return self

    def get_jwt_secret_key(self) -> str:
        """Return JWT secret key, providing safe dev default only in non-prod."""
        if self.JWT_SECRET_KEY:
            return self.JWT_SECRET_KEY
        if self.APP_ENV.lower() in ["dev", "development", "test"]:
            return "fraudlens-insecure-dev-jwt-secret-key-minimum-32-chars-long"
        raise ValueError("JWT_SECRET_KEY is not configured for non-dev environment.")

    def get_analyst_password(self) -> str:
        """Return analyst password, providing safe dev default only in non-prod."""
        if self.ANALYST_PASSWORD:
            return self.ANALYST_PASSWORD
        if self.APP_ENV.lower() in ["dev", "development", "test"]:
            return os.getenv("ANALYST_PASSWORD", "dev_local_analyst_password_2026")
        raise ValueError("ANALYST_PASSWORD is not configured for non-dev environment.")


settings = Settings()
