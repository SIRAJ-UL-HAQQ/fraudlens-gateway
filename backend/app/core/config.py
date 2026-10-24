import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "FraudLens Gateway"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
    # Model Thresholds
    FRAUD_THRESHOLD_HIGH: float = 0.80
    FRAUD_THRESHOLD_REVIEW: float = 0.50
    
    # Model Artifact Path
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ml/models/xgboost_fraud_model.joblib")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./fraudlens.db")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"
    ]

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")

settings = Settings()
