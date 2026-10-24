from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.models.schemas import HealthResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Real-Time Fraud Detection Gateway with Explainable AI (SHAP) & Hybrid Rules",
    openapi_tags=[
        {"name": "Health", "description": "Liveness and health checks"},
        {"name": "Authentication", "description": "Analyst login and token management"},
        {"name": "Prediction", "description": "Transaction fraud scoring and explanation"},
        {"name": "Transactions", "description": "Query and view scored transactions"},
        {"name": "Reviews", "description": "Analyst audit reviews and feedback"},
        {"name": "Metrics", "description": "Operational and model evaluation metrics"},
        {"name": "Export", "description": "Audit and compliance data export"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
)
def health_check() -> HealthResponse:
    """Return health status of the FraudLens Gateway."""
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version="1.0.0",
        database="connected",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
