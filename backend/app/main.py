from contextlib import asynccontextmanager
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import settings
from app.db.repository import DatabaseUnavailableError
from app.db.session import check_db_connection
from app.models.schemas import HealthResponse, ModelInfo
from app.services.scoring import scoring_service

# Structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraudlens.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: load ML artifact bundle once at startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} in environment: '{settings.APP_ENV}'")
    # Verify or initialize ML model artifact per contract
    scoring_service.load_artifact(settings.MODEL_PATH)
    logger.info(f"Model initialized: mode={scoring_service.mode}, version={scoring_service.model_version}")
    yield
    logger.info("Shutting down FraudLens Gateway.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Real-Time Fraud Detection Gateway with Explainable AI (SHAP) & Hybrid Rules",
    lifespan=lifespan,
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

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def latency_and_logging_middleware(request: Request, call_next):
    """Middleware recording per-request latency. Never logs passwords, tokens, or request bodies."""
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} ({duration_ms:.2f}ms)"
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(f"{request.method} {request.url.path} failed after {duration_ms:.2f}ms: {exc}")
        raise exc


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_exception_handler(request: Request, exc: DatabaseUnavailableError):
    """Return clean HTTP 503 when database connectivity is unavailable."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database service is currently unavailable. Operation aborted to preserve audit integrity."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
)
def health_check() -> JSONResponse:
    """Return health status including live database connectivity and ML model metadata."""
    is_db_connected = check_db_connection()
    model_info = ModelInfo(
        mode=scoring_service.mode,
        version=scoring_service.model_version,
    )

    if not is_db_connected:
        response_data = HealthResponse(
            status="unhealthy",
            service=settings.PROJECT_NAME,
            version="1.0.0",
            database="disconnected",
            model=model_info,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_data.model_dump(),
        )

    response_data = HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version="1.0.0",
        database="connected",
        model=model_info,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_data.model_dump(),
    )


app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
