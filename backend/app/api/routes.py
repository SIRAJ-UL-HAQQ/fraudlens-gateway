from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.core.auth import create_access_token, get_current_analyst, verify_password
from app.core.config import settings
from app.db.repository import DatabaseUnavailableError, TransactionRepository, get_repository
from app.models.schemas import (
    AnalystReviewIn,
    AnalystReviewRecord,
    ExplanationStatus,
    ExportFormat,
    GatewayDecision,
    LoginRequest,
    MetricsSummary,
    PaginatedTransactionsResponse,
    PredictionOut,
    TokenResponse,
    TransactionDetail,
    TransactionIn,
)
from app.services.scoring import scoring_service

router = APIRouter()


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Analyst login",
)
def login(credentials: LoginRequest) -> TokenResponse:
    """Authenticate an analyst and issue a JWT bearer token."""
    if not credentials.username or not credentials.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required",
        )
    if credentials.username != settings.ANALYST_USERNAME or not verify_password(credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = create_access_token(
        data={"sub": credentials.username, "username": credentials.username, "role": "analyst"}
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRES_IN_SECONDS,
        username=credentials.username,
        role="analyst",
    )


@router.post(
    "/predict",
    response_model=PredictionOut,
    tags=["Prediction"],
    summary="Score a transaction",
)
def predict(
    payload: TransactionIn,
    repo: Annotated[TransactionRepository, Depends(get_repository)],
    user: Annotated[dict, Depends(get_current_analyst)],
) -> PredictionOut:
    """Score incoming transaction, compute SHAP attribution, and persist audit record before responding.

    If database write fails, returns HTTP 503 without exposing an unpersisted decision.
    """
    prediction = scoring_service.predict(payload)
    try:
        repo.save_transaction(payload, prediction)
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence failed. Transaction scoring aborted to preserve audit integrity.",
        ) from e
    return prediction


@router.get(
    "/transactions",
    response_model=PaginatedTransactionsResponse,
    tags=["Transactions"],
    summary="List transactions with filters and pagination",
)
def list_transactions(
    repo: Annotated[TransactionRepository, Depends(get_repository)],
    user: Annotated[dict, Depends(get_current_analyst)],
    step_min: Annotated[int | None, Query(ge=1, description="Minimum PaySim step (>=1)")] = None,
    step_max: Annotated[int | None, Query(ge=1, description="Maximum PaySim step (>=1)")] = None,
    created_from: Annotated[datetime | None, Query(description="Filter by scoring ISO datetime (from)")] = None,
    created_to: Annotated[datetime | None, Query(description="Filter by scoring ISO datetime (to)")] = None,
    decision: Annotated[GatewayDecision | None, Query(description="Filter by Gateway Decision")] = None,
    explanation_status: Annotated[ExplanationStatus | None, Query(description="Filter by explanation stability")] = None,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="Items per page")] = 50,
) -> PaginatedTransactionsResponse:
    """Query transactions using dataset time (step) and system scoring time (created_at)."""
    try:
        return repo.list_transactions(
            step_min=step_min,
            step_max=step_max,
            created_from=created_from,
            created_to=created_to,
            decision=decision,
            explanation_status=explanation_status,
            page=page,
            page_size=page_size,
        )
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionDetail,
    tags=["Transactions"],
    summary="Get transaction details and explanation",
)
def get_transaction(
    transaction_id: str,
    repo: Annotated[TransactionRepository, Depends(get_repository)],
    user: Annotated[dict, Depends(get_current_analyst)],
) -> TransactionDetail:
    """Retrieve full transaction record including SHAP attribution and audit reviews."""
    try:
        tx = repo.get_transaction(transaction_id)
        if not tx:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction '{transaction_id}' not found",
            )
        return tx
    except HTTPException:
        raise
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


@router.post(
    "/transactions/{transaction_id}/review",
    response_model=AnalystReviewRecord,
    tags=["Reviews"],
    summary="Submit analyst review",
)
def submit_review(
    transaction_id: str,
    review_in: AnalystReviewIn,
    repo: Annotated[TransactionRepository, Depends(get_repository)],
    user: Annotated[dict, Depends(get_current_analyst)],
) -> AnalystReviewRecord:
    """Append a human review to the transaction's immutable audit log."""
    try:
        return repo.add_review(
            transaction_id=transaction_id,
            review_in=review_in,
            reviewed_by=user.get("username", settings.ANALYST_USERNAME),
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{transaction_id}' not found",
        )
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


@router.get(
    "/metrics",
    response_model=MetricsSummary,
    tags=["Metrics"],
    summary="Operational and ML performance metrics",
)
def get_metrics(
    repo: Annotated[TransactionRepository, Depends(get_repository)],
    user: Annotated[dict, Depends(get_current_analyst)],
) -> MetricsSummary:
    """Retrieve system latency, decision breakdown, and ML benchmark metrics."""
    try:
        return repo.get_metrics()
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


@router.get(
    "/export",
    tags=["Export"],
    summary="Export transactions as CSV or PDF",
)
def export_transactions(
    repo: Annotated[TransactionRepository, Depends(get_repository)],
    user: Annotated[dict, Depends(get_current_analyst)],
    format: Annotated[ExportFormat, Query(description="Export file format")] = ExportFormat.CSV,
    step_min: Annotated[int | None, Query(ge=1, description="Minimum step filter")] = None,
    step_max: Annotated[int | None, Query(ge=1, description="Maximum step filter")] = None,
    created_from: Annotated[datetime | None, Query(description="Scoring time from")] = None,
    created_to: Annotated[datetime | None, Query(description="Scoring time to")] = None,
    decision: Annotated[GatewayDecision | None, Query(description="Gateway decision filter")] = None,
    explanation_status: Annotated[ExplanationStatus | None, Query(description="Explanation stability filter")] = None,
) -> Response:
    """Export filtered transaction records."""
    try:
        content, media_type, filename = repo.export_transactions(
            format=format,
            step_min=step_min,
            step_max=step_max,
            created_from=created_from,
            created_to=created_to,
            decision=decision,
            explanation_status=explanation_status,
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e
