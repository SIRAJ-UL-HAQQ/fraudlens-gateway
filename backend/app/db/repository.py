import csv
import io
import math
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from app.models.schemas import (
    AnalystReviewIn,
    AnalystReviewRecord,
    ExplanationStatus,
    ExportFormat,
    GatewayDecision,
    MetricsSummary,
    PaginatedTransactionsResponse,
    PredictionOut,
    TransactionDetail,
    TransactionIn,
    TransactionSummary,
)


class TransactionRepository(ABC):
    """Abstract interface for transaction persistence and retrieval."""

    @abstractmethod
    def save_transaction(
        self,
        transaction_in: TransactionIn,
        prediction: PredictionOut,
        created_at: datetime | None = None,
    ) -> TransactionDetail:
        """Persist a newly scored transaction."""

    @abstractmethod
    def get_transaction(self, transaction_id: str) -> TransactionDetail | None:
        """Fetch a transaction by ID."""

    @abstractmethod
    def list_transactions(
        self,
        step_min: int | None = None,
        step_max: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        decision: GatewayDecision | None = None,
        explanation_status: ExplanationStatus | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> PaginatedTransactionsResponse:
        """List transactions with filtering and pagination."""

    @abstractmethod
    def add_review(
        self,
        transaction_id: str,
        review_in: AnalystReviewIn,
        reviewed_by: str,
    ) -> AnalystReviewRecord:
        """Append an analyst review to the immutable audit trail."""

    @abstractmethod
    def get_metrics(self) -> MetricsSummary:
        """Calculate system and model operational metrics."""

    @abstractmethod
    def export_transactions(
        self,
        format: ExportFormat,
        step_min: int | None = None,
        step_max: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        decision: GatewayDecision | None = None,
        explanation_status: ExplanationStatus | None = None,
    ) -> tuple[str, str, str]:
        """Export filtered transactions. Returns (content, media_type, filename)."""


class InMemoryTransactionRepository(TransactionRepository):
    """In-memory stub repository for contract testing and local dev."""

    def __init__(self) -> None:
        self._transactions: dict[str, TransactionDetail] = {}

    def save_transaction(
        self,
        transaction_in: TransactionIn,
        prediction: PredictionOut,
        created_at: datetime | None = None,
    ) -> TransactionDetail:
        now = created_at or datetime.now(UTC)
        detail = TransactionDetail(
            transaction_id=prediction.transaction_id,
            transaction_data=transaction_in,
            prediction=prediction,
            created_at=now,
            reviews=[],
            latest_review=None,
        )
        self._transactions[prediction.transaction_id] = detail
        return detail

    def get_transaction(self, transaction_id: str) -> TransactionDetail | None:
        return self._transactions.get(transaction_id)

    def list_transactions(
        self,
        step_min: int | None = None,
        step_max: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        decision: GatewayDecision | None = None,
        explanation_status: ExplanationStatus | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> PaginatedTransactionsResponse:
        filtered = list(self._transactions.values())

        if step_min is not None:
            filtered = [t for t in filtered if t.transaction_data.step >= step_min]
        if step_max is not None:
            filtered = [t for t in filtered if t.transaction_data.step <= step_max]
        if created_from is not None:
            filtered = [t for t in filtered if t.created_at >= created_from]
        if created_to is not None:
            filtered = [t for t in filtered if t.created_at <= created_to]
        if decision is not None:
            filtered = [t for t in filtered if t.prediction.gateway_decision == decision]
        if explanation_status is not None:
            filtered = [t for t in filtered if t.prediction.explanation_status == explanation_status]

        # Sort descending by created_at
        filtered.sort(key=lambda t: t.created_at, reverse=True)

        total = len(filtered)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start = (page - 1) * page_size
        end = start + page_size
        sliced = filtered[start:end]

        items = [
            TransactionSummary(
                transaction_id=t.transaction_id,
                step=t.transaction_data.step,
                type=t.transaction_data.type,
                amount=t.transaction_data.amount,
                name_orig=t.transaction_data.name_orig,
                name_dest=t.transaction_data.name_dest,
                risk_score=t.prediction.risk_score,
                gateway_decision=t.prediction.gateway_decision,
                explanation_status=t.prediction.explanation_status,
                rule_flags=t.prediction.rule_flags,
                created_at=t.created_at,
                reviewed=bool(t.reviews),
            )
            for t in sliced
        ]

        return PaginatedTransactionsResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def add_review(
        self,
        transaction_id: str,
        review_in: AnalystReviewIn,
        reviewed_by: str,
    ) -> AnalystReviewRecord:
        tx = self._transactions.get(transaction_id)
        if not tx:
            raise KeyError(f"Transaction {transaction_id} not found")

        record = AnalystReviewRecord(
            review_id=str(uuid.uuid4()),
            transaction_id=transaction_id,
            outcome=review_in.outcome,
            note=review_in.note,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.now(UTC),
        )
        tx.reviews.append(record)
        tx.latest_review = record
        return record

    def get_metrics(self) -> MetricsSummary:
        records = list(self._transactions.values())
        total = len(records)
        distribution = {
            GatewayDecision.LOW_RISK.value: 0,
            GatewayDecision.HIGH_RISK.value: 0,
            GatewayDecision.ANALYST_REVIEW.value: 0,
        }

        if total == 0:
            return MetricsSummary(
                total_transactions=0,
                decision_distribution=distribution,
                explanation_stability_ratio=1.0,
                average_risk_score=0.0,
                latency_p50_ms=12.0,
                latency_p95_ms=25.0,
                strategy_comparison={
                    "none": {"precision": 0.88, "recall": 0.62, "f1": 0.73, "pr_auc": 0.74, "stability": 0.71},
                    "class_weight": {"precision": 0.91, "recall": 0.84, "f1": 0.87, "pr_auc": 0.89, "stability": 0.88},
                    "smote_enn": {"precision": 0.85, "recall": 0.87, "f1": 0.86, "pr_auc": 0.87, "stability": 0.80},
                },
                evaluation_metrics={
                    "precision": 0.91,
                    "recall": 0.84,
                    "f1": 0.87,
                    "pr_auc": 0.89,
                    "mcc": 0.86,
                },
            )

        stable_count = 0
        total_risk = 0.0
        latencies: list[float] = []

        for r in records:
            dec = r.prediction.gateway_decision.value
            distribution[dec] = distribution.get(dec, 0) + 1
            if r.prediction.explanation_status == ExplanationStatus.STABLE:
                stable_count += 1
            total_risk += r.prediction.risk_score
            latencies.append(r.prediction.latency_ms)

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]

        return MetricsSummary(
            total_transactions=total,
            decision_distribution=distribution,
            explanation_stability_ratio=round(stable_count / total, 3),
            average_risk_score=round(total_risk / total, 3),
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            strategy_comparison={
                "none": {"precision": 0.88, "recall": 0.62, "f1": 0.73, "pr_auc": 0.74, "stability": 0.71},
                "class_weight": {"precision": 0.91, "recall": 0.84, "f1": 0.87, "pr_auc": 0.89, "stability": 0.88},
                "smote_enn": {"precision": 0.85, "recall": 0.87, "f1": 0.86, "pr_auc": 0.87, "stability": 0.80},
            },
            evaluation_metrics={
                "precision": 0.91,
                "recall": 0.84,
                "f1": 0.87,
                "pr_auc": 0.89,
                "mcc": 0.86,
            },
        )

    def export_transactions(
        self,
        format: ExportFormat,
        step_min: int | None = None,
        step_max: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        decision: GatewayDecision | None = None,
        explanation_status: ExplanationStatus | None = None,
    ) -> tuple[str, str, str]:
        page = self.list_transactions(
            step_min=step_min,
            step_max=step_max,
            created_from=created_from,
            created_to=created_to,
            decision=decision,
            explanation_status=explanation_status,
            page=1,
            page_size=10000,
        )

        if format == ExportFormat.CSV:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "Transaction ID",
                "Step",
                "Type",
                "Amount",
                "Origin",
                "Destination",
                "Risk Score",
                "Gateway Decision",
                "Explanation Status",
                "Rule Flags",
                "Created At",
            ])
            for item in page.items:
                writer.writerow([
                    item.transaction_id,
                    item.step,
                    item.type.value,
                    item.amount,
                    item.name_orig,
                    item.name_dest,
                    item.risk_score,
                    item.gateway_decision.value,
                    item.explanation_status.value,
                    ";".join(item.rule_flags),
                    item.created_at.isoformat(),
                ])
            return output.getvalue(), "text/csv", "transactions_export.csv"

        elif format == ExportFormat.PDF:
            # Minimal text-formatted PDF stub conforming to PDF 1.4 spec
            pdf_lines = [
                "%PDF-1.4",
                "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
                "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
                "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >> endobj",
                f"4 0 obj << /Length 55 >> stream\nBT /F1 12 Tf 72 712 Td (FraudLens Export: {len(page.items)} records) Tj ET\nendstream\nendobj",
                "xref",
                "0 5",
                "0000000000 65535 f ",
                "0000000009 00000 n ",
                "0000000058 00000 n ",
                "0000000115 00000 n ",
                "0000000219 00000 n ",
                "trailer << /Size 5 /Root 1 0 R >>",
                "startxref",
                "328",
                "%%EOF",
            ]
            return "\n".join(pdf_lines), "application/pdf", "transactions_export.pdf"


# Repository instance singleton
_repository = InMemoryTransactionRepository()


def get_repository() -> TransactionRepository:
    """Dependency injector for transaction repository."""
    return _repository
