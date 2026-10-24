import csv
from datetime import UTC, datetime
import io
import json
import math
from pathlib import Path
from typing import Any, List, Tuple
import uuid

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import AnalystReviewModel, TransactionRecord
from app.db.session import SessionLocal
from app.models.schemas import (
    AnalystReviewIn,
    AnalystReviewRecord,
    ExplanationStatus,
    ExportFormat,
    FeatureExplanation,
    GatewayDecision,
    MetricsSummary,
    PaginatedTransactionsResponse,
    PredictionOut,
    ReviewOutcome,
    ShapDirection,
    TransactionDetail,
    TransactionIn,
    TransactionSummary,
    TransactionType,
)


class DatabaseUnavailableError(Exception):
    """Raised when database connection or execution fails, triggering HTTP 503."""
    pass


def sanitize_csv_cell(value: Any) -> str:
    """Neutralize formula injection by prepending single quote to sensitive leading characters."""
    s = str(value) if value is not None else ""
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


def load_ml_reports_metrics() -> Tuple[dict | None, dict | None]:
    """Read ML benchmark metrics from ml/reports/metrics.json if present, else fallback to defaults."""
    metrics_path = Path(settings.METRICS_PATH)
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("strategy_comparison"), data.get("evaluation_metrics")
        except Exception:
            pass

    # Default fallback benchmarks conforming to docs/api-contract.md
    strategy_comparison = {
        "none": {"precision": 0.88, "recall": 0.62, "f1": 0.73, "pr_auc": 0.74, "stability": 0.71},
        "class_weight": {"precision": 0.91, "recall": 0.84, "f1": 0.87, "pr_auc": 0.89, "stability": 0.88},
        "smote_enn": {"precision": 0.85, "recall": 0.87, "f1": 0.86, "pr_auc": 0.87, "stability": 0.80},
    }
    evaluation_metrics = {
        "precision": 0.91,
        "recall": 0.84,
        "f1": 0.87,
        "pr_auc": 0.89,
        "mcc": 0.86,
    }
    return strategy_comparison, evaluation_metrics


def generate_pdf_export(items: List[TransactionSummary]) -> bytes:
    """Generate a clean, styled PDF document using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("<b>FraudLens Gateway — Transaction Export Report</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    summary_text = (
        f"Exported at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
        f"Total Records Included: {len(items)}"
    )
    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Spacer(1, 14))

    table_data = [
        ["Tx ID", "Step", "Type", "Amount", "Decision", "Status", "Score", "Time (UTC)"]
    ]
    for it in items:
        table_data.append([
            it.transaction_id[:8] + "...",
            str(it.step),
            it.type.value,
            f"{it.amount:,.2f}",
            it.gateway_decision.value,
            it.explanation_status.value,
            f"{it.risk_score:.2f}",
            it.created_at.strftime("%Y-%m-%d %H:%M"),
        ])

    col_widths = [65, 35, 60, 75, 80, 60, 45, 110]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


class TransactionRepository:
    """Abstract interface for transaction persistence and retrieval."""

    def save_transaction(
        self,
        transaction_in: TransactionIn,
        prediction: PredictionOut,
        created_at: datetime | None = None,
    ) -> TransactionDetail:
        raise NotImplementedError

    def get_transaction(self, transaction_id: str) -> TransactionDetail | None:
        raise NotImplementedError

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
        raise NotImplementedError

    def add_review(
        self,
        transaction_id: str,
        review_in: AnalystReviewIn,
        reviewed_by: str,
    ) -> AnalystReviewRecord:
        raise NotImplementedError

    def get_metrics(self) -> MetricsSummary:
        raise NotImplementedError

    def export_transactions(
        self,
        format: ExportFormat,
        step_min: int | None = None,
        step_max: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        decision: GatewayDecision | None = None,
        explanation_status: ExplanationStatus | None = None,
    ) -> Tuple[bytes | str, str, str]:
        raise NotImplementedError


class InMemoryTransactionRepository(TransactionRepository):
    """In-memory stub repository for tests and isolated verification."""

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

        # Deterministic ordering: created_at desc, transaction_id asc
        filtered.sort(key=lambda t: (-t.created_at.timestamp(), t.transaction_id))

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
        strat_comp, eval_metrics = load_ml_reports_metrics()

        if total == 0:
            return MetricsSummary(
                total_transactions=0,
                decision_distribution=distribution,
                explanation_stability_ratio=1.0,
                average_risk_score=0.0,
                latency_p50_ms=12.0,
                latency_p95_ms=25.0,
                strategy_comparison=strat_comp,
                evaluation_metrics=eval_metrics,
            )

        stable_count = 0
        total_risk = 0.0
        latencies: List[float] = []

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
            strategy_comparison=strat_comp,
            evaluation_metrics=eval_metrics,
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
    ) -> Tuple[bytes | str, str, str]:
        page = self.list_transactions(
            step_min=step_min,
            step_max=step_max,
            created_from=created_from,
            created_to=created_to,
            decision=decision,
            explanation_status=explanation_status,
            page=1,
            page_size=100000,
        )

        if len(page.items) > settings.EXPORT_MAX_ROWS:
            raise ValueError(
                f"Export matching {len(page.items)} records exceeds safety cap of "
                f"{settings.EXPORT_MAX_ROWS} rows. Please narrow your query filters."
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
            for it in page.items:
                writer.writerow([
                    sanitize_csv_cell(it.transaction_id),
                    sanitize_csv_cell(it.step),
                    sanitize_csv_cell(it.type.value),
                    sanitize_csv_cell(it.amount),
                    sanitize_csv_cell(it.name_orig),
                    sanitize_csv_cell(it.name_dest),
                    sanitize_csv_cell(it.risk_score),
                    sanitize_csv_cell(it.gateway_decision.value),
                    sanitize_csv_cell(it.explanation_status.value),
                    sanitize_csv_cell(";".join(it.rule_flags)),
                    sanitize_csv_cell(it.created_at.isoformat()),
                ])
            return output.getvalue(), "text/csv; charset=utf-8", "transactions_export.csv"
        else:
            pdf_bytes = generate_pdf_export(page.items)
            return pdf_bytes, "application/pdf", "transactions_export.pdf"


class SqlAlchemyTransactionRepository(TransactionRepository):
    """Production PostgreSQL/SQLite transaction repository implementing persistence."""

    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self.session_factory = session_factory or SessionLocal

    def _get_session(self) -> Session:
        try:
            return self.session_factory()
        except SQLAlchemyError as e:
            raise DatabaseUnavailableError("Failed to acquire database session") from e

    def save_transaction(
        self,
        transaction_in: TransactionIn,
        prediction: PredictionOut,
        created_at: datetime | None = None,
    ) -> TransactionDetail:
        now = created_at or datetime.now(UTC)
        top_feats_json = [f.model_dump() for f in prediction.top_features]

        db = self._get_session()
        try:
            rec = TransactionRecord(
                transaction_id=prediction.transaction_id,
                step=transaction_in.step,
                type=transaction_in.type.value,
                amount=transaction_in.amount,
                name_orig=transaction_in.name_orig,
                old_balance_orig=transaction_in.old_balance_orig,
                new_balance_orig=transaction_in.new_balance_orig,
                name_dest=transaction_in.name_dest,
                old_balance_dest=transaction_in.old_balance_dest,
                new_balance_dest=transaction_in.new_balance_dest,
                risk_score=prediction.risk_score,
                gateway_decision=prediction.gateway_decision.value,
                explanation_status=prediction.explanation_status.value,
                rule_flags=prediction.rule_flags,
                top_features=top_feats_json,
                imbalance_strategy=prediction.imbalance_strategy,
                model_version=prediction.model_version,
                latency_ms=prediction.latency_ms,
                created_at=now,
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)

            return TransactionDetail(
                transaction_id=rec.transaction_id,
                transaction_data=transaction_in,
                prediction=prediction,
                created_at=rec.created_at,
                reviews=[],
                latest_review=None,
            )
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseUnavailableError(f"Database write failed: {e}") from e
        finally:
            db.close()

    def get_transaction(self, transaction_id: str) -> TransactionDetail | None:
        db = self._get_session()
        try:
            rec = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == transaction_id).first()
            if not rec:
                return None

            tx_in = TransactionIn(
                step=rec.step,
                type=TransactionType(rec.type),
                amount=rec.amount,
                name_orig=rec.name_orig,
                old_balance_orig=rec.old_balance_orig,
                new_balance_orig=rec.new_balance_orig,
                name_dest=rec.name_dest,
                old_balance_dest=rec.old_balance_dest,
                new_balance_dest=rec.new_balance_dest,
            )

            top_features = [
                FeatureExplanation(
                    feature=f["feature"],
                    shap_value=f["shap_value"],
                    direction=ShapDirection(f["direction"]),
                )
                for f in (rec.top_features or [])
            ]

            pred_out = PredictionOut(
                transaction_id=rec.transaction_id,
                risk_score=rec.risk_score,
                top_features=top_features,
                explanation_status=ExplanationStatus(rec.explanation_status),
                gateway_decision=GatewayDecision(rec.gateway_decision),
                rule_flags=rec.rule_flags or [],
                imbalance_strategy=rec.imbalance_strategy,
                model_version=rec.model_version,
                latency_ms=rec.latency_ms,
            )

            reviews = [
                AnalystReviewRecord(
                    review_id=r.review_id,
                    transaction_id=r.transaction_id,
                    outcome=ReviewOutcome(r.outcome),
                    note=r.note,
                    reviewed_by=r.reviewed_by,
                    reviewed_at=r.reviewed_at,
                )
                for r in (rec.reviews or [])
            ]

            latest_rev = reviews[-1] if reviews else None

            return TransactionDetail(
                transaction_id=rec.transaction_id,
                transaction_data=tx_in,
                prediction=pred_out,
                created_at=rec.created_at,
                reviews=reviews,
                latest_review=latest_rev,
            )
        except SQLAlchemyError as e:
            raise DatabaseUnavailableError(f"Database query failed: {e}") from e
        finally:
            db.close()

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
        db = self._get_session()
        try:
            query = db.query(TransactionRecord)

            if step_min is not None:
                query = query.filter(TransactionRecord.step >= step_min)
            if step_max is not None:
                query = query.filter(TransactionRecord.step <= step_max)
            if created_from is not None:
                query = query.filter(TransactionRecord.created_at >= created_from)
            if created_to is not None:
                query = query.filter(TransactionRecord.created_at <= created_to)
            if decision is not None:
                query = query.filter(TransactionRecord.gateway_decision == decision.value)
            if explanation_status is not None:
                query = query.filter(TransactionRecord.explanation_status == explanation_status.value)

            total = query.count()
            total_pages = math.ceil(total / page_size) if total > 0 else 0

            # Deterministic ordering: created_at desc, transaction_id asc
            query = query.order_by(TransactionRecord.created_at.desc(), TransactionRecord.transaction_id.asc())

            offset = (page - 1) * page_size
            records = query.offset(offset).limit(page_size).all()

            items = [
                TransactionSummary(
                    transaction_id=r.transaction_id,
                    step=r.step,
                    type=TransactionType(r.type),
                    amount=r.amount,
                    name_orig=r.name_orig,
                    name_dest=r.name_dest,
                    risk_score=r.risk_score,
                    gateway_decision=GatewayDecision(r.gateway_decision),
                    explanation_status=ExplanationStatus(r.explanation_status),
                    rule_flags=r.rule_flags or [],
                    created_at=r.created_at,
                    reviewed=bool(r.reviews),
                )
                for r in records
            ]

            return PaginatedTransactionsResponse(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            )
        except SQLAlchemyError as e:
            raise DatabaseUnavailableError(f"Database query failed: {e}") from e
        finally:
            db.close()

    def add_review(
        self,
        transaction_id: str,
        review_in: AnalystReviewIn,
        reviewed_by: str,
    ) -> AnalystReviewRecord:
        db = self._get_session()
        try:
            tx = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == transaction_id).first()
            if not tx:
                raise KeyError(f"Transaction {transaction_id} not found")

            rev = AnalystReviewModel(
                review_id=str(uuid.uuid4()),
                transaction_id=transaction_id,
                outcome=review_in.outcome.value,
                note=review_in.note,
                reviewed_by=reviewed_by,
                reviewed_at=datetime.now(UTC),
            )
            db.add(rev)
            db.commit()
            db.refresh(rev)

            return AnalystReviewRecord(
                review_id=rev.review_id,
                transaction_id=rev.transaction_id,
                outcome=ReviewOutcome(rev.outcome),
                note=rev.note,
                reviewed_by=rev.reviewed_by,
                reviewed_at=rev.reviewed_at,
            )
        except KeyError:
            raise
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseUnavailableError(f"Database review write failed: {e}") from e
        finally:
            db.close()

    def get_metrics(self) -> MetricsSummary:
        db = self._get_session()
        try:
            total = db.query(TransactionRecord).count()
            strat_comp, eval_metrics = load_ml_reports_metrics()

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
                    strategy_comparison=strat_comp,
                    evaluation_metrics=eval_metrics,
                )

            # Decision distribution counts
            dec_counts = (
                db.query(TransactionRecord.gateway_decision, func.count(TransactionRecord.transaction_id))
                .group_by(TransactionRecord.gateway_decision)
                .all()
            )
            for dec, count in dec_counts:
                distribution[dec] = count

            # Stability count
            stable_count = (
                db.query(TransactionRecord)
                .filter(TransactionRecord.explanation_status == ExplanationStatus.STABLE.value)
                .count()
            )

            # Average risk score
            avg_risk = db.query(func.avg(TransactionRecord.risk_score)).scalar() or 0.0

            # p50 / p95 latency directly from stored latency_ms of scored transactions
            latencies = [row[0] for row in db.query(TransactionRecord.latency_ms).order_by(TransactionRecord.latency_ms.asc()).all()]
            p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
            p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

            return MetricsSummary(
                total_transactions=total,
                decision_distribution=distribution,
                explanation_stability_ratio=round(stable_count / total, 3),
                average_risk_score=round(float(avg_risk), 3),
                latency_p50_ms=round(float(p50), 2),
                latency_p95_ms=round(float(p95), 2),
                strategy_comparison=strat_comp,
                evaluation_metrics=eval_metrics,
            )
        except SQLAlchemyError as e:
            raise DatabaseUnavailableError(f"Database metrics query failed: {e}") from e
        finally:
            db.close()

    def export_transactions(
        self,
        format: ExportFormat,
        step_min: int | None = None,
        step_max: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        decision: GatewayDecision | None = None,
        explanation_status: ExplanationStatus | None = None,
    ) -> Tuple[bytes | str, str, str]:
        page = self.list_transactions(
            step_min=step_min,
            step_max=step_max,
            created_from=created_from,
            created_to=created_to,
            decision=decision,
            explanation_status=explanation_status,
            page=1,
            page_size=100000,
        )

        if len(page.items) > settings.EXPORT_MAX_ROWS:
            raise ValueError(
                f"Export matching {len(page.items)} records exceeds safety cap of "
                f"{settings.EXPORT_MAX_ROWS} rows. Please narrow your query filters."
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
            for it in page.items:
                writer.writerow([
                    sanitize_csv_cell(it.transaction_id),
                    sanitize_csv_cell(it.step),
                    sanitize_csv_cell(it.type.value),
                    sanitize_csv_cell(it.amount),
                    sanitize_csv_cell(it.name_orig),
                    sanitize_csv_cell(it.name_dest),
                    sanitize_csv_cell(it.risk_score),
                    sanitize_csv_cell(it.gateway_decision.value),
                    sanitize_csv_cell(it.explanation_status.value),
                    sanitize_csv_cell(";".join(it.rule_flags)),
                    sanitize_csv_cell(it.created_at.isoformat()),
                ])
            return output.getvalue(), "text/csv; charset=utf-8", "transactions_export.csv"
        else:
            pdf_bytes = generate_pdf_export(page.items)
            return pdf_bytes, "application/pdf", "transactions_export.pdf"


# Default repository instance
_repository: TransactionRepository = SqlAlchemyTransactionRepository()


def get_repository() -> TransactionRepository:
    """Dependency injector returning active repository."""
    return _repository


def set_repository(repo: TransactionRepository) -> None:
    """Setter for test harnesses or runtime swapping."""
    global _repository
    _repository = repo
