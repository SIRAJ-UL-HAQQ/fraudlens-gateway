# FraudLens Gateway — Architecture Specification

## 1. System Architecture & Layered Design

FraudLens Gateway is structured into strict horizontal layers to maintain clean separation of concerns, testability, and deterministic behavior:

```text
                     +---------------------------------------+
                     |         Frontend / API Client         |
                     +---------------------------------------+
                                         |  HTTP / JSON / JWT
                                         v
                     +---------------------------------------+
                     |               api/                    |
                     |  Routes, Serializers, HTTP Protocols  |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |            services/                  |
                     |  Scoring, Explainer, Heuristic Rules  |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |               db/                     |
                     | SQLAlchemy 2, Repositories, Migrations|
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |          PostgreSQL / SQLite          |
                     +---------------------------------------+
```

### Layer Constraints
- **`app/api/` (Presentation)**:
  - Handles HTTP status codes, request parsing, authentication headers, and response serialization.
  - Contains **no** business logic, scoring formulas, or direct database queries.
- **`app/services/` (Domain Logic)**:
  - Heuristic safety rules (`rules.py`), ML scoring orchestration (`scoring.py`), and SHAP attribution (`explainer.py`).
  - Pure functions and deterministic fallbacks.
  - Does not handle HTTP requests or HTTP exceptions.
- **`app/db/` (Persistence)**:
  - SQLAlchemy 2 declarative models (`models.py`), database session lifecycle (`session.py`), and repository pattern implementations (`repository.py`).
  - All database interactions are gated behind the abstract `TransactionRepository` interface.

---

## 2. Audit Trail & Immutability (SRS FR-10)

Compliance and financial fraud auditability require an immutable record of all transaction scores and analyst interventions:

1. **Dual Audit Trail Components**:
   - **`transactions` table**: Every transaction scored through `POST /api/v1/predict` is immediately persisted with its full input features, risk score, triggered rule flags, explanation status, top SHAP attributions, model version, and creation timestamp.
   - **`analyst_reviews` table**: Every analyst review submitted via `POST /api/v1/transactions/{id}/review` is appended as an independent record with an immutable UUID, review outcome (`Confirmed Fraud` or `False Positive`), review notes, reviewer username, and review timestamp.

2. **Immutable Guarantee**:
   - `UPDATE` and `DELETE` queries on both `transactions` and `analyst_reviews` are forbidden.
   - SQLAlchemy `before_update` and `before_delete` event listeners intercept any attempt to mutate or drop existing records, raising an `ImmutableRecordException`.
   - The repository exposes no methods for editing past records.

---

## 3. Resilience & Failure Modes

- **Database Down (503 Service Unavailable)**:
  - All database operations catch connection errors. If the database cannot be reached or a transaction cannot be persisted, the gateway returns `503 Service Unavailable`.
  - A decision is **never** returned across the network if it cannot be durably committed to the audit trail.
- **Model Fallback**:
  - In `dev` or `test` modes, if `MODEL_PATH` is not configured, the gateway falls back to a deterministic `StubExplainer` and stub scorer clearly stamped with `model_version="stub"`.
  - In `prod` mode, startup strictly requires a valid model bundle matching `docs/ml-artifact-contract.md`; missing artifacts immediately abort server startup.
