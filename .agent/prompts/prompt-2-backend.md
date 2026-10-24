# Prompt 2: Backend

ROLE: Backend engineer.  
MODE: Planning, then implement in small commits.

GOAL: Implement the FastAPI service behind the existing contract (docs/api-contract.md, schemas.py). Do not change the contract without telling me.

CONSTRAINTS (also copy into .agent/rules/10-backend.md):
- Layers: api/ (routes only) -> services/ (logic) -> db/ (persistence). No logic in routes.
- Config via pydantic-settings from environment variables; no hardcoded paths or secrets.
- The model path comes from env. If the artifact is missing, load a deterministic STUB
  model (clearly labelled model_version="stub") so dev and CI work without the real model.
- Audit log is append-only: no UPDATE or DELETE on it, ever.
- If the DB is down return 503 with a clear message, never a wrong classification.

TASKS:
1. App skeleton: main.py, CORS for the frontend origin, structured logging,
   error handlers, GET /health (checks DB connectivity).
2. DB: SQLAlchemy 2 models for transactions and audit_log, Alembic migrations,
   session management.
3. services/rules.py: pure functions, thresholds in config. Rules suited to PaySim:
   large TRANSFER/CASH_OUT amount; origin account drained to zero by the transaction;
   balance inconsistency (new_balance_dest - old_balance_dest != amount);
   very high amount for the type. Return a list of rule flag names.
4. services/scoring.py: preprocess -> model probability -> SHAP top features ->
   reliability status -> gateway decision:
   rules or high score => candidate High Risk; High Risk + Unstable => Analyst Review;
   below the low threshold => Low Risk. Thresholds in config, documented.
5. Persist each decision and its explanation to transactions + audit_log.
6. Endpoints: POST /predict, GET /transactions, GET /transactions/{id},
   POST /transactions/{id}/review, GET /metrics (reads ml/reports/metrics.json),
   GET /export (CSV first, PDF second).
7. Latency middleware: record per-request ms; expose p50/p95 via /metrics.
8. Auth: simple JWT login, one seeded analyst user from env vars, all data
   endpoints protected.
9. Tests (pytest): every rule with boundary values, decision-matrix tests covering
   every score x reliability combination, audit-log immutability, DB-down behaviour,
   malformed input, zero-amount and negative-amount transactions.

ACCEPTANCE: pytest green; OpenAPI at /docs matches docs/openapi.json; coverage report
included for services/. Report anything you could not verify.
