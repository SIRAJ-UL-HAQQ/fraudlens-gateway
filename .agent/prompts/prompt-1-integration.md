# Prompt 1: Integration (contract first)

ROLE: Integration engineer for FraudLens Gateway.  
MODE: Planning. Produce a plan first and wait for my approval.

GOAL: Define the system contract so backend, frontend and ML can be built in parallel without breaking each other.

TASKS:
1. Write backend/app/models/schemas.py (Pydantic v2):
   - TransactionIn: step:int, type: enum{PAYMENT,TRANSFER,CASH_OUT,DEBIT,CASH_IN},
     amount, name_orig, old_balance_orig, new_balance_orig, name_dest,
     old_balance_dest, new_balance_dest. The API uses snake_case; PaySim's raw names
     (oldbalanceOrg, newbalanceOrig, ...) are mapped in ONE function in the ML layer.
   - PredictionOut: transaction_id, risk_score (0..1), top_features
     [{feature, shap_value, direction}], explanation_status enum{Stable,Unstable},
     gateway_decision enum{Low Risk,High Risk,Analyst Review}, rule_flags [str],
     imbalance_strategy, model_version, latency_ms.
   - AnalystReview (outcome enum{Confirmed Fraud,False Positive}, note).
2. Write docs/api-contract.md listing every endpoint with request/response examples:
   POST /predict, GET /transactions (filters + pagination), GET /transactions/{id},
   POST /transactions/{id}/review (appends to audit log, never overwrites),
   GET /metrics, GET /export?format=csv|pdf&from&to, GET /health, POST /auth/login.
3. Generate docs/openapi.json from the FastAPI app, and a script to regenerate it.
4. Create scripts/smoke_test.sh: with the stack running, check /health, POST a sample
   transaction to /predict, confirm it appears in GET /transactions, confirm the
   response matches the schema. Exit non-zero on any failure.
5. Create frontend/src/mocks/fixtures.js with realistic example responses for every
   endpoint, matching the contract exactly.
6. Add a contract test in backend/app/tests/ that validates responses against the schema.

ACCEPTANCE: schemas import cleanly; contract test passes; docs match code.  
OUT OF SCOPE: real model, real database logic, UI components.  
FINISH WITH: a short list of design decisions I should confirm with my team.
