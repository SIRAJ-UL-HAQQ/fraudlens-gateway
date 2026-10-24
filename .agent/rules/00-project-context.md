---
trigger: always_on
always_on: true
---

# FraudLens Gateway: project context (always apply)

Project: Explanation-reliable financial fraud detection under class imbalance.
University Software Engineering lab project, 4-person team, 5th semester.
Scope is deliberately moderate. Prefer simple, readable, well-tested code over clever code.

Research question: does the SHAP explanation for a fraud alert stay stable across
imbalance-handling strategies (no resampling, class weighting, SMOTE/SMOTE-ENN)?
Gateway decisions: Low Risk | High Risk | Analyst Review.
A High Risk prediction with an Unstable explanation is routed to Analyst Review.

Dataset: PaySim (Kaggle, "Synthetic Financial Datasets For Fraud Detection").
Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
Never hardcode the fraud ratio; compute and report it from the data.

Stack: Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic, PostgreSQL,
scikit-learn, XGBoost, imbalanced-learn, SHAP, Optuna, React (Vite) + Tailwind + Recharts,
Docker Compose, GitHub Actions, pytest.

Hard rules:
- Chronological split by `step`. Never a random split. Resample the TRAIN set only.
- Never commit datasets, model artifacts, .env files, or secrets.
- backend/app/models/schemas.py and docs/api-contract.md are the API source of truth.
  Changing either requires updating the other, the tests, and the frontend client.
- Work only on a feature branch (feature/<name>). Never commit to main. Never force-push.
- For any task touching 3+ files: produce an implementation plan and wait for approval.
- Before finishing: run the project checks (see /run-checks) and report results honestly,
  including failures. Do not claim tests pass without running them.
- Keep functions small, add type hints and docstrings, no dead code.
