# Backend Conventions and Constraints

ROLE: Backend engineer.
MODE: Planning, then implement in small commits.
GOAL: Implement the FastAPI service behind the existing contract (docs/api-contract.md, schemas.py). Do not change the contract without telling me.

## Constraints
- Layers: `api/` (routes only) -> `services/` (logic) -> `db/` (persistence). No logic in routes.
- Config via `pydantic-settings` from environment variables; no hardcoded paths or secrets.
- The model path comes from env. If the artifact is missing, load a deterministic STUB model (clearly labelled `model_version="stub"`) so dev and CI work without the real model.
- Audit log is append-only: no UPDATE or DELETE on it, ever.
- If the DB is down return 503 with a clear message, never a wrong classification.
- Keep functions small, add type hints and docstrings, no dead code.
