# DevOps, Docker, and CI Conventions and Constraints

## Architecture & Container Services
- Keep the local stack minimal on purpose: 3 services only (`frontend`, `backend`, `postgres`). No Kubernetes.
- The stack must spin up with one command (`make up` or `docker compose up -d --wait`) from a clean `.env.example`.
- Frontend runs Nginx serving built static assets on port 5173 and reverse-proxying `/api/` and `/health` to backend:8000.
- Backend runs FastAPI with Uvicorn on port 8000, using multi-stage builds and a non-root user (`appuser`).
- Database runs PostgreSQL 16 Alpine with named volume persistence and `pg_isready` healthcheck.
- Host port bindings must strictly bind to `127.0.0.1` (`127.0.0.1:8000:8000`, `127.0.0.1:5173:5173`).

## Secrets & Configuration
- Never commit secrets, passwords, tokens, or environment-specific values to git.
- Never bake secrets or API credentials into container images.
- All configuration values originate from `.env`, documented completely in `.env.example`.
- Passwords must be validated as non-empty in `docker-compose.yml` (`?` parameter expansion).

## Artifacts & Datasets
- Dataset files (`data/`, `ml/data/`) and model artifacts (`ml/models/`) are NOT tracked in git and NOT baked into Docker images.
- Backend mounts `./ml/models` read-only (`:ro`). If model artifacts are absent, backend falls back automatically to the deterministic stub model.
- Research benchmark metrics are opt-in via read-only mount `./ml/reports:/app/ml/reports:ro`. In stub mode without reports, `ml_metrics_status` reports `"unavailable"`.

## CI Pipeline Guarantees
- CI must pass cleanly on a fresh checkout with no trained models and no dataset.
- All GitHub Actions must use pinned SHAs with verified Node 24 runtime support.
- Python dependencies in Linux Docker images and Ubuntu CI jobs must be locked via `backend/constraints-linux.txt`.
- Smoke tests must verify end-to-end contract compliance across backend (:8000) and frontend proxy (:5173).
