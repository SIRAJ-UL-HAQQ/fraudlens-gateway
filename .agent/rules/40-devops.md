# DevOps, Docker, and CI Conventions and Constraints

ROLE: DevOps engineer.
MODE: Planning.
GOAL: One-command local run and a trustworthy CI pipeline.

## Constraints
- Minimal on purpose: 3 services (frontend, backend, postgres). No Kubernetes.
- No secrets in images or in git. Everything from `.env`; commit only `.env.example`.
- Model artifacts and data are NOT in git and NOT baked into images. The backend mounts `ml/models` read-only; if absent, the backend falls back to the stub model.
- CI must pass on a clean checkout with no model and no dataset.
- Never commit datasets, model artifacts, .env files, or secrets.
