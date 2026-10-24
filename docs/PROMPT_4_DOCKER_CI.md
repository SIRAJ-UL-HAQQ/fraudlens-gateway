# Prompt 4: Docker and CI

ROLE: DevOps engineer.  
MODE: Planning.

GOAL: One-command local run and a trustworthy CI pipeline.

CONSTRAINTS (also copy into .agent/rules/40-devops.md):
- Minimal on purpose: 3 services (frontend, backend, postgres). No Kubernetes.
- No secrets in images or in git. Everything from .env; commit only .env.example.
- Model artifacts and data are NOT in git and NOT baked into images. The backend mounts
  ml/models read-only; if absent, the backend falls back to the stub model.
- CI must pass on a clean checkout with no model and no dataset.

TASKS:
1. backend/Dockerfile: python slim, multi-stage, non-root user, healthcheck.
   frontend/Dockerfile: build stage (node) + static serve stage (nginx).
   .dockerignore for each.
2. docker-compose.yml: postgres with a named volume and pg_isready healthcheck;
   backend depends_on postgres (service_healthy), runs migrations on start;
   frontend depends_on backend; ports and env from .env.
3. .env.example documenting every variable.
4. Makefile: up, down, test, lint, smoke, logs.
5. .github/workflows/ci.yml on push and on pull_request:
   - backend job: install with pip cache, ruff, pytest with coverage.
   - frontend job: npm ci, lint, build.
   - docker job: build both images, `docker compose up -d`, wait for health, run
     scripts/smoke_test.sh, always dump logs on failure, `docker compose down -v`.
6. .github/pull_request_template.md and a CONTRIBUTING.md summarising the branch
   and PR rules. Do NOT enable auto-merge.
7. A short docs/RUNNING.md: the exact steps from a clean clone to a working demo.

ACCEPTANCE: from a clean clone, `cp .env.example .env && make up` gives a working app;
CI green on a test PR. Tell me every manual step that remains and any GitHub settings
I must enable myself (branch protection, required checks).
