# Contributing to FraudLens Gateway

Welcome to the FraudLens Gateway repository. We follow strict engineering practices to maintain code quality and reproducible research.

## Branch and PR Guidelines
1. **Feature Branches Only**:
   - Always branch off `main`: `git checkout -b feature/<descriptive-name>`.
   - Never commit directly to `main`.
   - Never force-push (`git push --force`) to any shared or primary branch.
2. **Implementation Planning**:
   - For any task modifying **3 or more files**, draft an implementation plan first and get approval before writing code.
3. **No Secrets or Binary Model Artifacts**:
   - Never commit `.env` files, credentials, or API keys.
   - Never commit raw datasets (PaySim `.csv`) or serialized model binary artifacts (`*.joblib`, `*.pkl`, `ml/models/**`).
   - Tracked documentation and evaluation metrics JSON files (e.g. `docs/openapi.json`, `ml/reports/*.json`) are committed as part of code review.
   - Use `.env.example` as the canonical reference for environment variables.
4. **API Contract Integrity**:
   - `backend/app/models/schemas.py` and `docs/api-contract.md` are the dual sources of truth.
   - Any API schema change requires updating both documents, the tests, and the frontend client together.
5. **Merge & Branch Policy**:
   - **Squash Merging Only**: Only squash merging is permitted; standard merge commits and rebase merging are disabled.
   - **Head Branch Cleanup**: Automatic deletion of head branches upon merge is enabled.
   - **Auto-Merge Policy**: Auto-merge is disabled (`Allow auto-merge` unchecked). All pull requests require explicit human review after checks pass.
6. **Required CI Status Checks on `main`**:
   - The following four status checks must pass before merging to `main`:
     - `backend` (lint, OpenAPI sync, pytest + coverage >= 85%)
     - `ml` (lint, ML unit & stability tests)
     - `frontend` (lint, production Vite build)
     - `docker` (multi-stage image build, compose up, healthcheck, smoke test)
   - *(Note: In GitHub repository settings under `Branches` > `Branch protection rules`, status checks only become selectable in the search list after the CI workflow has executed at least once).*

## Development Workflow

### Local Setup
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Build and run via Docker Compose:
   ```bash
   make up
   # Or on Windows PowerShell:
   docker compose up -d --build --wait
   ```
3. Run smoke tests:
   ```bash
   make smoke
   # Or on Windows:
   bash scripts/smoke_test.sh
   ```

### Code Style & Quality Checks
Run local checks before opening a pull request:
```bash
make lint
make test
```
Or run individual linters:
- Backend & ML: `ruff check backend/ ml/`
- Frontend: `cd frontend && npm run lint`
- Backend tests: `PYTHONPATH=backend pytest backend/app/tests -v`
- ML tests: `PYTHONPATH=backend:. pytest ml/tests -v`
