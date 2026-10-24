---
description: Update README, docs/api-contract.md and docs/ARCHITECTURE.md to match the current code, and list inconsistencies.
---

# Workflow: /update-docs

Update README, docs/api-contract.md and docs/ARCHITECTURE.md to match the current code. List anything that was inconsistent.

## Execution Steps

1. **Audit Current State**:
   - Inspect `backend/app/models/schemas.py` and API route definitions in `backend/app/api/`.
   - Inspect frontend client (`frontend/src/api/client.js`) and UI mocks (`frontend/src/mocks/fixtures.js`).
   - Inspect ML pipelines, scripts, and reporting outputs.
   - Inspect `docker-compose.yml`, environment configuration, and service definitions.

2. **Synchronize Documentation**:
   - **`docs/api-contract.md`**: Ensure request/response schemas, field names (snake_case), enum values, and endpoint signatures strictly match `backend/app/models/schemas.py`.
   - **`docs/ARCHITECTURE.md`**: Update layer boundaries, data flow (Ingestion -> Heuristic Rules -> XGBoost Scoring -> SHAP Local Attribution -> Persistence / Audit Log), and service interactions.
   - **`README.md`**: Update project overview, quick start instructions, directory tree, and CLI commands.

3. **Inconsistency Log**:
   - Finish with an explicit bulleted list detailing:
     - Discrepancies identified between documentation and code.
     - Files updated to resolve the discrepancies.
     - Any unresolved ambiguities requiring team consensus.
