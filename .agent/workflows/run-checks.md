---
description: Run ruff, pytest, frontend lint and build, report a pass/fail table, and fix only failures caused by your own changes.
---

# Workflow: /run-checks

Run ruff, pytest, frontend lint and build, report a pass/fail table, and fix only failures caused by your own changes.

## Execution Steps

1. **Backend Linting**:
   - Run `ruff` across the backend and ml packages:
     ```bash
     ruff check backend/ ml/
     ```

2. **Backend Unit & Contract Tests**:
   - Run `pytest` with coverage:
     ```bash
     pytest backend/ -v --tb=short
     ```

3. **Frontend Lint & Build**:
   - Run linting and production build verification:
     ```bash
     cd frontend && npm run lint && npm run build
     ```

4. **Result Reporting Table**:
   - Compile and output a clear status table:
     | Check | Command | Status | Notes / Failures |
     |---|---|---|---|
     | Ruff Lint | `ruff check backend/ ml/` | PASS / FAIL | Details |
     | Pytest | `pytest backend/` | PASS / FAIL | Tests run / failed |
     | Frontend Lint | `npm run lint` | PASS / FAIL | Issues found |
     | Frontend Build | `npm run build` | PASS / FAIL | Output status |

6. **Honest Reporting & Remediation**:
   - Report results honestly, including failures. Never claim tests pass without running them.
   - Fix **only** failures caused by your own changes; do not modify unrelated code or mask pre-existing failures without explicit user consent.
