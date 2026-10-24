# Running FraudLens Gateway & Development Commands

## 1. Import Convention & Working Directory

This project standardizes on the **`app.*`** package namespace:
```python
from app.models.schemas import TransactionIn, PredictionOut
from app.db.repository import get_repository
```

### Execution Rules
- **Backend Directory Context**: Run backend commands from inside the `backend/` folder:
  ```bash
  cd backend
  python -m pytest app/tests -v
  ```
- **Root Directory Context**: If executing from the repository root, supply `PYTHONPATH=backend`:
  - **Linux / macOS / Git Bash**:
    ```bash
    PYTHONPATH=backend python -m pytest backend/app/tests -v
    ```
  - **Windows PowerShell**:
    ```powershell
    $env:PYTHONPATH="backend"; python -m pytest backend/app/tests -v
    ```

---

## 2. Common Development Commands

### Running Backend Tests
```bash
# From backend directory:
cd backend
python -m pytest app/tests -v --tb=short
```

### Regenerating OpenAPI Specification
When changes are made to `backend/app/models/schemas.py` or route definitions:
```bash
python scripts/generate_openapi.py
```
This updates [`docs/openapi.json`](file:///docs/openapi.json).

### Starting the Local Development Gateway
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`

### Running the Integration Smoke Test
With the gateway running (e.g. on `http://localhost:8000`):
```bash
bash scripts/smoke_test.sh http://localhost:8000
```

---

## 3. Makefile Targets

A `Makefile` is provided in the repository root for convenience:
- `make test`: Run backend contract and unit tests.
- `make openapi`: Regenerate `docs/openapi.json`.
- `make run`: Launch the backend development server.
- `make smoke`: Run the integration smoke test against `http://localhost:8000`.
