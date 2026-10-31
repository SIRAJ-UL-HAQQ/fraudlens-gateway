# Running FraudLens Gateway & Development Commands

## 1. Quick Start: Clean Clone to Working Demo

### One-Command Launch (Linux / macOS / Git Bash)
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Build and launch services in background
make up

# 3. Verify services with smoke test
make smoke

# 4. View logs or stop
make logs
make down
```

### Windows PowerShell Equivalent (No Make Required)
```powershell
# 1. Copy environment template
Copy-Item .env.example .env

# 2. Build and launch services in background
docker compose up -d --build --wait

# 3. Verify services with smoke test (via Git Bash)
& "C:\Program Files\Git\bin\bash.exe" scripts/smoke_test.sh

# 4. View logs or stop
docker compose logs -f
docker compose down -v
```

> [!NOTE]
> All host ports are strictly bound to `127.0.0.1`:
> - Backend API & Docs: `http://127.0.0.1:8000/docs`
> - Health Check: `http://127.0.0.1:8000/health`
> - Frontend UI & Reverse Proxy: `http://127.0.0.1:5173/`

> [!WARNING]
> **Credential Security**:
> Default placeholder passwords in `.env.example` are intended solely for local offline evaluation. In shared or production deployments, generate cryptographically random credentials. Passwords used in `DATABASE_URL` must be strictly URL-safe (avoid unencoded URI reserved characters such as `@`, `:`, `/`).

> [!IMPORTANT]
> **Known Frontend UI Limitation**:
> The frontend container builds successfully and delivers static assets (`index.html` with `<div id="root"></div>` and bundled JS). However, `frontend/index.html` loads `/src/App.jsx` directly without calling `ReactDOM.createRoot()`. Consequently, the browser renders an empty DOM root shell. Static delivery returns HTTP 200, but interactive rendering is blocked pending frontend entrypoint wiring.

---

## 2. Model Configuration & Fallback to Stub

FraudLens Gateway is designed to run in zero-dependency environments where model binary artifacts are not present:
- **Default Stub Mode**: When `MODEL_PATH` is left empty in `.env`, the backend falls back automatically to an internal deterministic heuristic model (`mode: "stub"`). No dataset or trained model files are required.
- **Trained Model Mounting**: Trained model artifacts located in `ml/models/<strategy>` are mounted into the backend container read-only at `/app/ml/models`. To activate a trained model, set the path in `.env`:
  ```bash
  MODEL_PATH=/app/ml/models/class_weight
  # or
  MODEL_PATH=/app/ml/models/smote_enn
  ```
  Restart the backend container to apply:
  ```bash
  docker compose up -d backend
  ```

- **Research Benchmark Metrics**: Research metrics reports located in `ml/reports/metrics.json` are mounted into the backend container read-only at `/app/ml/reports`. When present, `GET /api/v1/metrics` serves the strategy comparison table (`ml_metrics_status: "available"`). When absent or running in stub mode without metrics, `ml_metrics_status` reports `"unavailable"` and `ml_metrics` returns `null` so the frontend gracefully notifies the analyst.

---

## 3. Import Convention & Working Directory

This project standardizes on the **`app.*`** package namespace:
```python
from app.models.schemas import TransactionIn, PredictionOut
from app.db.repository import get_repository
```

### Execution Rules
- **Backend Directory Context**: When running from `backend/`:
  ```bash
  cd backend
  python -m pytest app/tests -v
  ```
- **Root Directory Context**: When executing from the repository root, provide `PYTHONPATH=backend:.`:
  - **Linux / macOS / Git Bash**:
    ```bash
    PYTHONPATH=backend:. pytest backend/app/tests -v
    ```
  - **Windows PowerShell**:
    ```powershell
    $env:PYTHONPATH="backend:."; pytest backend/app/tests -v
    ```

---

## 4. Database Migrations & Schema Drift Verification

FraudLens Gateway uses PostgreSQL for persistence with Alembic migrations.

> [!IMPORTANT]
> The database only stores scored transactions and analyst reviews. Raw PaySim training data is never stored in PostgreSQL.
> Tables must never be created or altered manually. Schema changes must proceed through Alembic migrations.
> `alembic.ini` resides at the **repository root**. Always run Alembic commands with `PYTHONPATH=backend` from the root directory.

### Verifying Migration Integrity
Run Alembic from the repository root with `PYTHONPATH=backend`:
```bash
PYTHONPATH=backend alembic upgrade head
PYTHONPATH=backend alembic downgrade base
PYTHONPATH=backend alembic upgrade head
PYTHONPATH=backend alembic check
```
`alembic check` verifies that model definitions in `backend/app/db/models.py` match the active database schema without generating throwaway revision files.

*(Note: The integration test `test_postgres_alembic_migrations` in `backend/app/tests/test_repository.py` also runs this full round-trip and drift check automatically against `TEST_DATABASE_URL`).*

---

## 5. Operational & Performance Scripts

### Demo Seeding (`scripts/seed_demo.py`)
Generates and posts varied transactions through `/api/v1/predict`.
- **Bash**:
  ```bash
  export ANALYST_USERNAME="analyst_1"
  export ANALYST_PASSWORD="<analyst_password>"
  python scripts/seed_demo.py --url http://127.0.0.1:8000 --count 100 --yes
  ```
- **PowerShell**:
  ```powershell
  $env:ANALYST_USERNAME="analyst_1"
  $env:ANALYST_PASSWORD="<analyst_password>"
  python scripts/seed_demo.py --url http://127.0.0.1:8000 --count 100 --yes
  ```

### Latency Benchmarking (`scripts/benchmark_latency.py`)
Measures scoring latency across $N$ requests.
- **Bash**:
  ```bash
  export ANALYST_USERNAME="analyst_1"
  export ANALYST_PASSWORD="<analyst_password>"
  python scripts/benchmark_latency.py --url http://127.0.0.1:8000 --num-requests 1000 --warmup 50 --yes
  ```
- **PowerShell**:
  ```powershell
  $env:ANALYST_USERNAME="analyst_1"
  $env:ANALYST_PASSWORD="<analyst_password>"
  python scripts/benchmark_latency.py --url http://127.0.0.1:8000 --num-requests 1000 --warmup 50 --yes
  ```

---

## 6. Makefile Targets & Raw Command Equivalents

| Target | Description | Raw Command (Linux / macOS / Git Bash) | Raw Command (Windows PowerShell) |
| :--- | :--- | :--- | :--- |
| `make up` | Start Docker Compose stack | `docker compose up -d --build --wait` | `docker compose up -d --build --wait` |
| `make down` | Stop & remove containers + volumes | `docker compose down -v` | `docker compose down -v` |
| `make test` | Run test suite | `PYTHONPATH=backend:. pytest backend/app/tests ml/tests -v` | `$env:PYTHONPATH="backend:."; pytest backend/app/tests ml/tests -v` |
| `make lint` | Run Ruff and ESLint | `ruff check backend/ ml/ && cd frontend && npm run lint` | `ruff check backend/ ml/; cd frontend; npm run lint` |
| `make smoke` | Run smoke test suite | `bash scripts/smoke_test.sh` | `& "C:\Program Files\Git\bin\bash.exe" scripts/smoke_test.sh` |
| `make logs` | Tail Docker Compose logs | `docker compose logs -f` | `docker compose logs -f` |
| `make openapi` | Regenerate `docs/openapi.json` | `PYTHONPATH=backend python scripts/generate_openapi.py` | `$env:PYTHONPATH="backend"; python scripts/generate_openapi.py` |
| `make run` | Launch backend server locally | `cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` | `cd backend; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` |
| `make seed` | Ingest varied demo transactions | `python scripts/seed_demo.py` | `python scripts/seed_demo.py` |
| `make benchmark` | Benchmark latency (1000 runs) | `python scripts/benchmark_latency.py` | `python scripts/benchmark_latency.py` |
| `make test-postgres` | Run PostgreSQL integration tests | `REQUIRE_POSTGRES=1 TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/fraudlens_test PYTHONPATH=backend python -m pytest backend/app/tests/test_repository.py -k "postgres" -v` | `$env:REQUIRE_POSTGRES="1"; $env:TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/fraudlens_test"; $env:PYTHONPATH="backend"; python -m pytest backend/app/tests/test_repository.py -k "postgres" -v` |
