# Running FraudLens Gateway & Development Commands

## 1. Import Convention & Working Directory

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
- **Root Directory Context**: When executing from the repository root, provide `PYTHONPATH=backend`:
  - **Linux / macOS / Git Bash**:
    ```bash
    PYTHONPATH=backend python -m pytest backend/app/tests -v
    ```
  - **Windows PowerShell**:
    ```powershell
    $env:PYTHONPATH="backend"; python -m pytest backend/app/tests -v
    ```

---

## 2. Database Migrations & Schema Verification

FraudLens Gateway uses PostgreSQL for persistence with Alembic migrations.

> [!IMPORTANT]
> The database only stores scored transactions and analyst reviews. Raw PaySim training data is never stored in PostgreSQL.
> Tables must never be created or altered manually. Schema changes must proceed through Alembic migrations.

### Verifying Migration Integrity
Run the complete round-trip migration cycle followed by `alembic check` to ensure no schema drift:
```bash
cd backend
alembic upgrade head
alembic downgrade base
alembic upgrade head
alembic check
```
`alembic check` verifies that model definitions in `app/db/models.py` match the active database schema without generating throwaway revision files.

---

## 3. Operational & Performance Scripts

### Demo Seeding (`scripts/seed_demo.py`)
Generates and posts varied transactions (obvious fraud, obvious legitimate, zero amount, borderline review cases) through `/api/v1/predict`.

> [!CAUTION]
> `seed_demo.py` writes permanent rows into the audit database. It must target a throwaway or dedicated development database.
> The script requires the `--yes` flag to proceed and reads analyst password exclusively from `ANALYST_PASSWORD`.

- **Bash**:
  ```bash
  export ANALYST_USERNAME="analyst_1"
  export ANALYST_PASSWORD="<analyst_password>"
  python scripts/seed_demo.py --url http://localhost:8000 --count 100 --yes
  ```
- **PowerShell**:
  ```powershell
  $env:ANALYST_USERNAME="analyst_1"
  $env:ANALYST_PASSWORD="<analyst_password>"
  python scripts/seed_demo.py --url http://localhost:8000 --count 100 --yes
  ```

### Latency Benchmarking (`scripts/benchmark_latency.py`)
Measures scoring latency across $N$ requests (defaults: 1000 requests, 50 warm-up), computing client round-trip and server `latency_ms` percentiles (min, max, mean, p50, p95, p99), and exporting a structured JSON report to `reports/benchmarks/`.

> [!CAUTION]
> Like the seeder, `benchmark_latency.py` persists transactions to the target database and requires `--yes`.
> It warns if the server is running in `stub` model mode.

- **Bash**:
  ```bash
  export ANALYST_USERNAME="analyst_1"
  export ANALYST_PASSWORD="<analyst_password>"
  python scripts/benchmark_latency.py --url http://localhost:8000 --num-requests 1000 --warmup 50 --yes
  ```
- **PowerShell**:
  ```powershell
  $env:ANALYST_USERNAME="analyst_1"
  $env:ANALYST_PASSWORD="<analyst_password>"
  python scripts/benchmark_latency.py --url http://localhost:8000 --num-requests 1000 --warmup 50 --yes
  ```

---

## 4. Makefile Targets & Raw Command Equivalents

Below is the complete reference of Makefile targets and their direct, raw shell equivalents:

| Target | Description | Raw Command (Linux / macOS / Git Bash) | Raw Command (Windows PowerShell) |
| :--- | :--- | :--- | :--- |
| `make test` | Run test suite | `cd backend && python -m pytest app/tests -v --tb=short` | `cd backend; python -m pytest app/tests -v --tb=short` |
| `make openapi` | Regenerate `docs/openapi.json` | `PYTHONPATH=backend python scripts/generate_openapi.py` | `$env:PYTHONPATH="backend"; python scripts/generate_openapi.py` |
| `make run` | Launch backend server | `cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` | `cd backend; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` |
| `make smoke` | Execute smoke test suite | `bash scripts/smoke_test.sh http://localhost:8000` | `bash scripts/smoke_test.sh http://localhost:8000` |
| `make lint` | Run Ruff linter check | `ruff check backend/` | `ruff check backend/` |
| `make seed` | Ingest varied demo transactions | `python scripts/seed_demo.py` | `python scripts/seed_demo.py` |
| `make benchmark` | Benchmark latency (1000 runs) | `python scripts/benchmark_latency.py` | `python scripts/benchmark_latency.py` |
| `make test-postgres` | Run PostgreSQL integration tests | `cd backend && REQUIRE_POSTGRES=1 python -m pytest app/tests/test_repository.py -k "postgres" -v` | `cd backend; $env:REQUIRE_POSTGRES="1"; python -m pytest app/tests/test_repository.py -k "postgres" -v` |

> [!NOTE]
> `make test-postgres` enforces that `TEST_DATABASE_URL` is set and exits with a non-zero status code if unset. The database name must end in `_test`.
