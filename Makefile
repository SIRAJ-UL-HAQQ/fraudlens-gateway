.PHONY: test openapi run smoke lint seed benchmark test-postgres

test:
	cd backend && python -m pytest app/tests -v --tb=short

openapi:
	python scripts/generate_openapi.py

run:
	cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

smoke:
	bash scripts/smoke_test.sh http://localhost:8000

lint:
	ruff check backend/

seed:
	python scripts/seed_demo.py

benchmark:
	python scripts/benchmark_latency.py

test-postgres:
	cd backend && REQUIRE_POSTGRES=1 python -m pytest app/tests/test_repository.py -k "postgres" -v
