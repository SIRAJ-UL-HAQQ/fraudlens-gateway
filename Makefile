.PHONY: up down test lint smoke logs openapi run seed benchmark test-postgres

up:
	docker compose up -d --build --wait

down:
	docker compose down -v

test:
	PYTHONPATH=backend:. pytest backend/app/tests ml/tests -v

lint:
	ruff check backend/ ml/
	cd frontend && npm run lint

smoke:
	bash scripts/smoke_test.sh

logs:
	docker compose logs -f

openapi:
	PYTHONPATH=backend python scripts/generate_openapi.py

run:
	cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

seed:
	python scripts/seed_demo.py

benchmark:
	python scripts/benchmark_latency.py

test-postgres:
	REQUIRE_POSTGRES=1 TEST_DATABASE_URL=$${TEST_DATABASE_URL:?TEST_DATABASE_URL is required} PYTHONPATH=backend python -m pytest backend/app/tests/test_repository.py -k "postgres" -v
