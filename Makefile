.PHONY: test openapi run smoke lint

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
