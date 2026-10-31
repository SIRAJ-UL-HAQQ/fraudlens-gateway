#!/bin/sh
set -e

# Run database migrations
echo "Running alembic database migrations..."
alembic upgrade head

# Exec uvicorn to become PID 1
echo "Starting FraudLens Gateway via uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
