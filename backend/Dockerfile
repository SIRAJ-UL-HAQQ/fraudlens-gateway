# Stage 1: Build virtual environment
FROM python:3.13.5-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt ./requirements.txt
COPY backend/constraints-linux.txt ./constraints.txt

# Swap xgboost with xgboost-cpu for backend runtime image (avoids 350MB+ CUDA/NCCL wheels)
RUN sed 's/xgboost==/xgboost-cpu==/' requirements.txt > requirements-docker.txt \
    && sed 's/xgboost==/xgboost-cpu==/' constraints.txt > constraints-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt -c constraints-docker.txt

# Stage 2: Runtime image
FROM python:3.13.5-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --user-group appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy backend application and database migrations
COPY --chown=appuser:appuser backend/app/ ./app/
COPY --chown=appuser:appuser alembic.ini ./alembic.ini
COPY --chown=appuser:appuser backend/alembic/ ./backend/alembic/

# Copy ml feature preprocessor module (needed for input transformation)
COPY --chown=appuser:appuser ml/__init__.py ./ml/__init__.py
COPY --chown=appuser:appuser ml/src/__init__.py ./ml/src/__init__.py
COPY --chown=appuser:appuser ml/src/features.py ./ml/src/features.py

# Build-time verification: ensures Linux wheels and ML features import cleanly
RUN python -c "import xgboost, sklearn, joblib, shap; import ml.src.features; print('Backend ML imports OK')"

# Copy entrypoint script
COPY --chown=appuser:appuser backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

ENTRYPOINT ["/app/entrypoint.sh"]
