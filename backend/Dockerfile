# Builds the shared backend image used by the API and worker.
FROM python:3.12-slim AS base




ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1

WORKDIR /app



COPY backend/requirements.txt ./backend/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && pip install --no-cache-dir -r backend/requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*



COPY backend/ ./backend/
COPY tests/ ./tests/



RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000



HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/ || exit 1


CMD ["uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8000"]
