# Backend image — serves the API, and (with a different command) the worker.
# One image for both because they share every dependency and import each other;
# two images would double a ~1GB crewai install to save nothing.
FROM python:3.12-slim AS base

# PYTHONUNBUFFERED so the structured JSON logs reach docker logs immediately
# rather than sitting in a pipe buffer, which makes a crashed container look
# silent. PYTHONDONTWRITEBYTECODE keeps the layer free of .pyc noise.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1

WORKDIR /app

# curl is for HEALTHCHECK below; build-essential is dropped again in the same
# layer so it never reaches the final image (some deps still build wheels).
COPY backend/requirements.txt ./backend/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && pip install --no-cache-dir -r backend/requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Source last: requirements change rarely, source changes constantly, so this
# ordering keeps the expensive install layer cached across code edits.
COPY backend/ ./backend/
COPY tests/ ./tests/

# Non-root. The app writes nothing to disk (results go to Supabase), so it
# needs no writable volume.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Hits the real health endpoint, not a TCP probe — uvicorn accepting a socket
# says nothing about whether the app booted (Supabase config, missing env).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/ || exit 1

# Overridden to `python backend/worker.py` for the worker service.
CMD ["uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8000"]
