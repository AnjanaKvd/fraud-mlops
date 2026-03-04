# =============================================================================
# Stage 1: Builder
# Install dependencies in an isolated environment to keep the final image lean.
# =============================================================================
FROM python:3.11-slim AS builder

# Set working directory for the build stage
WORKDIR /build

# Install system-level build tools needed by some Python packages (e.g. numpy/xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first — this layer is cached as long as requirements.txt
# doesn't change, which means `pip install` is skipped on code-only changes.
COPY requirements.txt .

# Install all Python dependencies into a dedicated prefix so they can be
# copied cleanly into the final stage without dragging build tools along.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================================
# Stage 2: Runtime (final image)
# Only the installed packages + app code land here — no compilers, no cache.
# =============================================================================
FROM python:3.11-slim AS runtime

# --- Environment variables ---------------------------------------------------
# MLFLOW_TRACKING_URI: where MLflow logs experiments / loads registered models.
# PORT:               which port uvicorn listens on (overridable at runtime).
ENV MLFLOW_TRACKING_URI=sqlite:////data/mlruns.db \
    PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install only the minimal runtime system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from the builder stage
COPY --from=builder /install /usr/local

# ---------------------------------------------------------------------------
# Security: run as a non-root user
# ---------------------------------------------------------------------------
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

# Set the application working directory
WORKDIR /app

# Copy application source code (done AFTER deps for better layer caching)
COPY app/ ./app/

# ---------------------------------------------------------------------------
# Volumes for runtime databases
# mlruns.db and predictions.db are mounted at runtime; do NOT bake data in.
# The host paths are controlled via docker-compose.yml or `-v` flags.
# ---------------------------------------------------------------------------
VOLUME ["/data"]

# Ensure the non-root user owns the app directory
RUN chown -R appuser:appgroup /app

# Switch to non-root user for all subsequent commands
USER appuser

# Expose the HTTP port the API listens on
EXPOSE 8000

# ---------------------------------------------------------------------------
# Startup command
# Uses the PORT env variable so the port is configurable without rebuilding.
# app.main:app  →  the `app` FastAPI instance inside app/main.py
# ---------------------------------------------------------------------------
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
