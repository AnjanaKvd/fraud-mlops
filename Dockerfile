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
RUN pip install --no-cache-dir --prefix=/install --timeout 120 --retries 5 -r requirements.txt


# =============================================================================
# Stage 2: Runtime (final image)
# Only the installed packages + app code land here — no compilers, no cache.
# =============================================================================
FROM python:3.11-slim AS runtime

# MLFLOW_TRACKING_URI: overridden by entrypoint.sh at runtime with a writable
#                       copy of the bundled DB (or a freshly created empty one).
# MLFLOW_DB_PATH:      entrypoint reads this to locate the source DB to copy.
# PORT:               which port uvicorn listens on (overridable at runtime).
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlruns.db \
    MLFLOW_DB_PATH=/app/mlruns.db \
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
# Bundle the MLflow tracking DB and model artifacts into the image so the
# container is fully self-contained for cloud deploys (Render, Fly.io, etc.)
# that have no external volume mounts.
#
# mlruns.db  — SQLite tracking store with registered model metadata
# mlruns/    — MLflow artifact tree (XGBoost model files, plots, etc.)
#
# Both files are now tracked by git (removed from .gitignore), so they are
# available at build time in the Docker context.
#
# Note: the entrypoint copies mlruns.db to /tmp/mlruns_patched.db at startup
# to get a writable copy; the original /app/mlruns.db stays read-only.
# ---------------------------------------------------------------------------
COPY mlruns.db ./mlruns.db
COPY mlruns/ /mlruns/

# Copy entrypoint script and make it executable (must be done as root,
# before we switch to the non-root user).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure the non-root user owns the app directory and mlruns artifacts
RUN chown -R appuser:appgroup /app /mlruns

# Switch to non-root user for all subsequent commands
USER appuser

# Expose the HTTP port the API listens on
EXPOSE 8000

# ---------------------------------------------------------------------------
# Entrypoint — copies mlruns.db to a writable /tmp location, patches any
# Windows artifact URIs, then starts uvicorn.
# Using ENTRYPOINT (not CMD) ensures the patch step always runs on container
# start, even if the operator passes extra arguments.
# ---------------------------------------------------------------------------
ENTRYPOINT ["/entrypoint.sh"]
