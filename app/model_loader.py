"""
model_loader.py — MLflow model loading and in-memory caching.

Responsibilities
----------------
1. Read the MLflow Tracking URI from the environment at import time.
2. On first call to get_model(), load the *Production* stage model from the
   MLflow Model Registry and cache it in module-level variables so subsequent
   requests reuse the same in-memory object without hitting the registry again.
3. Expose get_model() and get_model_version() for use in FastAPI route handlers
   and startup events (main.py).

Design notes
------------
• Module-level caching (_model / _model_version) is the simplest approach for
  a single-process API server.  For multi-worker deployments (Gunicorn with
  multiple workers) each worker loads its own copy, which is usually fine for
  read-only inference.
• We raise a RuntimeError (not just log a warning) when the model is missing
  so FastAPI's startup event fails loudly rather than silently serving errors.
"""

import logging
import os
from typing import Any

import mlflow
import mlflow.pyfunc

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info("MLflow tracking URI set to: %s", MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# Model Registry settings
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "FraudDetectionModel")
MODEL_STAGE: str = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_model: Any | None = None
_model_version: str | None = None
_startup_error: str | None = None  # set when model load fails at startup


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_latest_production_version() -> str:
    """
    Query the MLflow Model Registry for the latest version of MODEL_NAME
    that is currently in the Production stage.

    Returns
    -------
    str
        The version number as a string (e.g. "3").

    Raises
    ------
    RuntimeError
        If no Production version exists for the model — this typically means
        the training pipeline hasn't run yet or hasn't promoted a model.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        # search_model_versions is the current API (MLflow >= 2.0).
        # get_latest_versions() is deprecated and will be removed in a future release.
        filter_str = f"name='{MODEL_NAME}'"
        all_versions = client.search_model_versions(filter_str)
        versions = [v for v in all_versions if v.current_stage == MODEL_STAGE]
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"Could not query the MLflow Model Registry for model "
            f"'{MODEL_NAME}' (stage='{MODEL_STAGE}'). "
            f"Check that MLFLOW_TRACKING_URI='{MLFLOW_TRACKING_URI}' is "
            f"reachable and the model has been registered.\n"
            f"Original error: {exc}"
        ) from exc

    if not versions:
        raise RuntimeError(
            f"No '{MODEL_STAGE}' version found for model '{MODEL_NAME}' "
            f"in the MLflow Model Registry at '{MLFLOW_TRACKING_URI}'.\n"
            f"Tip: run the training pipeline and promote a model version to "
            f"'{MODEL_STAGE}' before starting the API server."
        )

    # Pick the version with the highest version number (most recently promoted).
    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Found model '%s' version '%s' in stage '%s' (run_id=%s).",
        MODEL_NAME,
        latest.version,
        MODEL_STAGE,
        latest.run_id,
    )
    return latest.version


def _resolve_local_source(source: str) -> str:
    """
    Convert a model_version.source URI to a Linux-safe local path.

    MLflow stores artifact URIs as they were on the training host, e.g.
    ``file:///F:/Github/fraud-mlops/mlruns/1/<run_id>/artifacts/model`` on
    Windows. Inside a Linux container those Windows-rooted paths are invalid.

    We patch them the same way the entrypoint patches the SQLite DB — replacing
    the Windows prefix with the container-side ``/mlruns`` mount point — then
    strip the ``file://`` scheme so MLflow gets a plain local path.

    If the source is already a Linux path (e.g. already patched in the DB),
    it is returned unchanged.
    """
    import re
    from urllib.parse import urlparse

    # Replace "file:///X:/.../<anything>/mlruns" → "file:///mlruns"
    source = re.sub(r'file:///[A-Za-z]:/[^"]*?/mlruns', "file:///mlruns", source)
    # Replace bare "/X:/.../<anything>/mlruns" → "/mlruns"
    source = re.sub(r'/[A-Za-z]:/[^"]*?/mlruns', "/mlruns", source)

    # Strip the file:// scheme to get a plain local path for load_model()
    parsed = urlparse(source)
    if parsed.scheme == "file":
        return parsed.path  # e.g. "/mlruns/1/<run_id>/artifacts/model"
    return source


def _load_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Load the Production model from the MLflow Model Registry.

    Rather than using the ``models:/Name/Stage`` URI (which routes through
    ``models_artifact_repo.py`` and re-resolves the raw ``model_versions.source``
    stored in the DB — a Windows path on this host), we manually look up the
    winning version's source, patch any Windows-style path to the container-side
    ``/mlruns`` mount, and call ``load_model`` with the direct local filesystem
    path. This completely bypasses the problematic resolution chain.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        A generic Python function wrapper that exposes a `.predict()` method
        accepting a pandas DataFrame and returning predictions.

    Raises
    ------
    RuntimeError
        If no Production model is found, or if loading from the resolved path
        fails.
    """
    client = mlflow.tracking.MlflowClient()

    # Find the winning Production version (same logic as _fetch_latest_production_version)
    filter_str = f"name='{MODEL_NAME}'"
    all_versions = client.search_model_versions(filter_str)
    versions = [v for v in all_versions if v.current_stage == MODEL_STAGE]

    if not versions:
        raise RuntimeError(
            f"No '{MODEL_STAGE}' version found for model '{MODEL_NAME}' "
            f"in the MLflow Model Registry at '{MLFLOW_TRACKING_URI}'."
        )

    best = max(versions, key=lambda v: int(v.version))
    raw_source = (
        best.source
    )  # e.g. "file:///F:/Github/.../mlruns/1/<rid>/artifacts/model"
    local_path = _resolve_local_source(raw_source)

    logger.info(
        "Loading model version '%s' (run_id=%s) from resolved path: %s",
        best.version,
        best.run_id,
        local_path,
    )

    try:
        loaded = mlflow.pyfunc.load_model(local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from resolved path '{local_path}'.\n"
            f"  Raw source in registry: {raw_source}\n"
            f"  Ensure the mlruns/ directory is mounted at /mlruns inside the container.\n"
            f"Original error: {exc}"
        ) from exc

    logger.info("Model loaded successfully from '%s'.", local_path)
    return loaded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model_on_startup() -> None:
    """
    Eagerly load the Production model and populate the module-level cache.

    Failure is non-fatal: if no Production model is found (e.g. a fresh Render
    deploy before training has run), a warning is logged and the server starts
    normally.  Endpoints that need the model (/predict, /health) will return
    HTTP 503 until a model is available and the container is restarted.
    """
    global _model, _model_version, _startup_error  # noqa: PLW0603

    logger.info(
        "Startup: loading '%s' (stage=%s) from the MLflow Model Registry …",
        MODEL_NAME,
        MODEL_STAGE,
    )

    try:
        version = _fetch_latest_production_version()
        _model = _load_model()
        _model_version = version
        logger.info(
            "Startup complete. Serving model '%s' version '%s'.",
            MODEL_NAME,
            _model_version,
        )
    except RuntimeError as exc:
        # Store the error so endpoints can surface it as a 503, but do NOT
        # re-raise — the server must stay up so /health can be polled and
        # Render doesn't mark the deploy as permanently failed.
        _startup_error = str(exc)
        logger.warning(
            "Model not loaded at startup (server will return 503 on /predict): %s",
            _startup_error,
        )


def get_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Return the cached Production model.

    Raises
    ------
    RuntimeError
        If the model failed to load at startup or has not yet been loaded.
    """
    global _model, _model_version  # noqa: PLW0603

    if _startup_error is not None and _model is None:
        raise RuntimeError(_startup_error)

    if _model is None:
        logger.warning(
            "get_model() called before load_model_on_startup(); "
            "performing lazy load now."
        )
        _model_version = _fetch_latest_production_version()
        _model = _load_model()

    return _model


def get_model_version() -> str:
    """
    Return the version string of the currently cached model.

    Returns
    -------
    str
        The MLflow model version number, e.g. ``"3"``.

    Raises
    ------
    RuntimeError
        If get_model() / load_model_on_startup() has not yet been called
        and the lazy-load also fails.
    """
    if _model_version is None:
        get_model()

    assert _model_version is not None, "Model version must be set after load."
    return _model_version
