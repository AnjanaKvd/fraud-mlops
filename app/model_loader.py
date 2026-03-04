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


def _load_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Load the Production model from the MLflow Model Registry.

    The model URI format ``models:/<name>/<stage>`` is resolved by MLflow to
    the artifact path of the latest Production version automatically.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        A generic Python function wrapper that exposes a `.predict()` method
        accepting a pandas DataFrame and returning predictions.

    Raises
    ------
    RuntimeError
        Propagated from _fetch_latest_production_version() if no Production
        model is registered, or from mlflow.pyfunc.load_model() on any
        loading failure.
    """
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from URI: %s", model_uri)

    try:
        loaded = mlflow.pyfunc.load_model(model_uri)
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"Failed to load model from '{model_uri}'.\n"
            f"Ensure that:\n"
            f"  • The training pipeline has completed successfully.\n"
            f"  • A model version has been promoted to '{MODEL_STAGE}'.\n"
            f"  • MLFLOW_TRACKING_URI ('{MLFLOW_TRACKING_URI}') is correct.\n"
            f"Original error: {exc}"
        ) from exc

    logger.info("Model loaded successfully from '%s'.", model_uri)
    return loaded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model_on_startup() -> None:
    """
    Eagerly load the Production model and populate the module-level cache.

    Call this once from FastAPI's ``@app.on_event("startup")`` handler in
    main.py so the first real prediction request doesn't pay the cold-start
    penalty and so startup failures are surfaced immediately (rather than
    on the first incoming request).

    Raises
    ------
    RuntimeError
        If the model cannot be found or loaded — lets FastAPI abort startup
        so the server never enters a broken state.
    """
    global _model, _model_version  # noqa: PLW0603 — intentional module cache

    logger.info(
        "Startup: loading '%s' (stage=%s) from the MLflow Model Registry …",
        MODEL_NAME,
        MODEL_STAGE,
    )

    version = _fetch_latest_production_version()

    _model = _load_model()
    _model_version = version

    logger.info(
        "Startup complete. Serving model '%s' version '%s'.",
        MODEL_NAME,
        _model_version,
    )


def get_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Return the cached Production model, loading it on first call if needed.

    Lazy-loading is a fallback for cases where load_model_on_startup() was
    not called (e.g. unit tests that mock the model at import time).

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        The in-memory model ready for inference via `.predict()`.

    Raises
    ------
    RuntimeError
        If the model is not cached and cannot be loaded.
    """
    global _model, _model_version  # noqa: PLW0603

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
