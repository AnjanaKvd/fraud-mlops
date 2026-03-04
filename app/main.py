"""
main.py — FastAPI application entry point for the Credit Card Fraud Detection API.

Structure
---------
- Lifespan context manager  : loads the ML model once on startup.
- GET  /                     : lightweight liveness probe.
- GET  /health               : readiness check with model version and uptime.
- POST /predict              : score a transaction, log it, return a result.
- GET  /metrics              : rolling summary of the last 100 predictions.
"""

import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import get_model, get_model_version, load_model_on_startup
from app.schemas import TransactionInput, PredictionOutput

from app.inference_logger import log_prediction


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory rolling prediction store
# ---------------------------------------------------------------------------
# A fixed-size deque keeps the last MAX_HISTORY predictions in memory with
# O(1) appends and automatic eviction of the oldest entry.
# Each entry is a dict: {"fraud_probability": float, "is_fraud": bool}
MAX_HISTORY = 100
_prediction_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)

# ---------------------------------------------------------------------------
# Server start time — used to compute uptime in /health
# ---------------------------------------------------------------------------
_start_time: float = time.time()

# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """
    FastAPI lifespan handler.

    Code before `yield` runs on startup; code after `yield` runs on shutdown.
    Loading the model here (rather than at import time) means:
      • Failed loads abort the server with a clear error before it starts
        accepting traffic.
      • The model is guaranteed to be in the cache for every request.
    """
    logger.info("=== Fraud Detection API starting up ===")
    try:
        load_model_on_startup()
    except RuntimeError as exc:
        # Re-raise so Uvicorn/Gunicorn surfaces the error and exits non-zero.
        logger.critical("Model failed to load on startup: %s", exc)
        raise

    logger.info("=== Startup complete — serving requests ===")
    yield
    # Shutdown hook (add any cleanup here, e.g. flush logs to a database)
    logger.info("=== Fraud Detection API shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time fraud scoring for credit-card transactions.\n\n"
        "Submit a transaction's 28 PCA features and the transaction amount "
        "to receive an instant fraud probability and binary classification.\n\n"
        "The underlying model is loaded from the **MLflow Model Registry** "
        "(Production stage) at startup."
    ),
    version="1.0.0",
    contact={
        "name": "MLOps Team",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# Allow all origins during development.  Tighten allowed_origins to specific
# domains before promoting to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev permissive — restrict in production
    # allow_credentials must be False when allow_origins=["*"].
    # The CORS spec forbids credentials with a wildcard origin; Starlette
    # raises ValueError if both are True.  Set specific origins + True in prod.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/",
    summary="Liveness probe",
    tags=["Health"],
)
async def root() -> dict[str, str]:
    """
    Lightweight liveness check.

    Returns a static JSON payload confirming the service is reachable.
    Suitable for use as a container liveness probe (e.g. in Kubernetes).

    Returns
    -------
    dict
        ``{"status": "healthy", "service": "Fraud Detection API"}``
    """
    return {"status": "healthy", "service": "Fraud Detection API"}


@app.get(
    "/health",
    summary="Readiness / health check",
    tags=["Health"],
)
async def health() -> dict[str, Any]:
    """
    Readiness check that verifies the model is loaded and reports runtime stats.

    Returns
    -------
    dict
        ``status``          — "ready" when the model is in memory.
        ``model_version``   — MLflow version of the loaded model.
        ``uptime_seconds``  — elapsed seconds since the server started.
        ``total_predictions``— cumulative count across all /predict calls
                               (resets on server restart; capped at last 100
                               in the rolling buffer).
    """
    try:
        version = get_model_version()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not ready: {exc}",
        ) from exc

    return {
        "status": "ready",
        "model_version": version,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "total_predictions": len(_prediction_history),
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Score a transaction for fraud",
    tags=["Prediction"],
)
async def predict(transaction: TransactionInput) -> PredictionOutput:
    """
    Run the fraud-detection model against a single transaction.

    The request body must contain all 28 PCA features (V1–V28) plus the
    transaction Amount.  The endpoint returns:

    - **fraud_probability** — model confidence that the transaction is fraud.
    - **is_fraud** — True when fraud_probability >= 0.5.
    - **model_version** — which model version produced this result.
    - **prediction_id** — UUID for tracing / audit purposes.
    - **timestamp** — UTC time of the prediction.

    Raises
    ------
    HTTPException 503
        If the model hasn't been loaded (should not happen after successful
        startup, but guards against edge cases).
    HTTPException 500
        If inference fails for any unexpected reason.
    """
    # --- 1. Retrieve the cached model ----------------------------------------
    try:
        model = get_model()
        model_version = get_model_version()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {exc}",
        ) from exc

    # --- 2. Build a single-row DataFrame matching the training feature order --
    # transaction.model_dump() preserves insertion order (V1…V28, Amount).
    feature_dict = transaction.model_dump()
    input_df = pd.DataFrame([feature_dict])

    # --- 3. Run inference -----------------------------------------------------
    try:
        # mlflow.pyfunc models expose .predict() which returns a numpy array
        # or a pandas Series / DataFrame depending on the flavour.
        raw_output = model.predict(input_df)

        # Normalise to a scalar probability.  For sklearn classifiers logged
        # with predict_proba, mlflow returns an ndarray of shape (n, 2).
        if hasattr(raw_output, "iloc"):
            # pandas Series / single-column DataFrame
            fraud_prob = float(raw_output.iloc[0])
        elif hasattr(raw_output, "__len__") and len(raw_output) > 0:
            prob_row = raw_output[0]
            # [[p_legit, p_fraud]] shape from predict_proba
            fraud_prob = (
                float(prob_row[1]) if hasattr(prob_row, "__len__") else float(prob_row)
            )
        else:
            fraud_prob = float(raw_output)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed for transaction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        ) from exc

    # --- 4. Build the response ------------------------------------------------
    is_fraud = fraud_prob >= 0.5
    result = PredictionOutput(
        fraud_probability=fraud_prob,
        is_fraud=is_fraud,
        model_version=model_version,
        # prediction_id and timestamp get auto-populated by PredictionOutput's
        # default_factory fields defined in schemas.py.
    )

    # --- 5. Persist to in-memory history and async log -----------------------
    _prediction_history.append({"fraud_probability": fraud_prob, "is_fraud": is_fraud})
    log_prediction(
        prediction_id=result.prediction_id,
        amount=feature_dict["Amount"],  # required by inference_logger schema
        fraud_probability=fraud_prob,
        is_fraud=is_fraud,
        model_version=model_version,
        timestamp=result.timestamp,
    )

    logger.info(
        "prediction_id=%s  fraud_prob=%.4f  is_fraud=%s  model_version=%s",
        result.prediction_id,
        fraud_prob,
        is_fraud,
        model_version,
    )

    return result


@app.get(
    "/metrics",
    summary="Rolling prediction metrics",
    tags=["Metrics"],
)
async def metrics() -> dict[str, Any]:
    """
    Summary statistics over the last 100 predictions held in memory.

    Useful for a lightweight operational dashboard or alerting rule
    (e.g. alert when fraud_rate spikes above a threshold).

    Returns
    -------
    dict
        ``window``          — maximum number of predictions tracked.
        ``count``           — actual number of predictions in the window.
        ``fraud_rate``      — fraction of predictions classified as fraud.
        ``avg_probability`` — mean fraud_probability across the window.
    """
    history = list(_prediction_history)  # snapshot to avoid race conditions
    count = len(history)

    if count == 0:
        return {
            "window": MAX_HISTORY,
            "count": 0,
            "fraud_rate": None,
            "avg_probability": None,
        }

    fraud_flags = [p["is_fraud"] for p in history]
    probabilities = [p["fraud_probability"] for p in history]

    return {
        "window": MAX_HISTORY,
        "count": count,
        "fraud_rate": round(sum(fraud_flags) / count, 4),
        "avg_probability": round(sum(probabilities) / count, 4),
    }
