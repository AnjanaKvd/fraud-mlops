"""
tests/test_api.py — Integration tests for the Fraud Detection FastAPI app.

Strategy
--------
The real MLflow model is NOT needed for these tests.  We monkey-patch the
three functions that touch MLflow (get_model, get_model_version,
load_model_on_startup) with lightweight stubs before the app is imported,
so no registry or database connection is required.

The stub model returns a fixed fraud_probability of 0.1 for any input,
which is enough to exercise all endpoints without any ML dependencies.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Stub model — replaces the real MLflow pyfunc model
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in that mirrors the mlflow.pyfunc.PyFuncModel interface."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # noqa: ARG002
        """Return a fixed probability of 0.1 for every row."""
        n = len(df)
        # Shape (n, 2) — [p_legit, p_fraud] — matches sklearn classifiers
        # logged with predict_proba.
        return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])


_STUB_MODEL = _StubModel()
_STUB_VERSION = "1"


# ---------------------------------------------------------------------------
# Patch model_loader BEFORE importing the FastAPI app so the lifespan
# context manager never tries to contact MLflow.
# ---------------------------------------------------------------------------
import app.model_loader as _ml  # noqa: E402 — must come after stub definition

_ml._model = _STUB_MODEL  # populate the module-level cache directly
_ml._model_version = _STUB_VERSION


def _noop_load() -> None:  # noqa: D401
    """No-op replacement for load_model_on_startup."""


_ml.load_model_on_startup = _noop_load  # type: ignore[assignment]

# Now import the app — the lifespan will call our no-op loader.
from app.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_transaction() -> dict:
    """
    A realistic credit-card transaction payload.

    All V features are within the typical PCA range (-5 to 5) and Amount is
    positive.  Values are taken from the first row of the public Kaggle
    Credit Card Fraud Detection dataset.
    """
    return {
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62,
    }


@pytest_asyncio.fixture()
async def client() -> AsyncClient:
    """
    Async HTTP test client wired to the FastAPI app via ASGITransport.

    Using ``lifespan="auto"`` ensures the startup/shutdown hooks run just
    as they would in production, exercising the full request lifecycle.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """GET /health → 200 and response contains 'model_version'."""
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body, f"Key 'model_version' missing from: {body}"
    assert body["status"] == "ready"


@pytest.mark.asyncio
async def test_predict_valid(client: AsyncClient, valid_transaction: dict) -> None:
    """POST /predict with valid data → 200 and 0 ≤ fraud_probability ≤ 1."""
    response = await client.post("/predict", json=valid_transaction)

    assert response.status_code == 200
    body = response.json()

    assert "fraud_probability" in body
    prob = body["fraud_probability"]
    assert 0.0 <= prob <= 1.0, f"fraud_probability {prob!r} out of [0, 1]"

    # Structural checks on the full response schema
    assert "is_fraud" in body
    assert "model_version" in body
    assert "prediction_id" in body
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_predict_invalid_amount(
    client: AsyncClient, valid_transaction: dict
) -> None:
    """POST /predict with negative Amount → 422 Unprocessable Entity."""
    payload = {**valid_transaction, "Amount": -50.0}
    response = await client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_field(
    client: AsyncClient, valid_transaction: dict
) -> None:
    """POST /predict with V1 omitted → 422 Unprocessable Entity."""
    payload = {k: v for k, v in valid_transaction.items() if k != "V1"}
    response = await client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient, valid_transaction: dict) -> None:
    """
    GET /metrics → 200 and response contains 'count'.

    We make one prediction first so the metrics window is non-empty and
    we can also assert that count > 0.
    """
    # Seed at least one prediction into the rolling window
    predict_resp = await client.post("/predict", json=valid_transaction)
    assert predict_resp.status_code == 200

    response = await client.get("/metrics")
    assert response.status_code == 200
    body = response.json()

    assert "count" in body, f"Key 'count' missing from: {body}"
    assert body["count"] >= 1
