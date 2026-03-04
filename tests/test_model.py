"""
tests/test_model.py — Unit tests for the model loading and inference layer.

Strategy
--------
These tests load the *real* MLflow model from the configured tracking URI
(defaulting to ``sqlite:///mlruns.db`` in the project root).  They are
intended to be run in an environment where the training pipeline has already
been executed and a Production model version exists in the registry.

If no Production model is available the tests are automatically skipped
rather than failing, so CI pipelines that haven't run training yet stay green.

Run with:
    pytest tests/test_model.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Number of V features in the dataset
N_V_FEATURES = 28
ALL_COLUMNS = [f"V{i}" for i in range(1, N_V_FEATURES + 1)] + ["Amount"]


def _make_transaction_df(n: int = 1) -> pd.DataFrame:
    """
    Build a DataFrame of *n* synthetic transactions.

    Values are random floats drawn from realistic ranges:
      • V1–V28: uniform in [-5, 5]  (PCA-compressed features)
      • Amount:  uniform in (0, 500] (transaction amount in any currency)
    """
    rng = np.random.default_rng(seed=42)
    v_cols = rng.uniform(-5.0, 5.0, size=(n, N_V_FEATURES))
    amounts = rng.uniform(0.01, 500.0, size=(n, 1))
    data = np.hstack([v_cols, amounts])
    return pd.DataFrame(data, columns=ALL_COLUMNS)


# ---------------------------------------------------------------------------
# Skip if no Production model is available
# ---------------------------------------------------------------------------


def _try_load_model():
    """
    Attempt to load the Production model; return it or raise/skip.

    Wrapped in a helper so the skip logic lives in one place and each
    individual test can call it independently (useful when tests are
    collected in isolation).
    """
    # Import here so model_loader's module-level MLflow config runs first.
    from app.model_loader import load_model_on_startup, get_model, get_model_version  # noqa: PLC0415

    try:
        load_model_on_startup()
    except RuntimeError as exc:
        pytest.skip(f"No Production model found in MLflow registry — skipping: {exc}")

    return get_model(), get_model_version()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_loads() -> None:
    """
    The model loader completes without raising an exception.

    Verifies that:
    • ``load_model_on_startup()`` can reach the MLflow tracking server.
    • A Production version of the model exists in the registry.
    • ``get_model()`` returns a non-None object.
    • ``get_model_version()`` returns a non-empty string.
    """
    model, version = _try_load_model()

    assert model is not None, "get_model() returned None"
    assert isinstance(version, str) and version, (
        f"get_model_version() returned an empty or non-string value: {version!r}"
    )


def test_model_predicts() -> None:
    """
    A single-row prediction yields a fraud probability in [0, 1].

    This exercises the full inference path:
    1. Build a one-row DataFrame with all required features.
    2. Call model.predict().
    3. Extract the fraud probability exactly as main.py does.
    4. Assert the probability is within the valid range.
    """
    model, _ = _try_load_model()

    df = _make_transaction_df(n=1)
    raw_output = model.predict(df)

    # Mirror the normalisation logic from app/main.py so this test stays
    # in sync with the production code path.
    if hasattr(raw_output, "iloc"):
        fraud_prob = float(raw_output.iloc[0])
    elif hasattr(raw_output, "__len__") and len(raw_output) > 0:
        prob_row = raw_output[0]
        fraud_prob = (
            float(prob_row[1]) if hasattr(prob_row, "__len__") else float(prob_row)
        )
    else:
        fraud_prob = float(raw_output)

    assert 0.0 <= fraud_prob <= 1.0, (
        f"fraud_probability {fraud_prob!r} is outside [0, 1]"
    )


def test_model_output_shape() -> None:
    """
    A batch of 10 rows produces exactly 10 probability values.

    Guarantees that the model handles batch inputs correctly and that the
    output has the expected shape (10,) or equivalent iterable of length 10.
    """
    model, _ = _try_load_model()

    df = _make_transaction_df(n=10)
    raw_output = model.predict(df)

    # The output may be an ndarray of shape (10,) or (10, 2).
    # Either way the first dimension must equal the number of input rows.
    output_arr = np.asarray(raw_output)
    assert output_arr.shape[0] == 10, (
        f"Expected first dimension 10, got shape {output_arr.shape}"
    )
