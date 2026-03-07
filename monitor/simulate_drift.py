# -*- coding: utf-8 -*-
"""
monitor/simulate_drift.py -- Portfolio demo: simulate data drift end-to-end.

*** FOR DEMONSTRATION PURPOSES ONLY ***
This script artificially manipulates transaction data to demonstrate the
drift-detection pipeline working in a controlled way.  Do NOT run this
against a production database.

What it proves
--------------
This demo is designed to convince a reviewer or interviewer that:

  1. The inference pipeline (FastAPI /predict) works correctly.
  2. The monitoring pipeline (Evidently drift detection) correctly identifies
     when input distributions shift.
  3. The retraining trigger fires at the right threshold.

Three-phase structure
---------------------
Phase 1 — Normal traffic
    Send 500 rows sampled from test.csv (the held-out set) through /predict.
    These rows are drawn from the same distribution the model was trained on,
    so drift detection should report NO DRIFT.

Phase 2 — Drifted traffic
    Send 500 rows that have been deliberately perturbed:
      • Amount × 10  — simulates inflation / different merchant category.
      • V1, V2, V3 += N(0, 2)  — simulates changing fraud patterns in PCA space.
      • 20 % rows forced to receive altered feature vectors that typically
        appear in fraud cases (higher amounts, extreme V1/V2).
    The model will still return predictions; drift detection should now flag
    DRIFT DETECTED with a significantly higher drift_share.

Phase 3 — Side-by-side comparison table
    Prints a formatted table contrasting the before/after metrics:
      • Per-feature drift score (Amount only, since V1-V28 aren't logged in DB)
      • drift_share
      • dataset_drift_detected
      • Whether the retraining trigger fired

Usage
-----
  # API must be running first (docker compose up or uvicorn app.main:app)
  python -m monitor.simulate_drift
  python -m monitor.simulate_drift --api-url http://localhost:8000 --n 200
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# tqdm — optional, falls back to a no-op if not installed.
try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

    class tqdm:  # type: ignore[no-redef]
        """Minimal no-op tqdm replacement so the script runs without it."""

        def __init__(self, iterable=None, **kwargs):
            self._iter = iter(iterable) if iterable is not None else iter([])
            self.desc = kwargs.get("desc", "")

        def __iter__(self):
            return self._iter

        def set_postfix(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


try:
    import requests
except ImportError as exc:
    raise ImportError(
        "The 'requests' library is required.  Install it with:  pip install requests"
    ) from exc

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monitor.drift_detector import run_drift_detection  # noqa: E402
from monitor.retrain_trigger import check_and_trigger  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("monitor.simulate_drift")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_DATA_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "test.csv"

# Feature columns that the API accepts.
V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES: list[str] = V_FEATURES + ["Amount"]

# API endpoint
DEFAULT_API_URL: str = "http://localhost:8000"
PREDICT_ENDPOINT: str = "/predict"

# Sleep between requests to avoid overwhelming the API
REQUEST_SLEEP_S: float = 0.1


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _load_test_sample(n: int, seed: int = 42) -> pd.DataFrame:
    """Load n random rows from test.csv, keeping only model input columns.

    Rows with Amount < 0 are dropped — the API schema enforces Amount >= 0
    via a Pydantic field validator (ge=0) and returns HTTP 422 otherwise.
    """
    logger.info("Loading test data from %s ...", TEST_DATA_PATH)
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found at '{TEST_DATA_PATH}'. "
            "Run the training pipeline first."
        )
    df = pd.read_csv(TEST_DATA_PATH, nrows=5000)  # read a chunk for sampling speed

    # Drop rows where Amount is negative — the /predict endpoint rejects them
    # with HTTP 422 (Pydantic ge=0 constraint on TransactionInput.Amount).
    before = len(df)
    df = df[df["Amount"] >= 0]
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with Amount < 0 (API rejects negatives).", dropped)

    sample = df.sample(n=min(n, len(df)), random_state=seed)[ALL_FEATURES].reset_index(
        drop=True
    )
    logger.info("Loaded %d rows for simulation.", len(sample))
    return sample


def _inject_drift(
    df: pd.DataFrame, fraud_fraction: float = 0.20, seed: int = 99
) -> pd.DataFrame:
    """
    Return a copy of *df* with deliberate distribution shifts applied.

    Modifications
    -------------
    Amount × 10
        Simulates an economic shock — e.g. a merchant category processing
        10× larger transactions than training-time data.  Amount is the
        most interpretable feature; the Wasserstein distance between the
        original and inflated distributions will be very large.

    V1, V2, V3 += Normal(0, 2)
        PCA components don't have a real-world unit, so we perturb them
        with Gaussian noise.  Standard deviation = 2 is chosen to be
        large relative to the typical range (≈ [-5, 5]) so the KS test
        picks up the shift reliably within 500 samples.

    20 % of rows set to extreme-fraud-like values
        For a visible and dramatic effect, one in five rows is overwritten
        with a profile that strongly resembles known fraud transactions:
        large Amount, extreme negative V1 (typical fraud signature).
    """
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    # --- 1. Inflate Amount ---
    drifted["Amount"] = drifted["Amount"] * 10

    # --- 2. Add noise to first three PCA components ---
    for v in ["V1", "V2", "V3"]:
        noise = rng.normal(loc=0, scale=2.0, size=len(drifted))
        drifted[v] = drifted[v] + noise

    # --- 3. Inject 20 % "fraud-like" rows ---
    fraud_idx = rng.choice(
        len(drifted), size=int(len(drifted) * fraud_fraction), replace=False
    )
    # Fraud-typical signature: extreme negative V1, high V2, large Amount.
    drifted.loc[fraud_idx, "V1"] = rng.normal(loc=-5.0, scale=0.5, size=len(fraud_idx))
    drifted.loc[fraud_idx, "V2"] = rng.normal(loc=3.5, scale=0.5, size=len(fraud_idx))
    drifted.loc[fraud_idx, "Amount"] = rng.uniform(1_000, 5_000, size=len(fraud_idx))

    logger.info(
        "Drift injected: Amount*10, V1/V2/V3 noise added, "
        "%d rows set to fraud-like values.",
        len(fraud_idx),
    )
    return drifted


def _preflight_check(api_url: str, sample_row: "pd.Series") -> bool:
    """
    Send a single test request to /predict and log the full response.

    Returns True if the call succeeds, False otherwise.
    Used to surface the real error reason before the full 500-row loop.
    """
    endpoint = f"{api_url.rstrip('/')}{PREDICT_ENDPOINT}"
    payload = {col: float(sample_row[col]) for col in ALL_FEATURES}
    logger.info("Preflight check: POST %s", endpoint)
    try:
        resp = requests.post(endpoint, json=payload, timeout=10)
        if resp.ok:
            logger.info(
                "Preflight PASSED  status=%d  body=%s",
                resp.status_code,
                resp.text[:200],
            )
            return True
        else:
            logger.error(
                "Preflight FAILED  status=%d\n  URL: %s\n  Response body: %s",
                resp.status_code,
                endpoint,
                resp.text,
            )
            return False
    except requests.exceptions.ConnectionError as exc:
        logger.error(
            "Preflight FAILED — cannot connect to %s\n  Error: %s\n"
            "  Is the API running? Try: docker compose up",
            endpoint,
            exc,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Preflight FAILED — unexpected error: %s", exc)
        return False


def _send_predictions(
    df: pd.DataFrame,
    api_url: str,
    phase_label: str,
    sleep_s: float = REQUEST_SLEEP_S,
) -> dict[str, int]:
    """
    POST each row in *df* to the /predict endpoint.

    Returns
    -------
    dict with keys ``success``, ``error`` — counts of API call outcomes.
    """
    endpoint = f"{api_url.rstrip('/')}{PREDICT_ENDPOINT}"
    counts = {"success": 0, "error": 0}
    # Log the first few HTTP errors in full so the cause is clear.
    _max_logged_errors = 3

    desc = f"[{phase_label}] Sending to /predict"
    rows_iter = tqdm(df.iterrows(), total=len(df), desc=desc, unit="req")

    for _, row in rows_iter:
        payload = {col: float(row[col]) for col in ALL_FEATURES}
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            response.raise_for_status()
            counts["success"] += 1
        except requests.exceptions.ConnectionError as exc:
            counts["error"] += 1
            if counts["error"] <= _max_logged_errors:
                logger.error(
                    "Connection error — is the API running at %s?\n  %s", endpoint, exc
                )
        except requests.exceptions.HTTPError as exc:
            counts["error"] += 1
            if counts["error"] <= _max_logged_errors:
                # Log the full response body so we can diagnose the failure.
                body = exc.response.text if exc.response is not None else "(no body)"
                logger.error(
                    "HTTP %s from POST %s\n  Response body: %s",
                    exc.response.status_code if exc.response is not None else "?",
                    endpoint,
                    body,
                )
        except Exception as exc:  # noqa: BLE001
            counts["error"] += 1
            if counts["error"] <= _max_logged_errors:
                logger.error("Unexpected request error: %s", exc)

        rows_iter.set_postfix(ok=counts["success"], err=counts["error"])
        time.sleep(sleep_s)

    logger.info(
        "[%s] Done -- %d successful, %d errors out of %d requests.",
        phase_label,
        counts["success"],
        counts["error"],
        len(df),
    )
    return counts


def _run_drift_phase(n_recent: int, label: str) -> dict[str, Any]:
    """Run drift detection and return the result dict, logging the phase label."""
    logger.info(
        "[%s] Running drift detection on last %d predictions …", label, n_recent
    )
    result = run_drift_detection(n_recent=n_recent)
    logger.info(
        "[%s] drift_share=%.4f  dataset_drift=%s  drifted_features=%d",
        label,
        result.get("drift_share", 0.0),
        result.get("dataset_drift_detected"),
        result.get("number_of_drifted_features", 0),
    )
    return result


def _print_comparison_table(
    before: dict[str, Any],
    after: dict[str, Any],
    before_triggered: bool,
    after_triggered: bool,
) -> None:
    """
    Print a side-by-side comparison of before and after drift metrics.
    """
    # Collect all feature names across both results.
    all_features = sorted(
        set(before.get("feature_drift_scores", {}).keys())
        | set(after.get("feature_drift_scores", {}).keys())
    )

    col_w = 30  # feature column width
    val_w = 18  # value column width

    sep = (
        "+"
        + "-" * (col_w + 2)
        + "+"
        + "-" * (val_w + 2)
        + "+"
        + "-" * (val_w + 2)
        + "+"
    )
    header = (
        f"| {'METRIC':<{col_w}} | "
        f"{'PHASE 1 (Normal)':<{val_w}} | "
        f"{'PHASE 2 (Drifted)':<{val_w}} |"
    )

    print()
    print("=" * len(sep))
    print("   DRIFT SIMULATION -- BEFORE vs AFTER COMPARISON")
    print("=" * len(sep))
    print(sep)
    print(header)
    print(sep)

    def _row(
        label: str, before_val: Any, after_val: Any, highlight: bool = False
    ) -> str:
        arrow = " *** " if highlight else "     "
        return (
            f"| {label:<{col_w}} | "
            f"{str(before_val):<{val_w}} | "
            f"{str(after_val):<{val_w}} |"
            f"{arrow}"
        )

    # Summary metrics
    b_share = f"{before.get('drift_share', 0.0):.4f}"
    a_share = f"{after.get('drift_share', 0.0):.4f}"
    share_drifted = after.get("drift_share", 0.0) > before.get("drift_share", 0.0)

    print(_row("drift_share", b_share, a_share, highlight=share_drifted))
    print(
        _row(
            "dataset_drift_detected",
            before.get("dataset_drift_detected"),
            after.get("dataset_drift_detected"),
            highlight=after.get("dataset_drift_detected", False),
        )
    )
    print(
        _row(
            "number_of_drifted_features",
            before.get("number_of_drifted_features", 0),
            after.get("number_of_drifted_features", 0),
        )
    )
    print(
        _row(
            "retrain_trigger_fired",
            before_triggered,
            after_triggered,
            highlight=after_triggered,
        )
    )
    print(
        _row(
            "current_sample_size",
            before.get("current_sample_size", "N/A"),
            after.get("current_sample_size", "N/A"),
        )
    )

    # Per-feature drift scores
    if all_features:
        print(sep)
        print(
            f"| {'  Per-feature drift scores':<{col_w}} | {'':<{val_w}} | {'':<{val_w}} |"
        )
        print(sep)
        before_scores = before.get("feature_drift_scores", {})
        after_scores = after.get("feature_drift_scores", {})
        for feat in all_features:
            b_score = (
                f"{before_scores.get(feat, 0.0):.6f}"
                if feat in before_scores
                else "N/A"
            )
            a_score = (
                f"{after_scores.get(feat, 0.0):.6f}" if feat in after_scores else "N/A"
            )
            drifted = after_scores.get(feat, 0.0) > before_scores.get(feat, 0.0) * 2
            print(_row(f"  {feat}", b_score, a_score, highlight=drifted))

    # Report paths
    print(sep)
    print(f"| {'  Reports':<{col_w}} | {'':<{val_w}} | {'':<{val_w}} |")
    print(sep)
    b_html = Path(before.get("report_html_path") or "—").name
    a_html = Path(after.get("report_html_path") or "—").name
    print(_row("  HTML report (filename)", b_html, a_html))
    print(sep)
    print()

    # Final verdict
    if after.get("dataset_drift_detected"):
        print(
            "[DRIFT] VERDICT: Drift injection SUCCESSFUL - Evidently detected distribution shift."
        )
    else:
        print("[OK]    VERDICT: No significant drift detected after injection.")

    if after_triggered:
        print("[WARNING] Retraining trigger FIRED (drift_share exceeded threshold).")
    else:
        print("[OK]      Retraining trigger did NOT fire.")
    print()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_simulation(
    n: int = 500,
    api_url: str = DEFAULT_API_URL,
    sleep_s: float = REQUEST_SLEEP_S,
    drift_threshold: float = 0.3,
) -> None:
    """
    Run the full three-phase drift simulation.

    Parameters
    ----------
    n : int
        Number of rows per phase. Default 500.
    api_url : str
        Base URL of the running FastAPI service.
    sleep_s : float
        Seconds to sleep between consecutive API requests.
    drift_threshold : float
        Passed to check_and_trigger(); fraction of features above which
        retraining is triggered.
    """
    print()
    print("=" * 56)
    print("  FRAUD MLOps -- DRIFT SIMULATION DEMO")
    print("=" * 56)
    print()

    # -----------------------------------------------------------------------
    # Load test data once; create drifted copy
    # -----------------------------------------------------------------------
    normal_df = _load_test_sample(n=n, seed=42)
    drifted_df = _inject_drift(normal_df.copy(), fraud_fraction=0.20, seed=99)

    # -----------------------------------------------------------------------
    # Preflight — test one request before sending the full batch
    # -----------------------------------------------------------------------
    print("  Running preflight check (1 test request) ...")
    if not _preflight_check(api_url, normal_df.iloc[0]):
        print()
        print("[ERROR] Preflight failed. See log output above for the exact error.")
        print(f"  API URL : {api_url}")
        print("  Tips:")
        print("    1. Check docker compose logs for startup / model-load errors.")
        print("    2. Try: curl http://localhost:8000/health")
        print(
            "    3. Try: curl -X POST http://localhost:8000/predict "
            "-H 'Content-Type: application/json' -d '{\"V1\":0,...}'"
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 1 -- Normal traffic
    # -----------------------------------------------------------------------
    print("-" * 56)
    print("  PHASE 1 -- Normal traffic (no drift expected)")
    print("-" * 56)

    counts_1 = _send_predictions(
        normal_df, api_url, phase_label="Phase 1", sleep_s=sleep_s
    )

    if counts_1["success"] == 0:
        print()
        print("[ERROR] No successful API calls in Phase 1.")
        print(f"   Make sure the API is running at {api_url}")
        print("   Start it with:  docker compose up  or  uvicorn app.main:app --reload")
        sys.exit(1)

    print(f"\n   [OK] {counts_1['success']} predictions logged.\n")

    print("  Running drift detection on Phase 1 data …")
    before_result = _run_drift_phase(n_recent=counts_1["success"], label="Phase 1")
    before_triggered = check_and_trigger(
        n_recent=counts_1["success"],
        drift_threshold=drift_threshold,
    )

    # -----------------------------------------------------------------------
    # Phase 2 — Drifted traffic
    # -----------------------------------------------------------------------
    print()
    print("-" * 56)
    print("  PHASE 2 -- Injecting drift (drift expected)")
    print("-" * 56)
    print()
    print("  Drift injections applied to the 500-row batch:")
    print("    - Amount * 10  (simulates economic/merchant-type shift)")
    print("    - V1, V2, V3 += Normal(0, 2)  (PCA space perturbation)")
    print("    - 20 % of rows set to fraud-like extreme values")
    print()

    counts_2 = _send_predictions(
        drifted_df, api_url, phase_label="Phase 2", sleep_s=sleep_s
    )
    print(f"\n   [OK] {counts_2['success']} predictions logged.\n")

    print("  Running drift detection on Phase 2 data …")
    # Use the most recent n predictions (= Phase 2 batch) for "current" data.
    after_result = _run_drift_phase(n_recent=counts_2["success"], label="Phase 2")
    after_triggered = check_and_trigger(
        n_recent=counts_2["success"],
        drift_threshold=drift_threshold,
    )

    # -----------------------------------------------------------------------
    # Phase 3 — Comparison report
    # -----------------------------------------------------------------------
    print()
    print("-" * 56)
    print("  PHASE 3 -- Side-by-side comparison")
    print("-" * 56)

    _print_comparison_table(
        before=before_result,
        after=after_result,
        before_triggered=before_triggered,
        after_triggered=after_triggered,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate data drift end-to-end for the fraud detection MLOps demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        metavar="N",
        help="Number of rows to send per phase.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        metavar="URL",
        help="Base URL of the running FastAPI service.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=REQUEST_SLEEP_S,
        metavar="SEC",
        help="Seconds to sleep between API requests.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.3,
        metavar="F",
        help="Fraction of drifted features that triggers retraining.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    run_simulation(
        n=args.n,
        api_url=args.api_url,
        sleep_s=args.sleep,
        drift_threshold=args.drift_threshold,
    )
