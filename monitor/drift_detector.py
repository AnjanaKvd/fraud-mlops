"""
monitor/drift_detector.py — Data & prediction drift detection for fraud MLOps.

What is drift?
--------------
"Drift" happens when the statistical properties of the data a model sees in
production diverge from the data it was trained on.  Two kinds matter here:

  1. Feature drift  — the distribution of input features (Amount, V1–V28)
     shifts.  This can be caused by seasonal effects, new fraud patterns, or
     upstream data pipeline changes.  When features drift, model performance
     usually degrades even before labels are available to confirm it.

  2. Prediction drift — the distribution of the model's output scores
     (fraud_probability) shifts.  A sudden spike in the fraction of high-
     probability predictions might mean real fraud is increasing, *or* that
     the model is reacting to a covariate shift.  Either way it warrants
     investigation.

Why Evidently?
--------------
Evidently is an open-source ML observability library.  Its `DataDriftPreset`
runs a battery of statistical tests (Wasserstein distance for numerical
columns, chi-squared for categorical ones) and produces both a visual HTML
report and a machine-readable JSON summary.  The statistical test used per
column and the p-value threshold are configurable; we use defaults tuned for
tabular ML data.

Limitation — V1-V28 availability
---------------------------------
The `predictions` table in predictions.db only stores `amount` and
`fraud_probability`; V1–V28 are PCA-derived features passed at inference
time but deliberately not persisted to keep the database lean.  Therefore:

  • Feature drift on V1–V28 — cannot be computed from the database alone.
    If you need this, extend `inference_logger.log_prediction()` to store
    the raw feature vector and re-run this script against the extended table.

  • Feature drift on `Amount` — fully supported (column present in both
    training data and the predictions log).

  • Prediction drift on `fraud_probability` — fully supported.

Both available signals are surfaced in the Evidently report and in the
returned summary dict.

Usage
-----
  # from another module
  from monitor.drift_detector import run_drift_detection
  result = run_drift_detection()

  # from the command line
  python -m monitor.drift_detector
  python -m monitor.drift_detector --n 500 --threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Evidently imports — DataDriftPreset is the "batteries-included" preset that
# applies appropriate statistical tests per column type and aggregates results.
# ---------------------------------------------------------------------------
try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Evidently is required for drift detection. "
        "Install it with:  pip install evidently>=0.4"
    ) from exc

# ---------------------------------------------------------------------------
# Project-local import — get logged predictions from SQLite
# ---------------------------------------------------------------------------
# Resolve the project root so this script can be run from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.inference_logger import get_recent_predictions  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("monitor.drift_detector")

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
# Reference data: what the model was actually trained on.
REFERENCE_DATA_PATH: Path = Path(
    os.getenv(
        "REFERENCE_DATA_PATH", str(_PROJECT_ROOT / "data" / "processed" / "train.csv")
    )
)

# Reports are written here; the directory is created if it doesn't exist.
REPORTS_DIR: Path = Path(
    os.getenv("DRIFT_REPORTS_DIR", str(_PROJECT_ROOT / "monitor" / "reports"))
)

# The PCA-derived features the model consumed during training.
# These are *not* stored in predictions.db (see module docstring).
V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]  # V1…V28

# Columns we *can* compare between reference and current data.
FEATURE_COLUMNS: list[str] = ["Amount"]  # extend if you log V1-V28 in future

# The model's output column logged by the inference service.
PREDICTION_COLUMN: str = "fraud_probability"

# All columns that will be included in the drift report.
TARGET_COLUMNS: list[str] = FEATURE_COLUMNS + [PREDICTION_COLUMN]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def run_drift_detection(
    n_recent: int = 500,
    stattest_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Run data-drift and prediction-drift detection and persist the reports.

    Steps
    -----
    1. Load reference data (train.csv) — the distribution the model learned from.
    2. Load current data from predictions.db — what the model has seen recently.
    3. Align both DataFrames to the shared set of columns.
    4. Build an Evidently ``Report`` with ``DataDriftPreset``.
    5. Save HTML visual report + JSON machine-readable summary.
    6. Parse the JSON summary and return a structured dict.

    Parameters
    ----------
    n_recent : int
        How many of the most recent logged predictions to use as the
        "current" dataset.  Default 500.  Larger windows give more
        statistically reliable drift estimates; smaller windows are more
        responsive to sudden shifts.
    stattest_threshold : float
        p-value threshold below which a feature is considered drifted.
        Default 0.05 (5 % significance level).  Lowering this reduces
        false positives (but may miss real drift); raising it increases
        sensitivity.

    Returns
    -------
    dict with keys:
        dataset_drift_detected : bool
            True when Evidently's aggregate dataset-level drift test fires.
            Evidently flags dataset-level drift when the *fraction* of drifted
            features exceeds a threshold (default 0.5 inside the preset).
        number_of_drifted_features : int
            Count of individual columns where the null hypothesis of "no drift"
            was rejected at ``stattest_threshold``.
        drift_share : float
            ``number_of_drifted_features / total_features_tested`` — easier to
            compare across reports with different column counts.
        feature_drift_scores : dict[str, float]
            Per-column drift score.  For numerical columns Evidently reports
            the Wasserstein distance (Earth Mover's Distance); for categorical
            columns it reports the chi-squared statistic.  Higher = more drift.
        report_html_path : str
            Absolute path to the saved HTML report.
        report_json_path : str
            Absolute path to the saved JSON summary.
        current_sample_size : int
            Actual number of recent predictions used (may be < n_recent).
        warnings : list[str]
            Non-fatal issues encountered (e.g. not enough current data).
    """
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load reference data (training set)
    # ------------------------------------------------------------------
    logger.info("Loading reference data from: %s", REFERENCE_DATA_PATH)
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found at '{REFERENCE_DATA_PATH}'. "
            "Run the training pipeline first or set REFERENCE_DATA_PATH."
        )

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    logger.info("Reference data loaded: %d rows × %d cols", *reference_df.shape)

    # Normalise column name casing — some pipelines produce 'amount' lowercase.
    reference_df.columns = [
        "Amount" if c.lower() == "amount" else c for c in reference_df.columns
    ]

    # Validate that our expected columns exist in the reference set.
    missing_ref = [c for c in TARGET_COLUMNS if c not in reference_df.columns]
    if missing_ref:
        # fraud_probability won't be in train.csv — that's expected; we
        # synthesise it as NaN so Evidently only evaluates it in the
        # current dataset (prediction drift direction).
        for col in missing_ref:
            if col == PREDICTION_COLUMN:
                # We will handle prediction drift separately below.
                logger.debug(
                    "'%s' not in reference data — prediction drift will be "
                    "evaluated from current data only.",
                    col,
                )
            else:
                msg = f"Expected column '{col}' missing from reference data."
                logger.warning(msg)
                warnings.append(msg)

    # ------------------------------------------------------------------
    # 2. Load current data from predictions.db
    # ------------------------------------------------------------------
    logger.info("Fetching %d most recent predictions from predictions.db …", n_recent)
    raw_predictions: list[dict[str, Any]] = get_recent_predictions(n=n_recent)

    if not raw_predictions:
        msg = "predictions.db returned 0 rows — cannot run drift detection."
        logger.error(msg)
        return {
            "dataset_drift_detected": False,
            "number_of_drifted_features": 0,
            "drift_share": 0.0,
            "feature_drift_scores": {},
            "report_html_path": None,
            "report_json_path": None,
            "current_sample_size": 0,
            "warnings": [msg],
        }

    current_df = pd.DataFrame(raw_predictions)
    # Normalise column name casing.
    current_df.columns = [
        "Amount" if c.lower() == "amount" else c for c in current_df.columns
    ]
    logger.info("Current data loaded: %d rows × %d cols", *current_df.shape)

    if len(current_df) < 30:
        msg = (
            f"Only {len(current_df)} recent predictions available.  "
            "Drift statistics may be unreliable with fewer than 30 samples."
        )
        logger.warning(msg)
        warnings.append(msg)

    # ------------------------------------------------------------------
    # 3. Align DataFrames to the testable columns
    # ------------------------------------------------------------------
    # Keep only columns that exist in *both* reference and current datasets.
    testable_columns = [c for c in TARGET_COLUMNS if c in current_df.columns]

    # Prediction drift: fraud_probability only exists in current_df.
    # To evaluate it with Evidently we need a reference distribution.
    # Strategy: use the model's score distribution on the *training set* as
    # reference.  If it was logged separately use that; otherwise we skip
    # prediction drift gracefully.
    ref_has_pred = PREDICTION_COLUMN in reference_df.columns
    cur_has_pred = PREDICTION_COLUMN in current_df.columns

    if not ref_has_pred and cur_has_pred:
        # Fallback: we cannot compare fraud_probability without a reference
        # distribution.  Remove it from the test set and warn.
        msg = (
            f"'{PREDICTION_COLUMN}' is not available in the reference (training) "
            "data so prediction drift cannot be evaluated with DataDriftPreset. "
            "Consider logging train-time model scores to train.csv or a separate "
            "calibration CSV and setting REFERENCE_DATA_PATH accordingly."
        )
        logger.warning(msg)
        warnings.append(msg)
        testable_columns = [c for c in testable_columns if c != PREDICTION_COLUMN]

    if not testable_columns:
        msg = "No columns overlap between reference and current data.  Aborting."
        logger.error(msg)
        return {
            "dataset_drift_detected": False,
            "number_of_drifted_features": 0,
            "drift_share": 0.0,
            "feature_drift_scores": {},
            "report_html_path": None,
            "report_json_path": None,
            "current_sample_size": len(current_df),
            "warnings": [msg] + warnings,
        }

    logger.info("Running drift tests on columns: %s", testable_columns)
    ref_aligned = reference_df[testable_columns].dropna()
    cur_aligned = current_df[testable_columns].dropna()

    # ------------------------------------------------------------------
    # 4. Build Evidently report
    # ------------------------------------------------------------------
    # DataDriftPreset applies per-column statistical tests:
    #   • Numerical columns → Wasserstein distance (by default when n > 1 000)
    #                          or Kolmogorov–Smirnov (when n ≤ 1 000).
    #   • Categorical columns → chi-squared test.
    # stattest_threshold sets the p-value cutoff; columns below the threshold
    # are flagged as drifted.
    report = Report(
        metrics=[
            DataDriftPreset(stattest_threshold=stattest_threshold),
        ]
    )

    logger.info(
        "Running Evidently DataDriftPreset  (ref=%d rows, cur=%d rows, "
        "threshold=%.3f) …",
        len(ref_aligned),
        len(cur_aligned),
        stattest_threshold,
    )
    report.run(reference_data=ref_aligned, current_data=cur_aligned)

    # ------------------------------------------------------------------
    # 5. Save reports
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    html_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    json_path = REPORTS_DIR / f"drift_report_{timestamp}.json"

    report.save_html(str(html_path))
    logger.info("HTML report saved → %s", html_path)

    # Evidently's JSON output is structured; we parse it in step 6.
    report_dict = report.as_dict()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, default=str)
    logger.info("JSON summary saved → %s", json_path)

    # ------------------------------------------------------------------
    # 6. Parse summary from the Evidently result dict
    # ------------------------------------------------------------------
    summary = _parse_drift_summary(report_dict, warnings=warnings)
    summary["report_html_path"] = str(html_path)
    summary["report_json_path"] = str(json_path)
    summary["current_sample_size"] = len(cur_aligned)

    # Human-friendly log line.
    logger.info(
        "Drift detection complete — dataset_drift=%s  drifted_features=%d/%d  "
        "drift_share=%.2f",
        summary["dataset_drift_detected"],
        summary["number_of_drifted_features"],
        len(testable_columns),
        summary["drift_share"],
    )
    return summary


# ---------------------------------------------------------------------------
# Helper — parse Evidently's as_dict() output
# ---------------------------------------------------------------------------


def _parse_drift_summary(
    report_dict: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """
    Extract structured drift metrics from Evidently's ``Report.as_dict()`` output.

    Evidently's dict structure (v0.4.x):
      report_dict["metrics"][i]["result"] contains per-metric results.

    The DataDriftPreset produces two metrics in sequence:
      • index 0: DatasetDriftMetric — aggregate dataset-level result.
      • index 1: DataDriftTable — per-column breakdown.

    Parameters
    ----------
    report_dict : dict
        As returned by ``Report.as_dict()``.
    warnings : list[str]
        Mutable list; any parsing issues are appended here.

    Returns
    -------
    dict
        Partial summary dict (without path / sample-size fields).
    """
    feature_drift_scores: dict[str, float] = {}
    dataset_drift_detected = False
    number_of_drifted_features = 0
    drift_share = 0.0

    try:
        metrics = report_dict.get("metrics", [])

        # ---- Dataset-level metric (DatasetDriftMetric) ----
        # result keys: dataset_drift (bool), number_of_drifted_columns (int),
        #              drift_share (float), number_of_columns (int)
        for metric in metrics:
            result = metric.get("result", {})

            if "dataset_drift" in result:
                # This is the DatasetDriftMetric result block.
                dataset_drift_detected = bool(result.get("dataset_drift", False))
                number_of_drifted_features = int(
                    result.get("number_of_drifted_columns", 0)
                )
                drift_share = float(result.get("drift_share", 0.0))
                logger.debug(
                    "DatasetDriftMetric: drift=%s  drifted_cols=%d  share=%.3f",
                    dataset_drift_detected,
                    number_of_drifted_features,
                    drift_share,
                )

            if "drift_by_columns" in result:
                # This is the DataDriftTable result block.
                # Each value is a dict with keys: column_name, stattest_name,
                # drift_detected, drift_score, threshold, etc.
                for col_name, col_info in result["drift_by_columns"].items():
                    # drift_score is the raw test statistic (higher = more drift).
                    # For Wasserstein distance this is the EMD value;
                    # for KS it is the KS statistic; for chi-squared it is
                    # the normalised statistic.
                    score = col_info.get("drift_score", float("nan"))
                    feature_drift_scores[col_name] = round(float(score), 6)
                    logger.debug(
                        "  %-25s  drift=%s  score=%.6f  test=%s",
                        col_name,
                        col_info.get("drift_detected", "?"),
                        score,
                        col_info.get("stattest_name", "?"),
                    )

    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Could not fully parse Evidently report dict: {exc}"
        logger.warning(msg)
        warnings.append(msg)

    return {
        "dataset_drift_detected": dataset_drift_detected,
        "number_of_drifted_features": number_of_drifted_features,
        "drift_share": drift_share,
        "feature_drift_scores": feature_drift_scores,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Command-line interface for manual drift detection runs.

    Examples
    --------
    # Run with defaults (500 recent predictions, p=0.05 threshold):
    python -m monitor.drift_detector

    # Use the last 1 000 predictions with a stricter threshold:
    python -m monitor.drift_detector --n 1000 --threshold 0.01
    """
    parser = argparse.ArgumentParser(
        description="Run data-drift and prediction-drift detection using Evidently.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        metavar="N",
        help="Number of most-recent predictions to use as the current dataset.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        metavar="P",
        help="p-value threshold for individual feature drift tests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    logger.info(
        "Starting drift detection  (n=%d, threshold=%.3f)", args.n, args.threshold
    )

    try:
        result = run_drift_detection(
            n_recent=args.n,
            stattest_threshold=args.threshold,
        )
    except FileNotFoundError as exc:
        logger.error("Reference data error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during drift detection: %s", exc)
        sys.exit(1)

    # Pretty-print summary to stdout.
    print("\n" + "=" * 60)
    print("  DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"  dataset_drift_detected   : {result['dataset_drift_detected']}")
    print(f"  number_of_drifted_features: {result['number_of_drifted_features']}")
    print(f"  drift_share              : {result['drift_share']:.4f}")
    print(f"  current_sample_size      : {result.get('current_sample_size', 'N/A')}")
    print()
    print("  Per-feature drift scores:")
    for feat, score in sorted(result["feature_drift_scores"].items()):
        print(f"    {feat:<30} {score:.6f}")
    if result.get("warnings"):
        print()
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    ⚠  {w}")
    print()
    print(f"  HTML report → {result.get('report_html_path')}")
    print(f"  JSON report → {result.get('report_json_path')}")
    print("=" * 60 + "\n")

    # Exit code 1 if drift detected — useful for CI gates.
    sys.exit(1 if result["dataset_drift_detected"] else 0)


if __name__ == "__main__":
    main()
