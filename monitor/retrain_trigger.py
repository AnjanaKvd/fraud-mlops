"""
monitor/retrain_trigger.py — Automated retraining trigger for fraud detection.

How this fits into the MLOps pipeline
--------------------------------------
Model performance silently degrades when the statistical distribution of
production traffic shifts away from the training distribution ("data drift").
This module is the action layer that sits on top of drift_detector.py:

  drift_detector.py  →  quantifies drift
  retrain_trigger.py →  decides whether to act on it

Flow:
  1. Run drift detection (via run_drift_detection()).
  2. If drift_share exceeds DRIFT_THRESHOLD, write a trigger file and log a
     warning — an external process (CI job, Airflow task, or cron script)
     polls for that file and kicks off the retraining pipeline.
  3. If no drift, confirm model health.

Production scheduling (cron / Airflow)
-----------------------------------------
In production you would schedule this script to run periodically, e.g.:

  Cron (Linux/Mac):
    # Every hour, at minute 0
    0 * * * * cd /app && python -m monitor.retrain_trigger >> /var/log/drift.log 2>&1

  Windows Task Scheduler:
    Program : python.exe
    Args    : -m monitor.retrain_trigger
    Schedule: Hourly

  Airflow DAG (sketch):
    from airflow.operators.python import PythonOperator
    from monitor.retrain_trigger import check_and_trigger

    trigger_task = PythonOperator(
        task_id="check_drift_trigger",
        python_callable=check_and_trigger,
        dag=dag,
    )

  GitHub Actions (example — runs daily at 02:00 UTC):
    on:
      schedule:
        - cron: '0 2 * * *'
    jobs:
      drift-check:
        steps:
          - run: python -m monitor.retrain_trigger

The trigger file (monitor/retrain_trigger.json) is then read by the next
stage in the pipeline (e.g. a separate "retrain" job) to decide whether
to call the training scripts.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-local import
# ---------------------------------------------------------------------------
# Resolve project root so the script works regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monitor.drift_detector import run_drift_detection  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("monitor.retrain_trigger")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fraction of features that must have drifted before retraining is triggered.
# 0.3 means: if more than 30 % of the monitored features show statistically
# significant drift, we consider the model stale and schedule retraining.
# Tune this value based on your tolerance:
#   • Lower  (e.g. 0.1) → trigger retraining more aggressively (fewer false
#     negatives but more unnecessary retrains).
#   • Higher (e.g. 0.5) → only retrain when drift is severe (fewer
#     unnecessary retrains but risks delayed response to genuine shift).
DRIFT_THRESHOLD: float = 0.3

# Where to write the trigger artefact consumed by the retraining pipeline.
TRIGGER_FILE: Path = _PROJECT_ROOT / "monitor" / "retrain_trigger.json"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def check_and_trigger(
    n_recent: int = 500,
    stattest_threshold: float = 0.05,
    drift_threshold: float = DRIFT_THRESHOLD,
) -> bool:
    """
    Run drift detection and fire the retraining trigger if warranted.

    Parameters
    ----------
    n_recent : int
        Number of recent predictions passed to ``run_drift_detection``.
    stattest_threshold : float
        Per-feature p-value significance level (passed to Evidently).
    drift_threshold : float
        Fraction of features that must have drifted to trigger retraining.
        Defaults to the module-level ``DRIFT_THRESHOLD`` (0.3).

    Returns
    -------
    bool
        ``True``  — retraining was triggered (drift_share > drift_threshold).
        ``False`` — model is healthy, no action needed.

    Side effects
    ------------
    • Writes ``monitor/retrain_trigger.json`` when triggered.
    • Logs WARNING / INFO messages throughout.
    • Prints a human-readable status line to stdout.
    """
    logger.info(
        "Starting drift check  (n_recent=%d, stattest_threshold=%.3f, "
        "drift_threshold=%.2f)",
        n_recent,
        stattest_threshold,
        drift_threshold,
    )

    # ------------------------------------------------------------------
    # Step 1 — Run drift detection
    # ------------------------------------------------------------------
    try:
        drift_result: dict[str, Any] = run_drift_detection(
            n_recent=n_recent,
            stattest_threshold=stattest_threshold,
        )
    except FileNotFoundError as exc:
        logger.error("Cannot run drift detection — reference data missing: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error running drift detection: %s", exc)
        return False

    drift_share: float = drift_result.get("drift_share", 0.0)
    dataset_drift: bool = drift_result.get("dataset_drift_detected", False)
    n_drifted: int = drift_result.get("number_of_drifted_features", 0)
    feature_scores: dict[str, float] = drift_result.get("feature_drift_scores", {})

    logger.info(
        "Drift result — dataset_drift=%s  drift_share=%.4f  "
        "drifted_features=%d  threshold=%.2f",
        dataset_drift,
        drift_share,
        n_drifted,
        drift_threshold,
    )

    # ------------------------------------------------------------------
    # Step 2 — Decide whether to trigger
    # ------------------------------------------------------------------
    # We use drift_share (fraction of features drifted) rather than the
    # Evidently dataset_drift flag because:
    #   a) We can tune the threshold independently of Evidently's internal
    #      default (which is 0.5 of features drifted).
    #   b) drift_share is a continuous signal; dataset_drift is binary.
    triggered: bool = drift_share > drift_threshold

    if triggered:
        # ---------------------------------------------------------------
        # Step 2a — Log a detailed warning
        # ---------------------------------------------------------------
        logger.warning(
            "⚠️  DRIFT THRESHOLD EXCEEDED — drift_share=%.4f > threshold=%.2f.  "
            "Drifted features: %d.  Feature scores: %s",
            drift_share,
            drift_threshold,
            n_drifted,
            feature_scores,
        )

        # ---------------------------------------------------------------
        # Step 2b — Write trigger file
        # ---------------------------------------------------------------
        # Downstream systems (CI jobs, Airflow sensors, kubernetes cronjobs)
        # poll for this file.  Its presence + triggered=true means "start
        # the retraining pipeline now".
        trigger_payload: dict[str, Any] = {
            "triggered": True,
            "reason": (
                f"drift_share ({drift_share:.4f}) exceeded threshold "
                f"({drift_threshold:.2f}).  "
                f"{n_drifted} feature(s) showed statistically significant drift."
            ),
            "drift_metrics": {
                "drift_share": drift_share,
                "dataset_drift_detected": dataset_drift,
                "number_of_drifted_features": n_drifted,
                "feature_drift_scores": feature_scores,
                # Include report paths so the downstream job can attach
                # the HTML report to a Slack alert / GitHub issue.
                "report_html_path": drift_result.get("report_html_path"),
                "report_json_path": drift_result.get("report_json_path"),
                "current_sample_size": drift_result.get("current_sample_size"),
                "warnings": drift_result.get("warnings", []),
            },
            # ISO-8601 UTC timestamp — parseable by any logging system.
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRIGGER_FILE, "w", encoding="utf-8") as fh:
            json.dump(trigger_payload, fh, indent=2, default=str)

        logger.info("Trigger file written → %s", TRIGGER_FILE)

        # ---------------------------------------------------------------
        # Step 2c — Console output (visible in cron logs / CI pipelines)
        # ---------------------------------------------------------------
        print()
        print("⚠️  RETRAINING TRIGGERED")
        print(
            f"   drift_share         : {drift_share:.4f}  (threshold: {drift_threshold:.2f})"
        )
        print(f"   drifted_features    : {n_drifted}")
        print(f"   trigger_file        : {TRIGGER_FILE}")
        print(f"   html_report         : {drift_result.get('report_html_path')}")
        print()

    else:
        # ---------------------------------------------------------------
        # Step 3 — No drift, model is healthy
        # ---------------------------------------------------------------
        logger.info(
            "✅ Model healthy — drift_share=%.4f is within threshold=%.2f.",
            drift_share,
            drift_threshold,
        )
        print()
        print("✅ No drift detected. Model healthy.")
        print(f"   drift_share : {drift_share:.4f}  (threshold: {drift_threshold:.2f})")
        print(f"   html_report : {drift_result.get('report_html_path')}")
        print()

    # ------------------------------------------------------------------
    # Step 4 — Return trigger status
    # ------------------------------------------------------------------
    return triggered


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run manually from the project root:

      python -m monitor.retrain_trigger
      python monitor/retrain_trigger.py

    Exit codes
    ----------
    0 — model healthy, no retraining needed.
    1 — retraining triggered (drift detected) OR unexpected error.
    """
    triggered = check_and_trigger()
    # Non-zero exit code when triggered so CI/cron can branch on it:
    #   if python -m monitor.retrain_trigger; then
    #       echo "Model OK"
    #   else
    #       echo "Starting retraining pipeline..."
    #       python train/train.py
    #   fi
    sys.exit(1 if triggered else 0)
