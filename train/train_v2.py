"""
Improved training script — Version 2 of the Fraud Detection model.

Improvements over train.py (v1):
  1. Feature engineering  — two new Amount-based features extracted from the
     *raw* Amount column before standard scaling, giving the model richer
     signal about transaction size:
       • amount_log    = log1p(Amount)          — compresses right-skewed dist.
       • amount_zscore = z-score of Amount      — explicit normalised deviation
  2. Hyperparameter update — larger learning_rate, deeper trees, fewer rounds
     (faster to train, often better generalisation on tabular fraud data):
       learning_rate=0.05, max_depth=6, n_estimators=200
  3. Registers the model as a NEW VERSION of 'FraudDetectionModel' in the
     same MLflow experiment so both versions are comparable side-by-side.
  4. Provides promote_best_model() — a self-contained helper you can call
     after evaluating both versions to move the winner to the Production stage.

Run AFTER train.py so that version 1 already exists in the registry.

Author: Anjana Kavidu
"""

import pathlib

import matplotlib  # noqa: E402 — must set backend before pyplot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "FraudDetectionModel"

RAW_DATA_PATH = "data/raw/creditcard.csv"  # original Kaggle CSV
ARTIFACTS_DIR = pathlib.Path("artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Amount-derived features to a copy of *df* that still contains the
    original (unscaled) 'Amount' column.

    New columns
    -----------
    amount_log    : log1p(Amount)
        Compresses the heavy right skew of transaction amounts.  Fraud often
        occurs at non-round, high amounts that cluster more clearly in log
        space than in linear space.

    amount_zscore : (Amount − mean) / std
        Explicit z-score, computed over the current split.  Combined with
        XGBoost's split-based learning this gives an extra axis the model
        can cut on independently of V1-V28 (which are already PCA-transformed).

    Note: we engineer on the RAW Amount BEFORE StandardScaler so the two
    features carry genuinely different information from each other and from
    the scaled Amount column that replaces the original afterwards.
    """
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"])
    mean_amt = df["Amount"].mean()
    std_amt = df["Amount"].std(ddof=0)
    df["amount_zscore"] = (df["Amount"] - mean_amt) / (std_amt + 1e-9)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (mirrors preprocess.py but done inline so we keep raw Amount)
# ─────────────────────────────────────────────────────────────────────────────


def load_and_prepare(raw_path: str):
    """
    Load the raw CSV, engineer new features on the original Amount column,
    then apply the same preprocessing as preprocess.py:
      - Drop 'Time'
      - StandardScale 'Amount'
      - 70 / 15 / 15 stratified split → train / val / test

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test  (all pd.DataFrame/Series)
    """
    df = pd.read_csv(raw_path)
    print(f"  → Raw data loaded   : {df.shape[0]:,} rows, {df.shape[1]} cols")

    # 1. Engineer features while Amount is still raw
    df = engineer_features(df)
    print(f"  → After engineering : {df.shape[1]} cols (+amount_log, +amount_zscore)")

    # 2. Drop Time (same as preprocess.py)
    df = df.drop(columns=["Time"])

    # 3. Scale Amount (same as preprocess.py)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # 4. Split features / target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 5. Stratified 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(
        f"  → Split sizes → train:{len(y_train):,}  "
        f"val:{len(y_val):,}  test:{len(y_test):,}"
    )
    print(f"  → Fraud in train  : {y_train.sum():,} ({y_train.mean() * 100:.3f}%)")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────


def save_feature_importance(
    model: XGBClassifier, feature_names, save_path: pathlib.Path, top_n: int = 15
):
    """Save a horizontal bar chart of the top-N most important features."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[sorted_idx]
    top_values = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features, top_values, color="#4C72B0")
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importances — v2", fontsize=14, fontweight="bold"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PROMOTION HELPER
# ─────────────────────────────────────────────────────────────────────────────


def promote_best_model(
    model_name: str = MODEL_NAME,
    metric: str = "average_precision",
    target_stage: str = "Production",
    tracking_uri: str = TRACKING_URI,
):
    """
    Compare ALL registered versions of *model_name* on *metric* and promote
    the best-performing version to *target_stage* in the MLflow Model Registry.

    How it works
    ────────────
    1. Lists every registered version via MlflowClient.
    2. For each version, fetches the parent MLflow run and reads the logged
       metric value.
    3. Archives the current Production model (if any).
    4. Transitions the winner to the target stage.

    Parameters
    ----------
    model_name   : str   Registry model name (default: FraudDetectionModel)
    metric       : str   MLflow metric key to rank by (higher = better)
    target_stage : str   Registry stage to promote into ('Production', etc.)
    tracking_uri : str   SQLite / HTTP tracking URI

    Usage
    -----
    After running both train.py and train_v2.py, call:
        python -c "from train.train_v2 import promote_best_model; promote_best_model()"

    Or append the call at the bottom of a CI/CD step:
        promote_best_model(metric="average_precision", target_stage="Production")
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for '{model_name}'.")

    print(f"\n{'─' * 60}")
    print(f"  Comparing {len(versions)} version(s) of '{model_name}'")
    print(f"  Ranking metric : {metric}")
    print(f"{'─' * 60}")

    best_version = None
    best_metric_val = -np.inf

    for v in versions:
        run = client.get_run(v.run_id)
        metric_val = run.data.metrics.get(metric)

        if metric_val is None:
            print(
                f"  v{v.version:>3}  run={v.run_id[:8]}…  "
                f"{metric} not logged — skipping"
            )
            continue

        flag = ""
        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_version = v
            flag = "  ◄ best so far"

        print(
            f"  v{v.version:>3}  run={v.run_id[:8]}…  "
            f"{metric}={metric_val:.4f}  stage={v.current_stage}{flag}"
        )

    if best_version is None:
        raise RuntimeError("Could not determine the best version.")

    print(f"\n  Winner → v{best_version.version}  ({metric}={best_metric_val:.4f})\n")

    # Archive any existing Production models for this name
    for v in versions:
        if v.current_stage == target_stage and v.version != best_version.version:
            print(f"  Archiving current {target_stage} → v{v.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived",
                archive_existing_versions=False,
            )

    # Promote the winner
    client.transition_model_version_stage(
        name=model_name,
        version=best_version.version,
        stage=target_stage,
        archive_existing_versions=True,  # safety net: archive anything else
    )
    print(
        f"  ✓ v{best_version.version} promoted to '{target_stage}' "
        f"in registry '{model_name}'\n"
    )
    return best_version


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION — TRAINING v2")
    print("  (new features + updated hyperparameters)")
    print("=" * 60)

    # ── 1. MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── 2. Load and prepare data ─────────────────────────────────────────────
    print("\n[1/4] Loading and engineering features …")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare(RAW_DATA_PATH)

    # ── 3. Hyperparameters — v2 changes highlighted ──────────────────────────
    params = {
        "max_depth": 6,  # ▲ was 5 — deeper trees, richer splits
        "learning_rate": 0.05,  # ▲ was 0.01 — faster convergence
        "n_estimators": 200,  # ▼ was 300 — fewer rounds (matches lr)
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # Class imbalance weight — computed fresh from this split
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        # Extra v2 info (for MLflow param logging)
        "new_features": "amount_log,amount_zscore",
    }

    # ── 4. Train inside an MLflow run ─────────────────────────────────────────
    print("\n[2/4] Training XGBClassifier v2 …")
    with mlflow.start_run(run_name="train-v2") as run:
        # Log all hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_version_tag", "v2")

        # Train
        # Pop the non-XGBoost key before passing to the classifier
        xgb_params = {k: v for k, v in params.items() if k != "new_features"}
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ── 5. Validation metrics ─────────────────────────────────────────────
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_proba)

        print("\n[3/4] Validation metrics:")
        print(f"  ROC-AUC          : {val_metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {val_metrics['average_precision']:.4f}")
        print(f"  F1 Score         : {val_metrics['f1_score']:.4f}")
        print(f"  Precision        : {val_metrics['precision']:.4f}")
        print(f"  Recall           : {val_metrics['recall']:.4f}")

        # Log metrics to MLflow
        for key, value in val_metrics.items():
            mlflow.log_metric(key, value)

        # ── 6. Feature importance plot ──────────────────────────────────────
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        importance_path = ARTIFACTS_DIR / "feature_importance_v2.png"
        save_feature_importance(model, X_train.columns, importance_path)
        mlflow.log_artifact(str(importance_path))

        # ── 7. Register model as new version ─────────────────────────────────
        print("\n[4/4] Registering model …")
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        # Add description to the new registry version so it appears in the UI
        client = MlflowClient()
        # Fetch the version that was just created (highest version number)
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        new_version = max(all_versions, key=lambda v: int(v.version))

        client.update_model_version(
            name=MODEL_NAME,
            version=new_version.version,
            description=(
                f"v2 — learning_rate=0.05, max_depth=6, n_estimators=200; "
                f"added amount_log & amount_zscore features. "
                f"Val AP={val_metrics['average_precision']:.4f} "
                f"ROC-AUC={val_metrics['roc_auc']:.4f}"
            ),
        )

        print(f"  ✓ Registered as '{MODEL_NAME}' version {new_version.version}")
        print(f"  ✓ MLflow run ID : {run.info.run_id}")


if __name__ == "__main__":
    main()
