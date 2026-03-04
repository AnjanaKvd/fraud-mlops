"""
Evaluation script for the Kaggle Credit Card Fraud Detection project.

This script:
  1. Loads the production model from the MLflow Model Registry
  2. Evaluates it on the held-out test set (data/processed/test.csv)
  3. Computes ROC-AUC, Average Precision, F1, Precision, Recall, Confusion Matrix
  4. Saves ROC curve, Precision-Recall curve, and Confusion Matrix heatmap
  5. Logs all results as a new MLflow run linked to the registered model version

Author: Anjana Kavidu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY PRECISION-RECALL MATTERS MORE THAN ACCURACY FOR FRAUD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Kaggle Credit Card Fraud dataset is **severely imbalanced**:
  • ~284,315 legitimate transactions  (~99.83 %)
  • ~492 fraudulent transactions       (~0.17 %)

### The Accuracy Illusion
A model that labels EVERY transaction as "legitimate" achieves 99.83 % accuracy
while catching exactly ZERO frauds.  Accuracy rewards the majority class and
gives a dangerously false sense of performance in imbalanced settings.

### Why ROC-AUC Alone Can Be Misleading
ROC-AUC measures the trade-off between True Positive Rate (Recall) and False
Positive Rate across all thresholds.  Because the negative class is so large,
even a high FPR produces a very small absolute number of false alarms,
which inflates ROC-AUC even for mediocre fraud detectors.

### The Right Metric: Average Precision (AP) / PR-AUC
Precision-Recall curves focus entirely on the *positive* (fraud) class:
  • Precision = of all flagged transactions, how many are truly fraud?
  • Recall    = of all actual frauds, how many did the model catch?

A high AP score means the model is simultaneously precise (few false alarms
→ low investigation cost) AND recalls most frauds (low financial loss).
The random-baseline AP equals the fraud prevalence (~0.0017), so even a
score of 0.80 represents a ~470× improvement over chance.

### Business Interpretation
  • A false negative  (missed fraud)   → direct financial loss to the bank.
  • A false positive  (false alarm)    → investigation cost, customer friction.
Both costs are meaningful; the PR curve lets stakeholders choose the right
operating point by picking a threshold that balances them explicitly.

In short:  **Use Average Precision / PR-AUC as the primary headline metric,
ROC-AUC as a secondary sanity check, and F1/Precision/Recall at the chosen
operating threshold for business reporting.**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pathlib

import matplotlib

matplotlib.use("Agg")  # must be called before pyplot is imported (headless CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "FraudDetectionModel"
TEST_DATA_PATH = "data/processed/test.csv"
ARTIFACTS_DIR = pathlib.Path("artifacts/evaluation")
TARGET_COL = "Class"

# Classification threshold (0.5 default; adjust for stricter precision/recall)
THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_name: str, client: MlflowClient):
    """
    Load the latest registered version of the model from the MLflow Model
    Registry.  Tries stage 'None' (unassigned) first; falls back to the
    numerically latest version across all stages.

    Returns
    -------
    model       : mlflow.pyfunc.PyFuncModel
    version_obj : mlflow.entities.model_registry.ModelVersion
    """
    # Collect all versions and pick the latest by version number
    all_versions = client.search_model_versions(f"name='{model_name}'")
    if not all_versions:
        raise ValueError(
            f"No registered versions found for model '{model_name}'. "
            "Run train/train.py first."
        )

    # Sort descending by integer version number and take the newest
    all_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)
    version_obj = all_versions[0]

    model_uri = f"models:/{model_name}/{version_obj.version}"
    print(f"  → Loading model  : {model_uri}")
    print(f"     version        : {version_obj.version}")
    print(f"     current_stage  : {version_obj.current_stage}")
    print(f"     run_id         : {version_obj.run_id}")

    model = mlflow.pyfunc.load_model(model_uri)
    return model, version_obj


def load_test_data(path: str):
    """Read the test CSV and split into features / labels."""
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(
            f"Test data not found at '{path}'. "
            "Run train/preprocess.py first to generate processed splits."
        )
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(
        f"  → Loaded {len(df):,} test samples  "
        f"({y.sum():,} fraud  /  {(y == 0).sum():,} legitimate)"
    )
    return X, y


def compute_metrics(y_true, y_proba, threshold: float = THRESHOLD):
    """
    Compute all evaluation metrics at the given probability threshold.

    Returns a dict with scalar metrics and a 2-D confusion-matrix array.
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "threshold": threshold,
    }, cm


def print_summary_table(metrics: dict, version_obj):
    """Pretty-print a summary table of evaluation results."""
    separator = "─" * 52
    print()
    print("=" * 52)
    print("  FRAUD DETECTION MODEL — EVALUATION SUMMARY")
    print("=" * 52)
    print(f"  Model  : {MODEL_NAME}  v{version_obj.version}")
    print(f"  Stage  : {version_obj.current_stage}")
    print(f"  Run ID : {version_obj.run_id}")
    print(separator)
    print(f"  {'Metric':<28} {'Value':>10}")
    print(separator)
    scalar_keys = [
        ("ROC-AUC", "roc_auc"),
        ("Average Precision", "average_precision"),
        ("F1 Score", "f1_score"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Threshold", "threshold"),
    ]
    for label, key in scalar_keys:
        val = metrics[key]
        print(f"  {label:<28} {val:>10.4f}")
    print(separator)
    print(f"  {'True  Positives (TP)':<28} {metrics['true_positives']:>10,}")
    print(f"  {'False Positives (FP)':<28} {metrics['false_positives']:>10,}")
    print(f"  {'False Negatives (FN)':<28} {metrics['false_negatives']:>10,}")
    print(f"  {'True  Negatives (TN)':<28} {metrics['true_negatives']:>10,}")
    print("=" * 52)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def plot_roc_curve(y_true, y_proba, save_path: pathlib.Path, roc_auc: float):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC Curve  (AUC = {roc_auc:.4f})")
    ax.plot(
        [0, 1], [0, 1], color="#AAAAAA", lw=1, linestyle="--", label="Random Baseline"
    )
    ax.fill_between(fpr, tpr, alpha=0.10, color="#4C72B0")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Fraud Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved ROC curve  : {save_path}")


def plot_precision_recall_curve(
    y_true, y_proba, save_path: pathlib.Path, avg_precision: float
):
    """
    Plot and save the Precision-Recall curve.

    The dotted horizontal line marks the random-classifier baseline, which
    equals the positive-class prevalence in the test set.  A good fraud
    detector should sit far above this line.
    """
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()  # fraud prevalence in test set

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        rec_vals,
        prec_vals,
        color="#DD8452",
        lw=2,
        label=f"PR Curve  (AP = {avg_precision:.4f})",
    )
    ax.axhline(
        y=baseline,
        color="#AAAAAA",
        lw=1,
        linestyle="--",
        label=f"Random Baseline ({baseline:.4f})",
    )
    ax.fill_between(rec_vals, prec_vals, alpha=0.10, color="#DD8452")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curve — Fraud Detection", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved PR curve   : {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: pathlib.Path):
    """Plot and save a labelled Confusion Matrix heatmap."""
    labels = ["Legitimate (0)", "Fraud (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Fraud Detection", fontsize=14, fontweight="bold")

    # Annotate cells with TP / FP / FN / TN labels for readability
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.75,
                cell_labels[i][j],
                ha="center",
                va="center",
                fontsize=10,
                color="grey",
                style="italic",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved conf matrix: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 52)
    print("  FRAUD DETECTION — EVALUATION PIPELINE")
    print("=" * 52)

    # ── 1. MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # ── 2. Load model from registry ──────────────────────────────────────────
    print("\n[1/4] Loading model from MLflow Registry …")
    model, version_obj = load_model(MODEL_NAME, client)

    # ── 3. Load test data ─────────────────────────────────────────────────────
    print("\n[2/4] Loading test data …")
    X_test, y_test = load_test_data(TEST_DATA_PATH)

    # ── 4. Predict ───────────────────────────────────────────────────────────
    print("\n[3/4] Running inference …")
    # mlflow.pyfunc.PyFuncModel.predict() returns a DataFrame or ndarray.
    # For XGBoost logged via mlflow.xgboost, it returns predicted probabilities
    # for the positive class (column index 1) when the model was saved with
    # a binary classification objective.
    raw_preds = model.predict(X_test)

    # Normalise to a 1-D probability array
    if isinstance(raw_preds, pd.DataFrame):
        # Some flavours return a DataFrame with columns [0, 1]
        if raw_preds.shape[1] == 2:
            y_proba = raw_preds.iloc[:, 1].values
        else:
            y_proba = raw_preds.iloc[:, 0].values
    else:
        y_proba = np.array(raw_preds).ravel()

    # ── 5. Compute metrics ───────────────────────────────────────────────────
    metrics, cm = compute_metrics(y_test, y_proba, threshold=THRESHOLD)

    # ── 6. Save plots ────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    roc_path = ARTIFACTS_DIR / "roc_curve.png"
    pr_path = ARTIFACTS_DIR / "precision_recall_curve.png"
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"

    plot_roc_curve(y_test, y_proba, roc_path, metrics["roc_auc"])
    plot_precision_recall_curve(y_test, y_proba, pr_path, metrics["average_precision"])
    plot_confusion_matrix(cm, cm_path)

    # ── 7. Print summary table ───────────────────────────────────────────────
    print_summary_table(metrics, version_obj)

    # ── 8. Log evaluation run to MLflow ──────────────────────────────────────
    print("[4/4] Logging evaluation run to MLflow …")
    tags = {
        "evaluation_type": "test_set",
        "model_name": MODEL_NAME,
        "model_version": version_obj.version,
        "model_stage": version_obj.current_stage,
        "source_run_id": version_obj.run_id,
        "mlflow.runName": f"eval-{MODEL_NAME}-v{version_obj.version}",
    }

    with mlflow.start_run(tags=tags) as eval_run:
        # Log scalar metrics
        scalar_metric_keys = [
            "roc_auc",
            "average_precision",
            "f1_score",
            "precision",
            "recall",
            "threshold",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        ]
        for key in scalar_metric_keys:
            mlflow.log_metric(key, metrics[key])

        # Log dataset info as params
        mlflow.log_param("test_data_path", TEST_DATA_PATH)
        mlflow.log_param("test_n_samples", len(y_test))
        mlflow.log_param("test_n_fraud", int(y_test.sum()))
        mlflow.log_param("model_version", version_obj.version)
        mlflow.log_param("threshold", THRESHOLD)

        # Log plots as artifacts
        mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(pr_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

        eval_run_id = eval_run.info.run_id

    print(f"  → MLflow evaluation run ID : {eval_run_id}")
    print(f"  → Plots saved to           : {ARTIFACTS_DIR.resolve()}")
    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
