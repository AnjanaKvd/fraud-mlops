"""
Training script for the Kaggle Credit Card Fraud Detection project.

This script:
1. Loads preprocessed train and validation data
2. Trains an XGBoost classifier with class imbalance handling
3. Logs parameters, metrics, artifacts, and model to MLflow
4. Registers the trained model in the MLflow Model Registry

Author: Anjana Kavidu
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def main():
    # -----------------------------------------------------
    # 1. MLflow setup
    # -----------------------------------------------------
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("fraud-detection")

    # -----------------------------------------------------
    # 2. Load processed datasets
    # -----------------------------------------------------
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]

    # -----------------------------------------------------
    # 3. Define model hyperparameters
    # -----------------------------------------------------
    params = {
        "max_depth": 5,
        "learning_rate": 0.01,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # Handle extreme class imbalance
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    # -----------------------------------------------------
    # 4. Start MLflow run
    # -----------------------------------------------------
    with mlflow.start_run():
        # Log hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # -------------------------------------------------
        # 5. Train the XGBoost model
        # -------------------------------------------------
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # -------------------------------------------------
        # 6. Validation predictions
        # -------------------------------------------------
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        # -------------------------------------------------
        # 7. Compute evaluation metrics
        # -------------------------------------------------
        roc_auc = roc_auc_score(y_val, y_val_proba)
        avg_precision = average_precision_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)

        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()

        # -------------------------------------------------
        # 8. Log metrics to MLflow
        # -------------------------------------------------
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", avg_precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.log_metric("true_positives", tp)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_negatives", fn)

        # -------------------------------------------------
        # 9. Feature importance plot
        # -------------------------------------------------
        importances = model.feature_importances_
        features = X_train.columns

        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importances)[::-1][:15]
        plt.barh(features[sorted_idx], importances[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Top 15 Feature Importances")
        plt.gca().invert_yaxis()

        os.makedirs("artifacts", exist_ok=True)
        feature_plot_path = "artifacts/feature_importance.png"
        plt.tight_layout()
        plt.savefig(feature_plot_path)
        plt.close()

        mlflow.log_artifact(feature_plot_path)

        # -------------------------------------------------
        # 10. Log and register model
        # -------------------------------------------------
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="FraudDetectionModel",
        )

        # -------------------------------------------------
        # 11. Print summary
        # -------------------------------------------------
        print("Training completed successfully.")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("Model registered as 'FraudDetectionModel'")


if __name__ == "__main__":
    main()
