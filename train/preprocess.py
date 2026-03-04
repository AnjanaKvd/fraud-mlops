"""
Preprocessing script for the Kaggle Credit Card Fraud Detection dataset.

This script:
1. Loads the raw dataset
2. Drops the 'Time' column
3. Scales the 'Amount' feature
4. Splits the data into train / validation / test sets
5. Saves the processed splits to disk
6. Prints class distributions
7. Computes scale_pos_weight for XGBoost

Author: Anjana Kavidu
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost.

    XGBoost uses this value to handle class imbalance.
    Formula:
        scale_pos_weight = number_of_negative_samples / number_of_positive_samples

    Parameters
    ----------
    y : pd.Series
        Target variable (0 = normal, 1 = fraud)

    Returns
    -------
    float
        scale_pos_weight value
    """
    num_negative = (y == 0).sum()
    num_positive = (y == 1).sum()
    return num_negative / num_positive


def main():
    # -----------------------------
    # 1. Load the dataset
    # -----------------------------
    data_path = "data/raw/creditcard.csv"
    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}\n")

    # -----------------------------
    # 2. Print class distribution BEFORE splitting
    # -----------------------------
    print("Class distribution (before split):")
    print(df["Class"].value_counts())
    print(df["Class"].value_counts(normalize=True))
    print("-" * 50)

    # -----------------------------
    # 3. Drop the 'Time' column
    # -----------------------------
    df = df.drop(columns=["Time"])

    # -----------------------------
    # 4. Scale the 'Amount' column
    # -----------------------------
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # -----------------------------
    # 5. Split features and target
    # -----------------------------
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # -----------------------------
    # 6. Train / Validation / Test split
    # -----------------------------
    # Step 1: Split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Step 2: Split temp into validation (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # -----------------------------
    # 7. Print class distribution AFTER splitting
    # -----------------------------
    print("Class distribution after split:\n")

    print("Train set:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True))
    print()

    print("Validation set:")
    print(y_val.value_counts())
    print(y_val.value_counts(normalize=True))
    print()

    print("Test set:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True))
    print("-" * 50)

    # -----------------------------
    # 8. Save processed datasets
    # -----------------------------
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print("Processed datasets saved to data/processed/")
    print("-" * 50)

    # -----------------------------
    # 9. Calculate scale_pos_weight
    # -----------------------------
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Recommended scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")


if __name__ == "__main__":
    main()
