# Exploratory Data Analysis (EDA) — Credit Card Fraud Detection

## Overview

This document summarizes the key findings from the Exploratory Data Analysis (EDA) performed on the **Credit Card Fraud Detection** dataset obtained from **Kaggle**.

The full EDA process, including code and visualizations, is documented in the accompanying Jupyter notebook:

> **`credit-card-fraud-detection-eda.ipynb`**

---

## Dataset Summary

* **Total transactions:** 284,807

* **Total features:** 31

* **Features:**

  * `V1`–`V28`: PCA-transformed numerical features
  * `Time`: Seconds elapsed since the first transaction
  * `Amount`: Transaction amount
  * `Class`: Target variable

    * `0` → Normal
    * `1` → Fraud

* **Fraud rate:** ~0.172%

* **Missing values:** None

The dataset is **highly imbalanced**, which has important implications for modeling and evaluation.

---

## Class Distribution

* Normal transactions dominate the dataset
* Fraudulent transactions represent **less than 0.2%** of all records

**Key implication:**
Accuracy alone is not a reliable metric. Precision, recall, F1-score, and ROC-AUC are more appropriate for model evaluation.

---

## Transaction Amount Analysis

* Fraudulent transactions tend to have **lower amounts** compared to normal transactions
* Amount distribution is **right-skewed**
* Significant overlap exists, but density differences are visible

**Insight:**
`Amount` is a useful feature but must be combined with others to detect fraud effectively.

---

## Correlation Analysis

* Most PCA features are weakly correlated with each other (expected due to PCA)
* Certain features show **strong correlation with the `Class` variable**
* Both positive and negative correlations are observed

**Insight:**
Some PCA components carry strong fraud-discriminative signals despite anonymization.

---

## Time-Based Fraud Patterns

* The `Time` feature was transformed into **hour-of-day**
* Fraudulent transactions show **non-uniform distribution across hours**
* Certain hours exhibit relatively higher fraud activity compared to normal transactions

**Insight:**
Temporal features can improve model performance and should be retained.

---

## Top Predictive Features

The top 5 features were selected based on **absolute correlation with the target variable (`Class`)**.

* These features show **clear distribution differences** between fraud and normal cases
* Fraud distributions often appear more concentrated or shifted

**Insight:**
These features are strong candidates for:

* Baseline models (e.g., Logistic Regression)
* Feature importance analysis
* Explainability techniques (SHAP, permutation importance)

---

## Key Takeaways

* The dataset is **severely imbalanced**
* Fraud patterns differ notably in:

  * Transaction amount
  * Time of occurrence
  * Several PCA-transformed features
* Feature distributions suggest good separability for supervised models
