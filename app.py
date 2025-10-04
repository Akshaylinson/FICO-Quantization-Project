"""
model.py

Prototype Probability-of-Default (PD) and Expected Loss model
for retail loans using borrower characteristics.

Setup:
    Place your dataset at ./data/Task 3 and 4_Loan_Data.csv

Run:
    python model.py
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

# ----------------------------
# Load Data
# ----------------------------
def load_loans_csv(path: str = "data/Task 3 and 4_Loan_Data.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {p.resolve()}")
    df = pd.read_csv(p)
    print(f"[load_loans_csv] Loaded {p} with shape {df.shape}")
    return df

# ----------------------------
# Train Models
# ----------------------------
def train_models(df: pd.DataFrame, target_col: str = "default",
                 test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    preprocessor = ColumnTransformer([("num", num_pipeline, numeric_cols)], remainder="drop")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    # Logistic Regression (with calibration)
    log_pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))])
    log_pipe.fit(X_train, y_train)
    log_calib = CalibratedClassifierCV(log_pipe, method="sigmoid", cv=3)
    log_calib.fit(X_train, y_train)

    # Random Forest
    rf_pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=200,
                                                                             random_state=random_state,
                                                                             n_jobs=-1))])
    rf_pipe.fit(X_train, y_train)

    # Evaluate
    log_probs = log_calib.predict_proba(X_test)[:, 1]
    rf_probs = rf_pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "logistic_test_roc_auc": float(roc_auc_score(y_test, log_probs)),
        "rf_test_roc_auc": float(roc_auc_score(y_test, rf_probs)),
        "logistic_test_pr_auc": float(average_precision_score(y_test, log_probs)),
        "rf_test_pr_auc": float(average_precision_score(y_test, rf_probs)),
    }

    chosen = "logistic" if metrics["logistic_test_roc_auc"] >= metrics["rf_test_roc_auc"] else "rf"

    print("[train_models] Chosen model:", chosen)
    print("[train_models] Metrics:", metrics)
    print("\nConfusion matrix (test, threshold=0.5):")
    chosen_probs = log_probs if chosen == "logistic" else rf_probs
    y_pred = (chosen_probs >= 0.5).astype(int)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return {
        "chosen": chosen,
        "log_calib": log_calib,
        "rf_pipe": rf_pipe,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
        "numeric_cols": numeric_cols
    }

# ----------------------------
# Prediction Function
# ----------------------------
def predict_pd_and_expected_loss(model_dict: Dict[str, Any],
                                 borrower_props: Dict[str, Any],
                                 ead_col: str = "loan_amt_outstanding",
                                 recovery_rate: float = 0.10) -> Dict[str, Any]:
    """
    borrower_props: dictionary with loan features, e.g.:
        {"income": 45000, "loan_amt_outstanding": 5000, ...}
    ead_col: which feature represents the exposure-at-default (EAD)
    recovery_rate: default=0.10 (10% recovery → 90% LGD)
    """
    numeric_cols = model_dict["numeric_cols"]
    row = pd.DataFrame([borrower_props])

    # Extract EAD
    if ead_col in row.columns:
        ead = float(row[ead_col].iloc[0])
    elif "ead" in row.columns:
        ead = float(row["ead"].iloc[0])
    else:
        ead = None

    for c in numeric_cols:
        if c not in row.columns:
            row[c] = np.nan
    row = row[numeric_cols]

    if model_dict["chosen"] == "logistic":
        pd_val = float(model_dict["log_calib"].predict_proba(row)[:, 1][0])
    else:
        pd_val = float(model_dict["rf_pipe"].predict_proba(row)[:, 1][0])

    lgd = 1.0 - recovery_rate
    expected_loss = None if ead is None else pd_val * ead * lgd

    return {"pd": pd_val, "ead": ead, "lgd": lgd,
            "recovery_rate": recovery_rate, "expected_loss": expected_loss}

# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    df = load_loans_csv()
    model_dict = train_models(df)

    # Plot ROC curves
    y_test = model_dict["y_test"]
    log_probs = model_dict["log_calib"].predict_proba(model_dict["X_test"])[:, 1]
    rf_probs = model_dict["rf_pipe"].predict_proba(model_dict["X_test"])[:, 1]

    fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr_log, tpr_log, label=f"Logistic AUC={roc_auc_score(y_test, log_probs):.3f}")
    plt.plot(fpr_rf, tpr_rf, label=f"RF AUC={roc_auc_score(y_test, rf_probs):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend()
    plt.show()

    # Example borrower
    example = {
        "income": 45000,
        "loan_amt_outstanding": 5000,
        "total_debt_outstanding": 10000,
        "credit_lines_outstanding": 1,
        "years_employed": 4,
        "fico_score": 620
    }
    result = predict_pd_and_expected_loss(model_dict, example)
    print("\nExample borrower result:")
    print(result)

