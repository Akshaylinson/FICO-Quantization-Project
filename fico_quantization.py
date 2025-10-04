"""
fico_quantization.py
Quantization of FICO scores into buckets using:
1. MSE (via k-means clustering on FICO scores)
2. Log-Likelihood (based on default distribution)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


# -------------------------------
# MSE-based Bucketing (k-means)
# -------------------------------
def mse_bucketing(fico_scores: np.ndarray, num_buckets: int):
    scores = fico_scores.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_buckets, random_state=42).fit(scores)
    labels = kmeans.predict(scores)

    buckets = {}
    for i in range(num_buckets):
        idx = np.where(labels == i)[0]
        buckets[i] = {
            "fico_range": (int(scores[idx].min()), int(scores[idx].max())),
            "mean_score": float(scores[idx].mean()),
            "count": len(idx)
        }

    mse = mean_squared_error(scores, kmeans.cluster_centers_[labels])
    return {"mse": mse, "buckets": buckets}


# -------------------------------
# Log-Likelihood-based Bucketing
# -------------------------------
def log_likelihood_bucketing(df: pd.DataFrame, num_buckets: int):
    """
    Splits fico_score into equal-frequency buckets, 
    then computes log-likelihood of defaults.
    """
    df = df.copy()
    df["bucket"] = pd.qcut(df["fico_score"], q=num_buckets, duplicates="drop")

    ll = 0
    bucket_stats = []
    for b, sub in df.groupby("bucket"):
        ni = len(sub)                  # number of records
        ki = sub["default"].sum()      # number of defaults
        pi = ki / ni if ni > 0 else 1e-6
        ll += ki * np.log(pi + 1e-6) + (ni - ki) * np.log(1 - pi + 1e-6)

        bucket_stats.append({
            "bucket": str(b),
            "count": ni,
            "defaults": int(ki),
            "pd": round(pi, 4)
        })

    return {"log_likelihood": ll, "buckets": bucket_stats}


# -------------------------------
# Main Runner
# -------------------------------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Task 3 and 4_Loan_Data (1).csv")

    if "fico_score" not in df.columns or "default" not in df.columns:
        raise ValueError("Dataset must contain 'fico_score' and 'default' columns")

    fico_scores = df["fico_score"].values

    # --- Run MSE bucketing ---
    mse_result = mse_bucketing(fico_scores, num_buckets=5)
    print("\nMSE Bucketing Result:")
    print(mse_result)

    # --- Run Log-Likelihood bucketing ---
    ll_result = log_likelihood_bucketing(df, num_buckets=5)
    print("\nLog-Likelihood Bucketing Result:")
    print(ll_result)

