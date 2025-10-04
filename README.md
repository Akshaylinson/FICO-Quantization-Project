🧮 FICO Quantization Project

Objective:
This project demonstrates how to quantize FICO credit scores into discrete risk categories for credit risk modeling.
It uses two approaches — Mean Squared Error (MSE) minimization and Log-Likelihood (LL) maximization — to create optimal FICO score buckets for predicting the Probability of Default (PD) in mortgage or loan portfolios.

📁 Project Structure
TASK 4/
│
├── fico_quantization.py              # Main Python script
├── Task 3 and 4_Loan_Data (1).csv    # Input dataset (loan data with FICO scores)
└── requirements.txt                  # Python dependencies

📦 Requirements

Create a Python virtual environment and install the required packages:

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

requirements.txt
numpy
pandas
scikit-learn
matplotlib
jupyter
pytest

▶️ How to Run the Project

Run the main script in VS Code terminal or command prompt:

python fico_quantization.py

📊 Output Overview

When you run the script, two sets of bucket analyses are generated:

1️⃣ MSE Bucketing Result

Buckets are created by minimizing variance (Mean Squared Error).
This groups FICO scores into clusters where each bucket has similar scores.

Example:

{
    "mse": 300.52,
    "buckets": {
        "0": {"fico_range": [563, 617], "mean_score": 593.24, "count": 2569},
        "1": {"fico_range": [618, 664], "mean_score": 640.88, "count": 3036}
    }
}

2️⃣ Log-Likelihood Bucketing Result

Buckets are formed by maximizing the likelihood of observing default patterns in each range.

Example:

{
    "log_likelihood": -4321.01,
    "buckets": [
        {"bucket": "(407.9, 587.0]", "count": 2050, "defaults": 817, "pd": 0.3985},
        {"bucket": "(587.0, 623.0]", "count": 1971, "defaults": 425, "pd": 0.2156}
    ]
}

🧠 How It Works

Dataset:
The script expects a CSV with at least two columns:

fico → Borrower FICO score (range 300–850)

default → Binary indicator (1 = defaulted, 0 = repaid)

MSE Bucketing:
Uses k-means clustering to minimize within-bucket variance of FICO scores.

Log-Likelihood Bucketing:
Uses quantile-based binning to split the FICO range, computing each bucket’s:

number of customers

number of defaults

probability of default (PD = defaults / total)

Output:
Displays summary statistics for each bucket in a formatted JSON-like structure.

💡 Insights

Lower FICO buckets correspond to higher default probabilities, confirming that FICO is a strong indicator of credit risk.

The quantized FICO buckets can be used as categorical features in future machine learning models (for example, logistic regression or gradient boosting).

⚙️ Optional Enhancements

Add visualizations (matplotlib) for:

FICO score distribution

Probability of default (PD) per bucket
