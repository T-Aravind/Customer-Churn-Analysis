"""Standalone prediction script for RetainIQ."""

import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import pandas as pd

from config import MODELS_DIR, SAMPLE_PATH
from scoring import score_single_customer
from validation import load_file_bytes

MODEL_PATH = MODELS_DIR / "churn_model.joblib"


def load_model() -> Any:
    """Load the trained machine learning pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_churn(customer: dict[str, Any]) -> dict[str, Any]:
    """Score a single customer using persisted model."""
    model = load_model()
    with open(SAMPLE_PATH, "rb") as f:
        raw = f.read()
    _, schema, _ = load_file_bytes(raw, "Telco-Customer-Churn.csv", is_training=True)

    result = score_single_customer(model, customer, schema)
    # Streamlit and legacy key compatibility
    result["churn_probability"] = result["churn_probability_pct"] / 100.0
    result["will_churn"] = bool(result["churn_probability"] >= 0.50)
    result["prediction"] = "Customer is likely to CHURN" if result["will_churn"] else "Customer is likely to STAY (Retained)"
    return result


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
    }

    result = predict_churn(sample_customer)
    print("\n--- Single Customer Risk Evaluation ---")
    print(f"Churn Probability: {result['churn_probability_pct']}%")
    print(f"Risk Tier: {result['risk_band']}")
    print(f"Priority SLA: {result['priority']}")
    print(f"Top Churn Driver: {result['top_driver']}")
    print(f"Recommended Action: {result['recommended_action']}")
