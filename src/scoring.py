"""Customer risk scoring, configurable threshold engine, and batch prediction."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import DEFAULT_HIGH_RISK_THRESHOLD, DEFAULT_MED_RISK_THRESHOLD
from explainability import explain_customer_prediction
from preprocessing import prepare_inference_features
from validation import Schema


def assign_risk_band(
    proba: float | np.ndarray,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    med_threshold: float = DEFAULT_MED_RISK_THRESHOLD,
) -> str | np.ndarray:
    """Classify churn probability into High, Medium, or Low risk bands based on configurable thresholds."""
    if isinstance(proba, (float, int, np.floating)):
        if proba >= high_threshold:
            return "High"
        elif proba >= med_threshold:
            return "Medium"
        return "Low"

    # Array evaluation
    conditions = [proba >= high_threshold, proba >= med_threshold]
    choices = ["High", "Medium"]
    return np.select(conditions, choices, default="Low")


def assign_priority(risk_band: str, proba: float) -> str:
    """Determine operational outreach SLA based on risk severity."""
    if risk_band == "High":
        return "Immediate Outreach (within 24–48 hours)" if proba >= 0.75 else "High Priority (within 3–5 days)"
    elif risk_band == "Medium":
        return "Retention Review (within 14 days)"
    return "Routine Account Monitoring"


def recommend_retention_action(customer_record: dict[str, Any], top_driver: str = "") -> str:
    """Prescribe specific, evidence-backed retention actions tailored to account attributes."""
    record_str = " ".join([f"{k}:{v}" for k, v in customer_record.items()]).lower()
    driver_lower = top_driver.lower()

    if "contract" in driver_lower or "month-to-month" in record_str:
        return "Offer 12-month contract migration incentive (₹500/mo billing discount + price lock guarantee)."
    if "electronic check" in record_str or "payment" in driver_lower or "upi" in record_str or "bank" in driver_lower:
        return "Promote UPI AutoPay / NACH e-mandate registration with a one-time ₹250 bill cashback."
    if "tenure" in driver_lower:
        return "Enroll in 90-day Customer Onboarding Success Program with scheduled account health checks."
    if "fiber" in record_str and ("charge" in driver_lower or "monthly" in driver_lower):
        return "Initiate VIP service quality check and offer complimentary high-speed broadband booster."
    if any(s in record_str for s in ("no tech support", "techsupport: no", "techsupport:no")):
        return "Bundle complimentary Technical Support & Device Protection for 6 months."
    if any(s in record_str for s in ("no online security", "onlinesecurity: no")):
        return "Provide complimentary Cyber Security & Cloud Backup suite to enhance stickiness."

    return "Schedule dedicated account manager check-in to review usage and customer satisfaction."


def score_single_customer(
    pipeline: Pipeline,
    payload: dict[str, Any],
    schema: Schema,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    med_threshold: float = DEFAULT_MED_RISK_THRESHOLD,
) -> dict[str, Any]:
    """Score an individual customer record with risk tier, revenue at risk, attributions, and action."""
    input_df = prepare_inference_features(pd.DataFrame([payload]), schema)
    proba = float(pipeline.predict_proba(input_df)[0, 1])

    risk_band = assign_risk_band(proba, high_threshold=high_threshold, med_threshold=med_threshold)
    priority = assign_priority(risk_band, proba)

    # Detect spend column for revenue at risk
    monthly_col = next((c for c in schema.numeric if "month" in c.lower() or "charge" in c.lower() or "spend" in c.lower()), None)
    monthly_rev = 0.0
    if monthly_col and payload.get(monthly_col) is not None:
        try:
            val_clean = (
                str(payload[monthly_col])
                .replace("₹", "")
                .replace("Rs.", "")
                .replace("Rs", "")
                .replace("INR", "")
                .replace("$", "")
                .replace(",", "")
                .strip()
            )
            monthly_rev = float(val_clean)
        except Exception:
            monthly_rev = 0.0

    annual_rev = monthly_rev * 12.0

    # Local Explanation
    local_explanation = explain_customer_prediction(pipeline, payload, schema, churn_probability=proba)
    top_elevators = local_explanation.get("top_risk_elevators", [])
    top_driver_name = top_elevators[0]["label"] if top_elevators else "Overall Profile"

    recommended_action = recommend_retention_action(payload, top_driver=top_driver_name)

    return {
        "churn_probability": round(proba, 4),
        "churn_probability_pct": round(proba * 100, 1),
        "risk_band": risk_band,
        "predicted_status": "Likely to Churn" if proba >= 0.50 else "Likely to Stay",
        "priority": priority,
        "monthly_revenue_at_risk": round(monthly_rev, 2) if risk_band in ("High", "Medium") else 0.0,
        "annual_revenue_at_risk": round(annual_rev, 2) if risk_band in ("High", "Medium") else 0.0,
        "top_driver": top_driver_name,
        "recommended_action": recommended_action,
        "local_explanation": local_explanation,
        "thresholds_used": {
            "high": high_threshold,
            "medium": med_threshold,
        },
    }


def score_batch_dataframe(
    pipeline: Pipeline,
    df: pd.DataFrame,
    schema: Schema,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    med_threshold: float = DEFAULT_MED_RISK_THRESHOLD,
) -> pd.DataFrame:
    """Batch score customer dataframe (does NOT require Churn column)."""
    inference_df = prepare_inference_features(df, schema)
    probas = pipeline.predict_proba(inference_df)[:, 1]

    scored = df.drop(columns=["_y"], errors="ignore").copy()
    scored["churn_probability"] = np.round(probas, 4)
    scored["churn_risk_pct"] = np.round(probas * 100, 1)
    scored["risk_band"] = assign_risk_band(probas, high_threshold=high_threshold, med_threshold=med_threshold)
    scored["predicted_status"] = np.where(probas >= 0.50, "Likely to Churn", "Likely to Stay")

    # Priority
    priorities = []
    for p, b in zip(probas, scored["risk_band"]):
        priorities.append(assign_priority(b, p))
    scored["retention_priority"] = priorities

    # Revenue At Risk
    monthly_col = next((c for c in schema.numeric if "month" in c.lower() or "charge" in c.lower()), None)
    if monthly_col and monthly_col in scored.columns:
        monthly_vals = pd.to_numeric(
            scored[monthly_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)
        scored["monthly_revenue_at_risk"] = np.where(
            scored["risk_band"].isin(["High", "Medium"]),
            np.round(monthly_vals, 2),
            0.0,
        )
        scored["annual_revenue_at_risk"] = np.round(scored["monthly_revenue_at_risk"] * 12.0, 2)

    # Actions
    actions = []
    for _, row in scored.iterrows():
        actions.append(recommend_retention_action(row.to_dict()))
    scored["recommended_action"] = actions

    return scored


def load_batch_file_and_score(
    pipeline: Pipeline,
    raw_bytes: bytes,
    filename: str,
    schema: Schema,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    med_threshold: float = DEFAULT_MED_RISK_THRESHOLD,
) -> pd.DataFrame:
    """Parse batch upload file and generate scored dataframe."""
    buffer = io.BytesIO(raw_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        incoming = pd.read_excel(buffer)
    else:
        incoming = pd.read_csv(buffer)

    if incoming.empty:
        raise ValueError("The batch file is empty.")

    incoming.columns = [str(c).strip() for c in incoming.columns]

    # Validate that at least a critical subset of predictor columns exist
    missing_cols = [c for c in schema.all_features if c not in incoming.columns]
    if len(missing_cols) > len(schema.all_features) * 0.5:
        sample_missing = ", ".join(missing_cols[:6])
        raise ValueError(
            f"Batch file is missing critical feature columns expected by the trained model: {sample_missing}"
        )

    return score_batch_dataframe(
        pipeline,
        incoming,
        schema,
        high_threshold=high_threshold,
        med_threshold=med_threshold,
    )
