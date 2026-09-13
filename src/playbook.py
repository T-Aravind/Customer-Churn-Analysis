"""Retention playbook generation based on empirical data drivers and risk segments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from validation import Schema


def generate_retention_playbook(
    df: pd.DataFrame,
    schema: Schema,
    metrics: dict[str, Any] | None = None,
    global_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate dynamic, evidence-grounded retention strategies derived from actual portfolio data."""
    y = df["_y"] if "_y" in df.columns else df[schema.target]
    portfolio_churn = float(y.mean())
    total_customers = len(df)

    monthly_col = next((c for c in schema.numeric if "month" in c.lower() or "charge" in c.lower()), None)
    tenure_col = next((c for c in schema.numeric if "tenure" in c.lower()), None)

    pillars: list[dict[str, Any]] = []

    # 1. Contract & High Lift Cohorts
    contract_col = next((c for c in schema.categorical if "contract" in c.lower()), None)
    if contract_col:
        contract_stats = (
            df.assign(_y=y)
            .groupby(contract_col, observed=False)
            .agg(customers=("_y", "size"), churners=("_y", "sum"), rate=("_y", "mean"))
            .reset_index()
        )
        highest_contract = contract_stats.sort_values("rate", ascending=False).iloc[0]
        c_name = str(highest_contract[contract_col])
        c_rate = float(highest_contract["rate"])
        c_count = int(highest_contract["customers"])
        c_lift = c_rate - portfolio_churn

        c_rev_risk = 0.0
        if monthly_col:
            c_rev_risk = float(df[(df[contract_col] == c_name) & (y == 1)][monthly_col].fillna(0).sum() * 12)

        pillars.append({
            "priority": "Immediate (0–30 Days)",
            "badge_color": "danger",
            "pillar_name": "Contract Stabilization & Term Migration",
            "target_segment": f"{contract_col}: {c_name}",
            "observed_metric": f"{c_rate:.1%} churn rate ({c_lift:+.1%} pts vs portfolio baseline)",
            "affected_accounts": f"{c_count:,} accounts",
            "annual_revenue_at_risk": f"₹{c_rev_risk:,.2f}" if c_rev_risk > 0 else "Calculated per account",
            "strategic_rationale": f"Accounts on {c_name} exhibit the highest volatility in the portfolio. Converting even 15% of this cohort to annual agreements delivers compounding retention gains.",
            "recommended_actions": [
                "Launch a 12-month contract migration incentive: offer a ₹500/month billing discount + price lock guarantee.",
                "Trigger automated retention save workflows 45 days prior to customer renewal dates.",
                "Route inbound cancellation requests from this segment to a specialized Tier-2 Save Desk.",
            ],
        })

    # 2. Early Lifecycle & Onboarding (Tenure)
    if tenure_col:
        new_cust = df[df[tenure_col] <= 6]
        if not new_cust.empty:
            new_churn_rate = float(y.loc[new_cust.index].mean())
            new_count = len(new_cust)
            tenure_lift = new_churn_rate - portfolio_churn

            tenure_rev_risk = 0.0
            if monthly_col:
                tenure_rev_risk = float(new_cust[y.loc[new_cust.index] == 1][monthly_col].fillna(0).sum() * 12)

            pillars.append({
                "priority": "Short-Term (30–60 Days)",
                "badge_color": "warning",
                "pillar_name": "New Account 90-Day Success Program",
                "target_segment": f"Accounts with tenure ≤ 6 months",
                "observed_metric": f"{new_churn_rate:.1%} early churn rate ({tenure_lift:+.1%} pts vs portfolio)",
                "affected_accounts": f"{new_count:,} onboarding accounts",
                "annual_revenue_at_risk": f"₹{tenure_rev_risk:,.2f}" if tenure_rev_risk > 0 else "Calculated per account",
                "strategic_rationale": "Attrition peaks in the first 6 months. Proactive onboarding reduces early buyer remorse and accelerates time-to-value.",
                "recommended_actions": [
                    "Implement a 30/60/90 day milestone check-in sequence via WhatsApp/SMS and account manager outreach.",
                    "Provide free onboarding consultation and guided feature walkthroughs for new signups.",
                    "Track early usage signals to flag disengaged accounts before the first quarterly billing cycle.",
                ],
            })

    # 3. Payment Method Optimization
    payment_col = next((c for c in schema.categorical if "payment" in c.lower() or "billing" in c.lower()), None)
    if payment_col:
        pay_stats = (
            df.assign(_y=y)
            .groupby(payment_col, observed=False)
            .agg(customers=("_y", "size"), churners=("_y", "sum"), rate=("_y", "mean"))
            .reset_index()
        )
        worst_pay = pay_stats.sort_values("rate", ascending=False).iloc[0]
        p_name = str(worst_pay[payment_col])
        p_rate = float(worst_pay["rate"])
        p_count = int(worst_pay["customers"])
        p_lift = p_rate - portfolio_churn

        pillars.append({
            "priority": "Medium-Term (60–90 Days)",
            "badge_color": "info",
            "pillar_name": "Frictionless Billing & Auto-Pay Promotion",
            "target_segment": f"{payment_col}: {p_name}",
            "observed_metric": f"{p_rate:.1%} churn rate ({p_lift:+.1%} pts vs portfolio baseline)",
            "affected_accounts": f"{p_count:,} accounts",
            "annual_revenue_at_risk": "High operational friction cost",
            "strategic_rationale": f"Manual and non-automated payment methods like {p_name} experience higher involuntary and convenience-based attrition.",
            "recommended_actions": [
                "Offer a one-time ₹250 cashback / bill discount for enrolling in UPI AutoPay or NACH bank mandate.",
                "Implement automated WhatsApp, SMS, and email reminders 5 days prior to invoice due dates.",
                "Simplify digital payment options (UPI, Net Banking, QR code) to eliminate transaction friction.",
            ],
        })

    # 4. Service Support & Value-Add Add-ons
    support_col = next((c for c in schema.categorical if any(k in c.lower() for k in ("tech", "support", "security", "backup", "protection"))), None)
    if support_col:
        no_support = df[df[support_col].astype(str).str.lower().str.contains("no")]
        if not no_support.empty:
            no_sup_rate = float(y.loc[no_support.index].mean())
            no_sup_count = len(no_support)
            sup_lift = no_sup_rate - portfolio_churn

            pillars.append({
                "priority": "Ongoing Program",
                "badge_color": "success",
                "pillar_name": "Value-Add Protection & Support Bundling",
                "target_segment": f"Accounts without {support_col}",
                "observed_metric": f"{no_sup_rate:.1%} churn rate ({sup_lift:+.1%} pts vs portfolio baseline)",
                "affected_accounts": f"{no_sup_count:,} unprotected accounts",
                "annual_revenue_at_risk": "Moderate stickiness deficit",
                "strategic_rationale": f"Subscribers lacking {support_col} have lower switching barriers and significantly higher churn propensity.",
                "recommended_actions": [
                    f"Create bundled tiers offering 6 months of complimentary {support_col} on contract renewals.",
                    "Highlight protection value in monthly customer account newsletters and app notifications.",
                    "Incentivize customer service reps to recommend support add-ons during routine support interactions.",
                ],
            })

    # Operational cadence
    cadence = [
        {
            "frequency": "Weekly",
            "title": "Score & Triage High-Risk Accounts",
            "action": "Upload latest customer extract, score the book, and assign all accounts with Churn Probability ≥ 60% to retention reps.",
        },
        {
            "frequency": "Bi-Weekly",
            "title": "Campaign Performance Review",
            "action": "Track save rates across contract conversion offers and auto-pay enrollment campaigns.",
        },
        {
            "frequency": "Monthly",
            "title": "Executive Portfolio Health Check",
            "action": "Review portfolio churn rate trends, realized revenue savings, and recalibrate risk thresholds in RetainIQ.",
        },
        {
            "frequency": "Quarterly",
            "title": "Model Retraining & Governance",
            "action": "Retrain models with the newest quarter of historical outcomes to adapt to changing customer behavioral patterns.",
        },
    ]

    return {
        "portfolio_summary": {
            "total_customers": total_customers,
            "portfolio_churn_rate": round(portfolio_churn, 4),
            "pillars_count": len(pillars),
        },
        "strategic_pillars": pillars,
        "operational_cadence": cadence,
    }
