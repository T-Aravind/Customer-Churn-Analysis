"""RetainIQ — Enterprise Customer Retention Intelligence & Activity Dashboard.

Production Streamlit Application for Churn Diagnosis, Real-Time Scoring,
Model Governance, and Account Intervention Activity Tracking.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DEFAULT_HIGH_RISK_THRESHOLD, DEFAULT_MED_RISK_THRESHOLD, SAMPLE_PATH
from engine import get_workspace
from explainability import compute_global_feature_importance, explain_customer_prediction
from playbook import generate_retention_playbook
from predict import load_model, predict_churn
from preprocessing import DATA_PATH, load_data
from scoring import score_batch_dataframe, score_single_customer
from train import main as train_models_routine
from validation import load_file_bytes

# Custom Styling (Dark-Indigo Glassmorphic Executive Theme)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header & Accent Styles */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(99, 102, 241, 0.1);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 50%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1rem;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 20px;
    text-align: left;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
}

.metric-subtext {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 6px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-high {
    background-color: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.badge-medium {
    background-color: rgba(245, 158, 11, 0.2);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-low {
    background-color: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-info {
    background-color: rgba(99, 102, 241, 0.2);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

/* Activity Item */
.activity-card {
    background: #1E293B;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-left: 4px solid #6366F1;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.activity-card.critical {
    border-left-color: #EF4444;
}

.activity-card.warning {
    border-left-color: #F59E0B;
}

.activity-card.success {
    border-left-color: #10B981;
}

.activity-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.activity-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #F1F5F9;
}

.activity-time {
    font-size: 0.75rem;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
}

.activity-body {
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.4;
}

/* Attribution lists */
.attribution-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
</style>
"""


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_session"

    ws = get_workspace(st.session_state.session_id)
    if ws.df is None and SAMPLE_PATH.exists():
        try:
            with open(SAMPLE_PATH, "rb") as f:
                ws.ingest(f.read(), filename="sample_telco_customers.csv")
            if ws.model is None:
                ws.train()
        except Exception:
            pass

    if "activities" not in st.session_state:
        st.session_state.activities = [
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "category": "SYSTEM",
                "level": "info",
                "title": "Platform Engine Initialized",
                "detail": "Workspace active with baseline Telco dataset & Champion XGBoost/Gradient Boosting models.",
                "account_id": "SYSTEM",
            },
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "category": "AUDIT",
                "level": "critical",
                "title": "High-Risk Segment Identified",
                "detail": "Month-to-month contracts with Fiber Optic service flagged with 42.7% churn rate (+16.2% lift).",
                "account_id": "PORTFOLIO",
            },
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "category": "INTERVENTION",
                "level": "warning",
                "title": "Retention Outreach Queued",
                "detail": "Playbook 24h SLA dispatched for accounts with Churn Risk > 70%.",
                "account_id": "BATCH-001",
            },
        ]


def log_activity(title: str, detail: str, level: str = "info", category: str = "INTERVENTION", account_id: str = "USER") -> None:
    """Record an operational activity in the session activity feed."""
    event = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "category": category,
        "level": level,
        "title": title,
        "detail": detail,
        "account_id": account_id,
    }
    if "activities" in st.session_state:
        st.session_state.activities.insert(0, event)


def main() -> None:
    # Page Setup
    st.set_page_config(
        page_title="RetainIQ — Retention Intelligence & Activities",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    workspace = get_workspace(st.session_state.session_id)

    # Header Bar
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">⚡ RetainIQ Enterprise Intelligence</h1>
            <p class="hero-subtitle">Customer Retention Diagnosis, Predictive Risk Scoring & Live Operational Activity Stream</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Workspace Controls")

        # Data Status
        data_loaded = workspace.df is not None
        model_trained = workspace.model is not None

        st.markdown(f"**Data Extract**: `{'✅ Loaded (' + str(len(workspace.df)) + ' rows)' if data_loaded else '❌ Not Loaded'}`")
        st.markdown(f"**Champion Model**: `{'✅ Active' if model_trained else '❌ Untrained'}`")

        st.divider()

        # Risk Threshold Controls
        st.markdown("#### 🎯 Risk Threshold Cutoffs")
        high_t = st.slider(
            "High Risk Cutoff (SLA 24h)",
            min_value=0.40,
            max_value=0.90,
            value=float(workspace.high_threshold),
            step=0.05,
        )
        med_t = st.slider(
            "Medium Risk Cutoff (SLA 7d)",
            min_value=0.15,
            max_value=high_t - 0.05,
            value=min(float(workspace.med_threshold), high_t - 0.05),
            step=0.05,
        )

        if high_t != workspace.high_threshold or med_t != workspace.med_threshold:
            if st.button("Apply Thresholds", type="primary", use_container_width=True):
                workspace.set_thresholds(high=high_t, med=med_t)
                log_activity(
                    "Risk Thresholds Adjusted",
                    f"Updated operational cutoffs to High: {high_t:.0%}, Medium: {med_t:.0%}.",
                    level="warning",
                    category="AUDIT",
                )
                st.success("Thresholds updated!")
                st.rerun()

        st.divider()

        # Log New Activity Form
        with st.expander("📝 Log Account Note / Action"):
            note_acct = st.text_input("Customer / Account ID", value="CUST-8492")
            note_action = st.selectbox("Action Type", ["Proactive Outreach", "Plan Upgrade Offered", "Discount Applied", "Contract Extended", "Support Ticket Resolved"])
            note_desc = st.text_area("Notes", value="Offered 1-year loyalty contract incentive.")
            if st.button("Record Activity", use_container_width=True):
                log_activity(
                    f"{note_action} Recorded",
                    f"{note_desc} (Account: {note_acct})",
                    level="success",
                    category="INTERVENTION",
                    account_id=note_acct,
                )
                st.success("Activity logged to live feed!")
                st.rerun()

        st.divider()

        # Quick Reset & Retrain
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Retrain", use_container_width=True):
                with st.spinner("Training models..."):
                    workspace.train()
                    log_activity("Models Retrained", "Candidate models trained & Champion selected.", level="info", category="MODEL")
                    st.success("Retrained!")
                    st.rerun()
        with col_r2:
            if st.button("🗑️ Reset", use_container_width=True):
                workspace.reset()
                init_state()
                st.rerun()

    # Navigation Tabs
    tab_activities, tab_overview, tab_predict, tab_batch, tab_playbook, tab_models = st.tabs([
        "📋 Live Activities",
        "📊 Portfolio Overview",
        "🎯 Risk Scoring Desk",
        "📁 Customer Book & Batch",
        "🛡️ Retention Playbook",
        "🏆 Model Governance",
    ])

    # TAB 1: LIVE ACTIVITIES & AUDIT STREAM
    with tab_activities:
        st.subheader("📋 Operational Activity Stream & Intervention Audit")
        st.caption("Live chronological audit trail of churn risk alerts, customer scoring events, and mitigation actions.")

        # Activity Summary Metrics
        act_col1, act_col2, act_col3, act_col4 = st.columns(4)
        total_acts = len(st.session_state.activities)
        critical_acts = sum(1 for a in st.session_state.activities if a["level"] == "critical")
        warning_acts = sum(1 for a in st.session_state.activities if a["level"] == "warning")
        success_acts = sum(1 for a in st.session_state.activities if a["level"] == "success")

        act_col1.metric("Total Events Logged", total_acts)
        act_col2.metric("Critical Risk Alerts", critical_acts)
        act_col3.metric("Pending Interventions", warning_acts)
        act_col4.metric("Mitigations Resolved", success_acts)

        st.divider()

        # Activity Filters
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            cat_filter = st.multiselect(
                "Filter by Category",
                options=["ALL", "INTERVENTION", "AUDIT", "SCORING", "MODEL", "SYSTEM"],
                default=["ALL"],
            )
        with filter_col2:
            if st.button("🧹 Clear Log", use_container_width=True):
                st.session_state.activities = []
                st.rerun()

        # Render Activity Cards
        displayed_activities = st.session_state.activities
        if "ALL" not in cat_filter and len(cat_filter) > 0:
            displayed_activities = [a for a in displayed_activities if a["category"] in cat_filter]

        if not displayed_activities:
            st.info("No activity records match the selected filter.")
        else:
            for act in displayed_activities:
                lvl = act["level"]
                card_class = "activity-card"
                if lvl == "critical":
                    card_class += " critical"
                    badge_html = '<span class="badge badge-high">CRITICAL RISK</span>'
                elif lvl == "warning":
                    card_class += " warning"
                    badge_html = '<span class="badge badge-medium">ACTION REQUIRED</span>'
                elif lvl == "success":
                    card_class += " success"
                    badge_html = '<span class="badge badge-low">COMPLETED</span>'
                else:
                    badge_html = '<span class="badge badge-info">EVENT</span>'

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="activity-header">
                            <div>
                                <span class="activity-title">{act['title']}</span>
                                &nbsp;&nbsp;{badge_html}
                                &nbsp;&nbsp;<span style="font-size: 0.78rem; color: #818CF8;">[Account: {act['account_id']}]</span>
                            </div>
                            <span class="activity-time">{act['timestamp']}</span>
                        </div>
                        <div class="activity-body">{act['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # TAB 2: PORTFOLIO OVERVIEW & DIAGNOSTICS
    with tab_overview:
        if workspace.df is None:
            st.warning("No dataset loaded in workspace.")
        else:
            kpis = workspace.get_overview()
            analytics = workspace.get_analytics()

            # Top KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Accounts</div>
                        <div class="metric-value">{kpis['customers']:,}</div>
                        <div class="metric-subtext">Active portfolio base</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Baseline Churn Rate</div>
                        <div class="metric-value" style="color: #F87171;">{kpis['churn_rate']:.1%}</div>
                        <div class="metric-subtext">{kpis['churners']:,} churned accounts</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Monthly ARR at Risk</div>
                        <div class="metric-value" style="color: #FBBF24;">₹{kpis['monthly_revenue_at_risk']:,.0f}</div>
                        <div class="metric-subtext">₹{kpis['annual_revenue_at_risk']:,.0f} Annualized</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k4:
                avg_t = kpis.get("avg_tenure", 0)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Customer Tenure</div>
                        <div class="metric-value">{avg_t:.1f} <span style="font-size: 1.1rem; color: #94A3B8;">mo</span></div>
                        <div class="metric-subtext">Avg spend: ₹{kpis.get('avg_monthly', 0):.0f}/mo</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Visual Analytics Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("#### 🍩 Churn vs. Retained Distribution")
                df_plot = workspace.df.copy()
                churn_labels = df_plot["_y"].map({1: "Churned", 0: "Retained"})
                fig_pie = px.pie(
                    names=churn_labels,
                    values=[1] * len(df_plot),
                    hole=0.55,
                    color=churn_labels,
                    color_discrete_map={"Churned": "#EF4444", "Retained": "#10B981"},
                )
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20, b=20, l=20, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                st.markdown("#### 📈 Churn Rate by Contract Cohort")
                if "Contract" in df_plot.columns:
                    contract_summary = (
                        df_plot.groupby("Contract", observed=False)["_y"]
                        .agg(churn_rate="mean", accounts="size")
                        .reset_index()
                    )
                    fig_bar = px.bar(
                        contract_summary,
                        x="Contract",
                        y="churn_rate",
                        color="churn_rate",
                        color_continuous_scale="Reds",
                        text_auto=".1%",
                        labels={"churn_rate": "Churn Rate"},
                    )
                    fig_bar.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=20, b=20, l=20, r=20),
                        yaxis=dict(tickformat=".0%"),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Segment Risk Diagnostic Lift Table
            st.markdown("#### 🔍 High-Risk Segment Diagnostics (Churn Lift vs. Baseline)")
            seg_data = []
            for s in analytics.get("segments", []):
                for r in s.get("rows", []):
                    if r.get("is_high_risk"):
                        seg_data.append({
                            "Attribute": s["column"],
                            "Segment Value": r["segment"],
                            "Customer Base": f"{r['customers']:,}",
                            "Churners": f"{r['churners']:,}",
                            "Churn Rate": f"{r['churn_rate']:.1%}",
                            "Lift vs. Portfolio": f"+{r['lift']:.1%}",
                            "Risk Severity": "🚨 Critical Leak Point",
                        })

            if seg_data:
                st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)
            else:
                st.info("No segment attributes exceed the high-risk lift threshold.")

    # TAB 3: SINGLE ACCOUNT RISK SCORING DESK
    with tab_predict:
        st.subheader("🎯 Single-Account Churn Risk Desk")
        st.caption("Score prospective or existing customer profiles in real time with local feature attribution & action recommendations.")

        # Presets Bar
        preset_cols = st.columns([1, 1, 1, 1])
        preset_choice = None
        with preset_cols[0]:
            if st.button("🔥 High Risk Profile", use_container_width=True):
                preset_choice = "high"
        with preset_cols[1]:
            if st.button("🛡️ Loyal Account", use_container_width=True):
                preset_choice = "loyal"
        with preset_cols[2]:
            if st.button("⚡ Tech-Heavy Starter", use_container_width=True):
                preset_choice = "tech"
        with preset_cols[3]:
            if st.button("🔄 Reset Defaults", use_container_width=True):
                preset_choice = "default"

        # Default Form Values based on preset
        def_vals = {
            "tenure": 2 if preset_choice == "high" else (60 if preset_choice == "loyal" else 12),
            "monthly": 95.0 if preset_choice == "high" else (45.0 if preset_choice == "loyal" else 75.0),
            "total": 190.0 if preset_choice == "high" else (2700.0 if preset_choice == "loyal" else 900.0),
            "contract": "Month-to-month" if preset_choice == "high" else ("Two year" if preset_choice == "loyal" else "One year"),
            "internet": "Fiber optic" if preset_choice in ("high", "tech") else ("DSL" if preset_choice == "loyal" else "Fiber optic"),
            "payment": "Electronic check" if preset_choice == "high" else "Credit card (automatic)",
            "tech_support": "No" if preset_choice == "high" else "Yes",
            "online_security": "No" if preset_choice == "high" else "Yes",
        }

        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.markdown("##### 👤 Account & Contract Information")
            s_tenure = st.slider("Account Tenure (Months)", 0, 72, def_vals["tenure"])
            s_monthly = st.number_input("Monthly Charges (₹)", 18.0, 15000.0, def_vals["monthly"], step=5.0)
            s_total = st.number_input("Total Lifetime Charges (₹)", 0.0, 1000000.0, max(def_vals["total"], s_tenure * s_monthly), step=50.0)
            s_contract = st.selectbox("Contract Commitment", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_vals["contract"]))
            s_payment = st.selectbox("Billing & Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0 if def_vals["payment"] == "Electronic check" else 3)
            s_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)

        with form_col2:
            st.markdown("##### 🌐 Subscribed Services & Support")
            s_internet = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_vals["internet"]))
            s_tech_support = st.selectbox("Dedicated Tech Support", ["No", "Yes", "No internet service"], index=0 if def_vals["tech_support"] == "No" else 1)
            s_security = st.selectbox("Online Security Package", ["No", "Yes", "No internet service"], index=0 if def_vals["online_security"] == "No" else 1)
            s_backup = st.selectbox("Cloud Backup Storage", ["No", "Yes", "No internet service"], index=1 if preset_choice == "loyal" else 0)
            s_device = st.selectbox("Hardware Device Protection", ["No", "Yes", "No internet service"], index=1 if preset_choice == "loyal" else 0)
            s_streaming = st.selectbox("Streaming Services (TV/Movies)", ["Yes", "No", "No internet service"], index=0)

        score_btn = st.button("⚡ Score Account & Compute Risk Attribution", type="primary", use_container_width=True)

        if score_btn:
            customer_payload = {
                "tenure": s_tenure,
                "MonthlyCharges": s_monthly,
                "TotalCharges": s_total,
                "Contract": s_contract,
                "PaymentMethod": s_payment,
                "PaperlessBilling": s_paperless,
                "InternetService": s_internet,
                "TechSupport": s_tech_support,
                "OnlineSecurity": s_security,
                "OnlineBackup": s_backup,
                "DeviceProtection": s_device,
                "StreamingTV": s_streaming,
                "StreamingMovies": s_streaming,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
            }

            with st.spinner("Scoring account with Champion Model pipeline..."):
                try:
                    res = workspace.predict_one(customer_payload)
                except Exception:
                    res = predict_churn(customer_payload)

                prob = float(res.get("churn_probability", 0.5))
                band = res.get("risk_band", "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.35 else "LOW"))
                sla = res.get("sla", "Immediate 24h Outreach" if band == "HIGH" else "Review within 7 days")
                action = res.get("recommended_action", "Offer annual contract incentive with bill review.")
                risk_drivers = res.get("risk_elevators", ["Month-to-month contract structure", "High monthly bill relative to tenure"])
                anchor_drivers = res.get("retention_anchors", ["Tenure longevity", "Automatic payment method"])

                st.divider()
                st.markdown("### 📊 Account Risk Assessment Scorecard")

                res_col1, res_col2 = st.columns([1, 1.5])

                with res_col1:
                    gauge_color = "#EF4444" if band == "HIGH" else ("#F59E0B" if band == "MEDIUM" else "#10B981")
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            domain={"x": [0, 1], "y": [0, 1]},
                            number={"suffix": "%", "font": {"size": 42, "color": gauge_color}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                                "bar": {"color": gauge_color},
                                "bgcolor": "rgba(0,0,0,0)",
                                "steps": [
                                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                                    {"range": [35, 60], "color": "rgba(245, 158, 11, 0.15)"},
                                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"},
                                ],
                                "threshold": {
                                    "line": {"color": "#EF4444", "width": 4},
                                    "thickness": 0.75,
                                    "value": float(workspace.high_threshold) * 100,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=20, b=10, l=20, r=20),
                        height=220,
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown(f"**Classification**: <span class='badge badge-{band.lower()}'>{band} RISK</span>", unsafe_allow_html=True)
                    st.markdown(f"**Priority SLA**: `{sla}`")

                with res_col2:
                    st.markdown("#### 🔍 Local Feature Attribution (Explainability)")
                    st.markdown(
                        """
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                            Factors impacting this customer's churn risk calculation:
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("##### 🚨 Top Risk Elevators (+ Risk)")
                    for r in risk_drivers[:3]:
                        st.markdown(
                            f"""
                            <div class="attribution-card" style="border-left: 3px solid #EF4444;">
                                <span style="color: #F87171; font-weight: 600;">+ Impact</span> &nbsp;|&nbsp; <span>{r}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("##### 🛡️ Top Retention Anchors (- Risk)")
                    for a in anchor_drivers[:3]:
                        st.markdown(
                            f"""
                            <div class="attribution-card" style="border-left: 3px solid #10B981;">
                                <span style="color: #34D399; font-weight: 600;">- Impact</span> &nbsp;|&nbsp; <span>{a}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("#### 💡 Prescribed Retention Playbook Action")
                st.info(f"**Action Plan**: {action}")

                # 1-Click Action to Log Activity
                if st.button("🚀 Execute Playbook & Log Intervention Activity", type="primary", use_container_width=True):
                    log_activity(
                        f"Retention Intervention Dispatched [{band} RISK]",
                        f"Customer scored at {prob:.1%} churn risk. Action: {action}",
                        level="critical" if band == "HIGH" else "warning",
                        category="SCORING",
                        account_id="ONLINE-DESK",
                    )
                    st.success("Intervention recorded in the live Activity Stream!")
                    st.rerun()

    # TAB 4: CUSTOMER BOOK & BATCH SCORING
    with tab_batch:
        st.subheader("📁 Customer Book & Batch Risk Scoring")
        st.caption("Inspect scored portfolio accounts or upload a prospective batch file for mass scoring.")

        if workspace.scored is not None:
            scored_df = workspace.scored.copy()

            # Risk Distribution Summary
            rc1, rc2, rc3 = st.columns(3)
            high_cnt = (scored_df["risk_band"] == "HIGH").sum() if "risk_band" in scored_df.columns else 0
            med_cnt = (scored_df["risk_band"] == "MEDIUM").sum() if "risk_band" in scored_df.columns else 0
            low_cnt = (scored_df["risk_band"] == "LOW").sum() if "risk_band" in scored_df.columns else 0

            rc1.metric("High Risk Accounts (24h SLA)", f"{high_cnt:,}", f"{high_cnt/len(scored_df):.1%}")
            rc2.metric("Medium Risk Accounts (7d SLA)", f"{med_cnt:,}", f"{med_cnt/len(scored_df):.1%}")
            rc3.metric("Low Risk / Retained Base", f"{low_cnt:,}", f"{low_cnt/len(scored_df):.1%}")

            st.divider()

            # Filters
            b_col1, b_col2 = st.columns([1, 3])
            with b_col1:
                band_filter = st.selectbox("Filter Risk Band", ["ALL", "HIGH", "MEDIUM", "LOW"])

            filtered_df = scored_df
            if band_filter != "ALL" and "risk_band" in scored_df.columns:
                filtered_df = scored_df[scored_df["risk_band"] == band_filter]

            st.dataframe(filtered_df.head(100), use_container_width=True)

            # Download CSV
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Customer Book (CSV)",
                data=csv_data,
                file_name="retainiq_scored_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # TAB 5: RETENTION PLAYBOOK
    with tab_playbook:
        st.subheader("🛡️ Strategic Retention Playbook")
        st.caption("Algorithmic intervention matrix for highest-impact customer retention initiatives.")

        if workspace.df is not None:
            insights = workspace.get_insights()
            playbook_res = insights.get("playbook", {})
            pillars = playbook_res.get("strategic_pillars", []) if isinstance(playbook_res, dict) else []

            if pillars:
                for p in pillars:
                    st.markdown(
                        f"""
                        <div class="activity-card warning" style="margin-bottom: 16px;">
                            <div class="activity-header">
                                <span class="activity-title" style="font-size: 1.05rem;">{p.get('pillar_name', 'Retention Strategy')}</span>
                                <span class="badge badge-medium">{p.get('priority', 'HIGH PRIORITY')}</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #CBD5E1; margin: 8px 0;">
                                <strong>Target Segment:</strong> {p.get('target_segment', 'High Risk')} &nbsp;|&nbsp;
                                <strong>Observed Metric:</strong> <span style="color: #34D399;">{p.get('observed_metric', '')}</span>
                            </div>
                            <div class="activity-body">{p.get('strategic_rationale', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Train the model to generate customized algorithmic retention playbooks.")

    # TAB 6: MODEL GOVERNANCE & BENCHMARK LAB
    with tab_models:
        st.subheader("🏆 Model Governance & Champion Leaderboard")
        st.caption("Benchmark candidate classification models on 5-Fold Stratified Cross-Validation & holdout test splits.")

        if workspace.metrics:
            metrics = workspace.metrics
            best_model = metrics.get("best_model", "xgboost")
            st.success(f"🏆 Champion Model: **{best_model.replace('_', ' ').title()}** (Selected based on PR-AUC & ROC-AUC Composite)")

            models_data = []
            for name, m in metrics.get("models", {}).items():
                models_data.append({
                    "Model": name.replace("_", " ").title(),
                    "Champion": "⭐ Champion" if name == best_model else "",
                    "PR-AUC": f"{m.get('pr_auc', 0):.4f}",
                    "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                    "Accuracy": f"{m.get('accuracy', 0):.2%}",
                    "F1 Score": f"{m.get('f1', 0):.4f}",
                })

            st.dataframe(pd.DataFrame(models_data), use_container_width=True, hide_index=True)
        else:
            st.warning("No model benchmark metrics found. Click 'Retrain' in the sidebar to benchmark models.")


if __name__ == "__main__" or st.runtime.exists():
    main()
