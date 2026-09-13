"""Streamlit dashboard for customer churn prediction."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import load_model, predict_churn  # noqa: E402
from preprocessing import DATA_PATH, load_data  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
)

METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_resource
def get_model():
    return load_model()


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df["Churn"] == "Yes").mean()
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Avg Tenure (months)", f"{df['tenure'].mean():.1f}")
    col4.metric("Avg Monthly Spend", f"₹{df['MonthlyCharges'].mean():.2f}")

    left, right = st.columns(2)
    with left:
        fig = px.pie(
            df,
            names="Churn",
            title="Churn Distribution",
            color="Churn",
            color_discrete_map={"Yes": "#ef553b", "No": "#636efa"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .reset_index(name="Churn Rate")
        )
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            title="Churn Rate by Contract Type",
            color="Churn Rate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_model_metrics() -> None:
    st.subheader("Model Performance")
    if not METRICS_PATH.exists():
        st.warning("Train the model first: `python src/train.py`")
        return

    metrics = json.loads(METRICS_PATH.read_text())
    st.info(f"Best model: **{metrics['best_model'].replace('_', ' ').title()}**")

    rows = []
    for name, values in metrics["models"].items():
        rows.append(
            {
                "Model": name.replace("_", " ").title(),
                "Accuracy": values["accuracy"],
                "F1 Score": values["f1"],
                "ROC AUC": values["roc_auc"],
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


def render_predictor() -> None:
    st.subheader("Churn Prediction")
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges (₹)", 18.0, 10000.0, 650.0)
        total_charges = st.number_input("Total Charges (₹)", 0.0, 500000.0, 5000.0)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    addon_cols = st.columns(5)
    online_security = addon_cols[0].selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = addon_cols[1].selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = addon_cols[2].selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = addon_cols[3].selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = addon_cols[4].selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    if st.button("Predict Churn", type="primary"):
        customer = {
            "SeniorCitizen": senior_citizen,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }
        result = predict_churn(customer)
        probability = result["churn_probability"]
        st.metric("Churn Probability", f"{probability:.1%}")
        if result["will_churn"]:
            st.error(result["prediction"])
        else:
            st.success(result["prediction"])


def main() -> None:
    st.title("Customer Churn Analysis Dashboard")
    st.caption("Telco customer churn exploration and prediction")

    df = get_dataset()
    tab_overview, tab_predict, tab_model = st.tabs(
        ["Overview", "Predict Churn", "Model Metrics"]
    )

    with tab_overview:
        render_overview(df)
    with tab_predict:
        try:
            get_model()
            render_predictor()
        except FileNotFoundError as exc:
            st.error(str(exc))
    with tab_model:
        render_model_metrics()


if __name__ == "__main__":
    main()
