# RetainIQ — Enterprise Customer Retention Intelligence

RetainIQ is an enterprise-grade AI analytics and machine learning platform for churn diagnosis, predictive risk scoring, model governance, and account intervention playbooks.

Every customer workspace is dynamically driven by the **uploaded extract** (CSV or Excel) — preprocessing pipelines, schema detection, leakage prevention, and model training are executed strictly on that data without cross-tenant pollution.

---

## Key Capabilities

1. **Intake & Automated Validation**:
   - Accepts `.csv`, `.xlsx`, and `.xls` files.
   - Robust schema detection (numeric, categorical, identifier, and target columns).
   - Pre-flight data quality checks: missingness audit, duplicate IDs, target class imbalance, and post-churn data leakage detection.
2. **Executive Portfolio Dashboard**:
   - Real-time customer volume, baseline churn rate, and monthly/annual recurring revenue at risk.
   - Cohort tenure distributions and portfolio composition charts.
3. **Segment Diagnostics & Lift Analysis**:
   - Empirical churn rates and percentage lift vs. portfolio baseline across all categorical attributes.
   - Highlights high-risk churn leak points (e.g., month-to-month contracts, electronic check payment).
4. **Model Lab & Champion Selection**:
   - Trains and benchmarks **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **XGBoost**.
   - Evaluates on Stratified 5-Fold Cross-Validation and holdout test splits.
   - Selects Champion Model based on PR-AUC (Precision-Recall AUC) and ROC-AUC composite score.
5. **Model Explainability & Feature Attribution**:
   - Global feature importance with intuitive business impact descriptions.
   - Local customer-level explanations: isolates top risk elevators and retention anchors for individual accounts.
6. **Risk Scoring Desk & Batch Ingestion**:
   - Interactive single-account scoring desk with dynamic form fields based on uploaded schema.
   - Batch scoring of prospective accounts (no churn label required) with downloadable enriched CSV.
   - Configurable risk thresholds (High, Medium, Low) and operational outreach SLAs.
7. **Retention Playbook Generator**:
   - Automated rule-based and model-guided action prescriptions.
   - Prioritized interventions with estimated impact and targeted accounts.
8. **Executive Exports & Delivery**:
   - **Executive Excel Workbook (.xlsx)**: Multi-tab, professionally styled workbook (Executive Summary, Segment Leak Points, Model Governance, Scored Customer Book).
   - **Printable Executive Briefing Report (HTML/PDF)**: Client-ready executive deck.
   - **Scored Book Export (CSV)**: Full customer extract enriched with churn probabilities, risk bands, and SLA priorities.

---

## Quick Start

### 1. Installation

```bash
cd Customer_Churn_Analysis
pip install -r requirements.txt
```

### 2. Launch RetainIQ Web Platform

```bash
# Launch via unified CLI
python app.py

# Or launch via run script
python run.py
```

Open your browser at **http://localhost:8000** (API documentation is available at **http://localhost:8000/api/docs**).

---

## Command Line Interface (CLI) Options

`app.py` provides convenient options for training, scoring, testing, and running prototypes:

```bash
python app.py                 # Launch RetainIQ FastAPI Web Platform (default port 8000)
python app.py --port 8080     # Launch on custom port
python app.py --train         # Train candidate models on baseline Telco dataset
python app.py --predict       # Run sample single-customer prediction & attribution
python app.py --test          # Run complete test suite (38 unit & integration tests)
python app.py --streamlit     # Launch lightweight Streamlit prototype
```

---

## Deployment & Docker

### Docker Container

```bash
# Build Docker image
docker build -t retainiq .

# Run container on port 8000
docker run -p 8000:8000 retainiq
```

### Cloud Production

Run behind any ASGI host (e.g. Render, Railway, Azure App Service, AWS App Runner, or Kubernetes):

```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000 --workers 4
```

Persist the `uploads/` directory on a persistent volume if you want client workspaces to persist across container restarts.

---

## Repository Architecture

```
Customer_Churn_Analysis/
├── app.py                     # Universal CLI and application entrypoint
├── run.py                     # Fast server launcher (uvicorn)
├── Dockerfile                 # Production Docker container definition
├── requirements.txt           # Python dependencies
├── data/                      # Baseline dataset
│   └── Telco-Customer-Churn.csv
├── src/                       # Core Analytics & ML Engine
│   ├── config.py              # Central platform configuration & paths
│   ├── engine.py              # Unified engine facade & form generators
│   ├── explainability.py      # Global importance & local customer attributions
│   ├── models.py              # ML candidate training, CV & champion selection
│   ├── playbook.py            # Automated retention playbook engine
│   ├── predict.py             # Standalone prediction module
│   ├── preprocessing.py       # ColumnTransformer & feature prep
│   ├── reporting.py           # Multi-tab Excel workbook & HTML briefing exports
│   ├── scoring.py             # Single & batch scoring with threshold engine
│   ├── train.py               # Standalone training script
│   ├── validation.py          # Data validation, schema & leakage detection
│   └── workspace.py           # Session isolation & state persistence
├── web/                       # Web Application Layer
│   ├── server.py              # FastAPI REST endpoints & session manager
│   ├── templates/index.html   # Modern responsive single-page application UI
│   └── static/                # Vanilla CSS design system & client JavaScript
│       ├── css/app.css
│       └── js/app.js
├── app/                       # Lightweight prototype
│   └── streamlit_app.py       # Streamlit interactive dashboard
├── tests/                     # Comprehensive Test Suite (38 tests)
│   ├── run_all_tests.py       # Test discovery runner
│   ├── test_api_e2e.py        # End-to-end HTTP API tests
│   ├── test_explainability.py # Feature importance & attribution tests
│   ├── test_full_platform.py  # Platform integration tests
│   ├── test_models.py         # ML algorithms & metric tests
│   ├── test_preprocessing.py  # Pipeline & imputation tests
│   ├── test_reporting.py      # Excel & HTML report tests
│   ├── test_scoring.py        # Scoring & threshold tests
│   └── test_validation.py     # Schema, leakage & data checks
└── models/                    # Serialized models & metrics
```

---

## Running Tests

Execute the automated test suite anytime:

```bash
python tests/run_all_tests.py
# or
python app.py --test
```

