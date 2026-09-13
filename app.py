"""RetainIQ Unified Application Entry Point.

Usage:
  python app.py                 # Launch RetainIQ FastAPI Web Platform on port 8000
  python app.py --port 8080     # Launch on custom port
  python app.py --train         # Train candidate models on sample data
  python app.py --predict       # Run sample prediction
  python app.py --test          # Run full test suite
  python app.py --streamlit     # Launch lightweight Streamlit prototype
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RetainIQ — Enterprise Customer Retention Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the web server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the web server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train candidate models on baseline Telco dataset",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run single-account risk prediction demonstration",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit and integration test suite",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Launch the lightweight Streamlit prototype",
    )

    args = parser.parse_args()

    if args.train:
        from src.train import main as train_main
        train_main()
    elif args.predict:
        import src.predict as predict_module
        if not (PROJECT_ROOT / "models" / "churn_model.joblib").exists():
            print("Model not found. Training first...")
            from src.train import main as train_main
            train_main()
        sample_cust = {
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
        res = predict_module.predict_churn(sample_cust)
        print("\n--- Account Risk Scoring ---")
        print(f"Churn Probability: {res['churn_probability_pct']}%")
        print(f"Risk Band: {res['risk_band']}")
        print(f"Priority SLA: {res['priority']}")
        print(f"Top Driver: {res['top_driver']}")
        print(f"Recommended Action: {res['recommended_action']}")
    elif args.test:
        from tests.run_all_tests import run_suite
        run_suite()
    elif args.streamlit:
        cmd = [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py")]
        subprocess.run(cmd)
    else:
        import uvicorn
        print(f"\n=======================================================")
        print(f"  Starting RetainIQ Platform on http://{args.host}:{args.port}")
        print(f"  API Docs available at http://{args.host}:{args.port}/api/docs")
        print(f"=======================================================\n")
        uvicorn.run("web.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
