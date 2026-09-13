"""End-to-end integration test against live FastAPI server."""

import json
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"


def run_e2e():
    base_url = "http://localhost:8000"

    print("1. Healthcheck...")
    req = urllib.request.urlopen(f"{base_url}/api/health")
    health = json.loads(req.read().decode("utf-8"))
    assert health["ok"] is True
    print("   Health OK:", health)

    print("2. Upload & Validation...")
    with open(SAMPLE_PATH, "rb") as f:
        file_bytes = f.read()

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="Telco-Customer-Churn.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    upload_req = urllib.request.Request(
        f"{base_url}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    upload_res = urllib.request.urlopen(upload_req)
    session_cookie = upload_res.headers.get("Set-Cookie", "")
    session_val = session_cookie.split(";")[0] if session_cookie else ""
    res_json = json.loads(upload_res.read().decode("utf-8"))
    assert res_json["validation"]["is_valid"] is True
    print(f"   Upload OK: {res_json['schema']['rows']} rows, status: {res_json['validation']['status']}")

    headers = {"Cookie": session_val}

    print("3. Executive Overview KPIs...")
    ov_req = urllib.request.Request(f"{base_url}/api/overview", headers=headers)
    ov = json.loads(urllib.request.urlopen(ov_req).read().decode("utf-8"))
    assert ov["kpis"]["customers"] > 7000
    print(f"   Overview OK: {ov['kpis']['customers']:,} accounts, {ov['kpis']['churn_rate']:.1%} churn rate")

    print("4. Training Candidate Models...")
    train_req = urllib.request.Request(
        f"{base_url}/api/train",
        data=b"{}",
        headers={"Cookie": session_val, "Content-Type": "application/json"},
    )
    train_res = json.loads(urllib.request.urlopen(train_req).read().decode("utf-8"))
    assert "best_model" in train_res
    print(f"   Champion Model: {train_res['best_model_display']} (PR-AUC: {train_res['models'][train_res['best_model']]['pr_auc']})")

    print("5. Model Insights & Explainability...")
    ins_req = urllib.request.Request(f"{base_url}/api/insights", headers=headers)
    ins = json.loads(urllib.request.urlopen(ins_req).read().decode("utf-8"))
    assert len(ins["global_drivers"]) > 0
    print(f"   Top Driver: #{ins['global_drivers'][0]['rank']} {ins['global_drivers'][0]['feature']} ({ins['global_drivers'][0]['importance_pct']}%)")

    print("6. Single Customer Scoring with Attribution...")
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
    pred_req = urllib.request.Request(
        f"{base_url}/api/predict",
        data=json.dumps(sample_cust).encode("utf-8"),
        headers={"Cookie": session_val, "Content-Type": "application/json"},
    )
    pred_res = json.loads(urllib.request.urlopen(pred_req).read().decode("utf-8"))
    assert 0.0 <= pred_res["churn_probability"] <= 1.0
    print(f"   Prediction: {pred_res['churn_probability_pct']}% prob, Risk: {pred_res['risk_band']}")
    print(f"   Action: {pred_res['recommended_action']}")

    print("7. Batch Customer Scoring (Without Churn Column)...")
    batch_csv = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "1001-TEST,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,85.5,171.0\n"
        "1002-TEST,Male,0,Yes,Yes,65,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,Yes,Two year,No,Credit card,60.2,3913.0\n"
    ).encode("utf-8")

    batch_boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    batch_body = (
        f"--{batch_boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="prospective_accounts.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8") + batch_csv + f"\r\n--{batch_boundary}--\r\n".encode("utf-8")

    batch_req = urllib.request.Request(
        f"{base_url}/api/predict-batch",
        data=batch_body,
        headers={"Cookie": session_val, "Content-Type": f"multipart/form-data; boundary={batch_boundary}"},
    )
    batch_res = urllib.request.urlopen(batch_req)
    scored_csv_text = batch_res.read().decode("utf-8")
    assert "churn_probability" in scored_csv_text
    assert "risk_band" in scored_csv_text
    print("   Batch Scoring OK: Received enriched CSV with probabilities & risk tiers.")

    print("8. Executive Excel Workbook Export...")
    excel_req = urllib.request.Request(f"{base_url}/api/export/excel", headers=headers)
    excel_bytes = urllib.request.urlopen(excel_req).read()
    assert len(excel_bytes) > 5000
    print(f"   Excel Export OK: {len(excel_bytes):,} bytes workbook generated.")

    print("9. Executive HTML/PDF Report Export...")
    report_req = urllib.request.Request(f"{base_url}/api/export/report", headers=headers)
    report_html = urllib.request.urlopen(report_req).read().decode("utf-8")
    assert "RetainIQ Executive Churn Intelligence Briefing" in report_html
    print(f"   HTML Report OK: {len(report_html):,} characters briefing rendered.")

    print("\n=======================================================")
    print("  ALL API ENDPOINTS & WORKFLOWS VERIFIED SUCCESSFULLY! ")
    print("=======================================================\n")


if __name__ == "__main__":
    run_e2e()
