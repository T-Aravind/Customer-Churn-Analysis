"""Comprehensive Full-Platform Automation & Quality Assurance Test Suite for RetainIQ."""

import http.cookiejar
import json
import sys
import unittest
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import threading
import time
import uvicorn
from web.server import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
BASE_URL = "http://127.0.0.1:8000"


class RetainIQPlatformTests(unittest.TestCase):
    _server_thread = None

    @classmethod
    def setUpClass(cls):
        cls.cookie_jar = http.cookiejar.CookieJar()
        cls.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cls.cookie_jar)
        )

        # Check if an external server is already reachable
        try:
            req = urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=1)
            if req.status == 200:
                return
        except Exception:
            pass

        class ServerThread(threading.Thread):
            def __init__(self):
                super().__init__(daemon=True)
                config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="error")
                self.server = uvicorn.Server(config=config)

            def run(self):
                self.server.run()

            def stop(self):
                self.server.should_exit = True

        cls._server_thread = ServerThread()
        cls._server_thread.start()

        for _ in range(40):
            try:
                res = urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=1)
                if res.status == 200:
                    break
            except Exception:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        if cls._server_thread is not None:
            cls._server_thread.stop()

    def request(self, path: str, method: str = "GET", data: bytes = None, headers: dict = None):
        url = f"{BASE_URL}{path}"
        req = urllib.request.Request(url, data=data, method=method)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        return self.opener.open(req)

    def test_01_static_and_html_delivery(self):
        """Verify HTML template, CSS styling, and JavaScript logic are served."""
        res = self.request("/")
        self.assertEqual(res.status, 200)
        html = res.read().decode("utf-8")
        self.assertIn("RetainIQ", html)
        self.assertIn("sidebar", html)
        self.assertIn("/static/css/app.css", html)
        self.assertIn("/static/js/app.js", html)

        # Static CSS
        css_res = self.request("/static/css/app.css")
        self.assertEqual(css_res.status, 200)
        css = css_res.read().decode("utf-8")
        self.assertIn(":root", css)

        # Static JS
        js_res = self.request("/static/js/app.js")
        self.assertEqual(js_res.status, 200)
        js = js_res.read().decode("utf-8")
        self.assertIn("pages", js)

    def test_02_health_and_sample_download(self):
        """Verify healthcheck and sample dataset download."""
        res = self.request("/api/health")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("product"), "RetainIQ")

        sample_res = self.request("/api/sample")
        self.assertEqual(sample_res.status, 200)
        sample_csv = sample_res.read().decode("utf-8")
        self.assertIn("customerID", sample_csv)
        self.assertIn("Churn", sample_csv)

    def test_03_data_upload_and_validation(self):
        """Test dataset upload, schema parsing, and data quality validation."""
        with open(SAMPLE_PATH, "rb") as f:
            file_bytes = f.read()

        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="Telco-Customer-Churn.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        res = self.request(
            "/api/upload",
            method="POST",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertTrue(data["validation"]["is_valid"])
        self.assertEqual(data["schema"]["rows"], 7043)
        self.assertAlmostEqual(data["kpis"]["churn_rate"], 0.265, delta=0.01)
        self.assertEqual(len(data["schema"]["leakage_columns"]), 0)

    def test_04_overview_kpis(self):
        """Test executive dashboard KPIs calculations."""
        res = self.request("/api/overview")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        kpis = data["kpis"]
        self.assertIn("customers", kpis)
        self.assertIn("churn_rate", kpis)
        self.assertIn("monthly_revenue_at_risk", kpis)
        self.assertEqual(kpis["customers"], 7043)

    def test_05_analytics_diagnostics(self):
        """Test categorical segment lift analysis and preview."""
        res = self.request("/api/analytics")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertIn("segments", data)
        self.assertIn("tenure_bins", data)
        self.assertIn("preview", data)
        self.assertTrue(len(data["segments"]) > 0)
        self.assertTrue(len(data["preview"]) > 0)

    def test_06_model_training_and_benchmarking(self):
        """Test multi-model training, cross-validation, and champion selection."""
        res = self.request(
            "/api/train",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertIn("best_model", data)
        self.assertIn("models", data)
        models = data["models"]
        for expected_key in ["logistic_regression", "random_forest", "gradient_boosting"]:
            self.assertIn(expected_key, models, f"Model key {expected_key} missing in evaluation")

    def test_07_model_insights_and_explainability(self):
        """Test global feature importance and governance scorecard."""
        res = self.request("/api/insights")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertIn("global_drivers", data)
        self.assertTrue(len(data["global_drivers"]) > 0)
        self.assertIn("model_governance", data)
        self.assertIn("playbook", data)

    def test_08_dynamic_form_fields(self):
        """Test retrieval of form fields for single-account desk."""
        res = self.request("/api/fields")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertIn("fields", data)
        self.assertTrue(len(data["fields"]) > 5)

    def test_09_single_customer_prediction(self):
        """Test scoring a single customer and receiving risk attribution."""
        customer = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
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
            "MonthlyCharges": 89.0,
            "TotalCharges": 89.0,
        }
        res = self.request(
            "/api/predict",
            method="POST",
            data=json.dumps(customer).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertIn("churn_probability", data)
        self.assertIn("risk_band", data)
        self.assertIn("local_explanation", data)
        self.assertIn("top_driver", data)
        self.assertIn("recommended_action", data)
        self.assertEqual(data["risk_band"], "High")

    def test_10_batch_customer_prediction(self):
        """Test scoring a batch CSV file without target column."""
        batch_csv = (
            "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
            "AC-100,Female,0,No,No,3,Yes,No,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,85.5,256.5\n"
            "AC-200,Male,0,Yes,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,Yes,Two year,No,Credit card,55.0,3300.0\n"
        ).encode("utf-8")

        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="prospective_batch.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + batch_csv + f"\r\n--{boundary}--\r\n".encode("utf-8")

        res = self.request(
            "/api/predict-batch",
            method="POST",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(res.status, 200)
        scored_csv = res.read().decode("utf-8")
        self.assertIn("churn_probability", scored_csv)
        self.assertIn("risk_band", scored_csv)
        self.assertIn("AC-100", scored_csv)
        self.assertIn("AC-200", scored_csv)

    def test_11_threshold_adjustment(self):
        """Test customizing risk classification thresholds."""
        res = self.request(
            "/api/thresholds",
            method="POST",
            data=json.dumps({"high": 0.75, "medium": 0.40}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertEqual(data["high"], 0.75)
        self.assertEqual(data["medium"], 0.40)

    def test_12_exports(self):
        """Test Excel workbook, HTML report, and enriched customer book exports."""
        # Excel
        res_xl = self.request("/api/export/excel")
        self.assertEqual(res_xl.status, 200)
        xl_data = res_xl.read()
        self.assertTrue(len(xl_data) > 5000)

        # HTML Report
        res_rep = self.request("/api/export/report")
        self.assertEqual(res_rep.status, 200)
        rep_html = res_rep.read().decode("utf-8")
        self.assertIn("Executive Churn Intelligence Briefing", rep_html)

        # Book
        res_book = self.request("/api/export/book")
        self.assertEqual(res_book.status, 200)
        book_csv = res_book.read().decode("utf-8")
        self.assertIn("customerID", book_csv)

    def test_13_error_handling(self):
        """Verify defensive error handling on invalid requests."""
        # Upload empty file
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="empty.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n\r\n--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{BASE_URL}/api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            self.opener.open(req)
            self.fail("Should have raised HTTP 400 for empty file")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_14_workspace_reset(self):
        """Test workspace session reset clears state."""
        res = self.request(
            "/api/reset",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertTrue(data.get("ok"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
