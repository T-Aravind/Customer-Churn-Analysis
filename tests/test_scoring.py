"""Unit tests for customer risk scoring, threshold configuration, and batch prediction."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import train_and_compare_models
from scoring import (
    assign_priority,
    assign_risk_band,
    score_batch_dataframe,
    score_single_customer,
)
from validation import detect_schema, sanitize_dataframe


class TestScoring(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "customerID": [f"ID{i}" for i in range(n)],
            "tenure": np.random.randint(1, 72, size=n),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Credit card"], size=n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
            "Churn": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
        })
        self.schema = detect_schema(self.df, "test.csv")
        self.clean_df = sanitize_dataframe(self.df, self.schema)
        self.pipeline, self.summary = train_and_compare_models(self.clean_df, self.schema)

    def test_assign_risk_band(self):
        self.assertEqual(assign_risk_band(0.75, high_threshold=0.60, med_threshold=0.35), "High")
        self.assertEqual(assign_risk_band(0.45, high_threshold=0.60, med_threshold=0.35), "Medium")
        self.assertEqual(assign_risk_band(0.20, high_threshold=0.60, med_threshold=0.35), "Low")

    def test_score_single_customer(self):
        sample_payload = {
            "tenure": 2,
            "MonthlyCharges": 95.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
        }
        res = score_single_customer(self.pipeline, sample_payload, self.schema)
        self.assertIn("churn_probability", res)
        self.assertIn("risk_band", res)
        self.assertIn("predicted_status", res)
        self.assertIn("recommended_action", res)
        self.assertIn("local_explanation", res)
        self.assertTrue(0.0 <= res["churn_probability"] <= 1.0)

    def test_score_batch_without_churn_column(self):
        batch_df = self.df.drop(columns=["Churn"]).copy()
        scored_df = score_batch_dataframe(self.pipeline, batch_df, self.schema)
        self.assertIn("churn_probability", scored_df.columns)
        self.assertIn("risk_band", scored_df.columns)
        self.assertIn("predicted_status", scored_df.columns)
        self.assertIn("recommended_action", scored_df.columns)
        self.assertEqual(len(scored_df), len(batch_df))


if __name__ == "__main__":
    unittest.main()
