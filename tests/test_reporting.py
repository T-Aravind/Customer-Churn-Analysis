"""Unit tests for reporting and exports (Excel and HTML)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import train_and_compare_models
from playbook import generate_retention_playbook
from reporting import (
    generate_excel_executive_workbook,
    generate_html_executive_report,
)
from scoring import score_batch_dataframe
from validation import detect_schema, sanitize_dataframe


class TestReporting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 80
        self.df = pd.DataFrame({
            "customerID": [f"ID{i}" for i in range(n)],
            "tenure": np.random.randint(1, 72, size=n),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Credit card"], size=n),
            "Churn": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
        })
        self.schema = detect_schema(self.df, "test.csv")
        self.clean_df = sanitize_dataframe(self.df, self.schema)
        self.pipeline, self.metrics = train_and_compare_models(self.clean_df, self.schema)
        self.scored = score_batch_dataframe(self.pipeline, self.clean_df, self.schema)

    def test_generate_playbook(self):
        pb = generate_retention_playbook(self.clean_df, self.schema, self.metrics)
        self.assertIn("strategic_pillars", pb)
        self.assertIn("operational_cadence", pb)
        self.assertTrue(len(pb["strategic_pillars"]) >= 2)

    def test_generate_excel_workbook(self):
        excel_bytes = generate_excel_executive_workbook(
            self.clean_df,
            self.schema,
            metrics=self.metrics,
            scored_df=self.scored,
            pipeline=self.pipeline,
        )
        self.assertIsInstance(excel_bytes, bytes)
        self.assertGreater(len(excel_bytes), 2000)

    def test_generate_html_report(self):
        html_str = generate_html_executive_report(
            self.clean_df,
            self.schema,
            metrics=self.metrics,
        )
        self.assertIsInstance(html_str, str)
        self.assertIn("RetainIQ Executive Churn Intelligence Briefing", html_str)
        self.assertIn("Model Performance", html_str)


if __name__ == "__main__":
    unittest.main()
