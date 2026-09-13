"""Unit tests for model explainability."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explainability import (
    compute_global_feature_importance,
    explain_customer_prediction,
)
from models import train_and_compare_models
from validation import detect_schema, sanitize_dataframe


class TestExplainability(unittest.TestCase):
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

    def test_global_feature_importance(self):
        drivers = compute_global_feature_importance(self.pipeline, self.schema)
        self.assertTrue(len(drivers) > 0)
        self.assertIn("feature", drivers[0])
        self.assertIn("importance_pct", drivers[0])
        self.assertIn("description", drivers[0])
        # Sum of importance pct should be approximately 100%
        total_pct = sum(d["importance_pct"] for d in drivers)
        self.assertAlmostEqual(total_pct, 100.0, delta=1.5)

    def test_local_customer_explanation(self):
        sample = self.clean_df.iloc[0].to_dict()
        exp = explain_customer_prediction(self.pipeline, sample, self.schema, churn_probability=0.75)
        self.assertIn("top_risk_elevators", exp)
        self.assertIn("top_retention_anchors", exp)
        self.assertIn("risk_summary", exp)


if __name__ == "__main__":
    unittest.main()
