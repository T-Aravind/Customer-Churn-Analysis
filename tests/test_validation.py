"""Unit tests for data validation, schema detection, and leakage prevention."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validation import (
    detect_schema,
    load_file_bytes,
    normalize_target,
    sanitize_dataframe,
    validate_dataset,
)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "customerID": ["C01", "C02", "C03", "C04", "C05"] * 10,
            "tenure": np.random.randint(1, 72, size=50),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=50),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=50),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=50),
            "Churn": np.random.choice(["Yes", "No"], size=50, p=[0.3, 0.7]),
        })

    def test_normalize_target_strings(self):
        s = pd.Series(["Yes", "no", "YES", "No", "Churn", "Stay"])
        norm, invalid = normalize_target(s)
        self.assertEqual(list(norm), [1, 0, 1, 0, 1, 0])
        self.assertEqual(invalid, 0)

    def test_normalize_target_numeric(self):
        s = pd.Series([1, 0, 1, 0, 0])
        norm, invalid = normalize_target(s)
        self.assertEqual(list(norm), [1, 0, 1, 0, 0])

    def test_detect_schema(self):
        schema = detect_schema(self.sample_df, "test.csv")
        self.assertEqual(schema.target, "Churn")
        self.assertEqual(schema.id_column, "customerID")
        self.assertIn("tenure", schema.numeric)
        self.assertIn("MonthlyCharges", schema.numeric)
        self.assertIn("Contract", schema.categorical)
        self.assertIn("InternetService", schema.categorical)

    def test_leakage_detection(self):
        leak_df = self.sample_df.copy()
        leak_df["cancellation_date"] = "2024-01-01"
        leak_df["churn_reason"] = "Price too high"

        schema = detect_schema(leak_df, "test_leak.csv")
        self.assertIn("cancellation_date", schema.leakage_columns)
        self.assertIn("churn_reason", schema.leakage_columns)
        self.assertNotIn("cancellation_date", schema.all_features)
        self.assertNotIn("churn_reason", schema.all_features)

    def test_validate_dataset_checks(self):
        report = validate_dataset(self.sample_df, "test.csv", is_training=True)
        self.assertTrue(report["is_valid"])
        self.assertIn(report["status"], ["valid", "warning"])
        self.assertTrue(len(report["checks"]) >= 6)

    def test_duplicate_id_warning(self):
        dup_df = self.sample_df.copy()
        dup_df.loc[0, "customerID"] = dup_df.loc[1, "customerID"]
        report = validate_dataset(dup_df, "dup.csv", is_training=True)
        self.assertTrue(any("duplicate" in str(w).lower() for w in report["warnings"]))

    def test_insufficient_rows_error(self):
        small_df = self.sample_df.head(10)
        report = validate_dataset(small_df, "small.csv", is_training=True)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("insufficient" in str(e).lower() for e in report["errors"]))

    def test_flexible_target_detection(self):
        # Multi-word column names
        for col_name in ["Customer Churn", "Attrition_Flag", "Churn Value", "Did_Churn", "Exited"]:
            df = self.sample_df.copy().rename(columns={"Churn": col_name})
            schema = detect_schema(df, "test.csv")
            self.assertEqual(schema.target, col_name)

    def test_value_based_target_detection(self):
        # Column named 'Outcome' with Churned/Retained values
        df = self.sample_df.copy().rename(columns={"Churn": "AccountOutcome"})
        df["AccountOutcome"] = np.random.choice(["Churned", "Stayed"], size=len(df))
        schema = detect_schema(df, "test.csv")
        self.assertEqual(schema.target, "AccountOutcome")

    def test_excel_loading_and_multi_sheet(self):
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Sheet 1: Empty cover notes
            pd.DataFrame({"Notes": ["Confidential report"]}).to_excel(writer, sheet_name="Cover", index=False)
            # Sheet 2: Real dataset
            self.sample_df.to_excel(writer, sheet_name="Data", index=False)

        clean_df, schema, report = load_file_bytes(buf.getvalue(), "customers.xlsx", is_training=True)
        self.assertTrue(report["is_valid"])
        self.assertEqual(schema.target, "Churn")
        self.assertEqual(len(clean_df), len(self.sample_df))


if __name__ == "__main__":
    unittest.main()

