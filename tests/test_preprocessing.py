"""Unit tests for preprocessing pipelines."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
    prepare_inference_features,
    prepare_xy,
)
from validation import detect_schema, sanitize_dataframe


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "customerID": [f"ID{i}" for i in range(60)],
            "tenure": np.random.randint(1, 72, size=60),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=60),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=60),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Credit card"], size=60),
            "Churn": np.random.choice(["Yes", "No"], size=60, p=[0.3, 0.7]),
        })
        self.schema = detect_schema(self.df, "test.csv")
        self.clean_df = sanitize_dataframe(self.df, self.schema)

    def test_preprocessor_fit_transform(self):
        X, y = prepare_xy(self.clean_df, self.schema)
        preprocessor = build_preprocessor(self.schema)
        transformed = preprocessor.fit_transform(X)
        self.assertEqual(transformed.shape[0], len(X))
        self.assertGreater(transformed.shape[1], len(self.schema.numeric))

    def test_feature_names_out(self):
        X, y = prepare_xy(self.clean_df, self.schema)
        preprocessor = build_preprocessor(self.schema)
        preprocessor.fit(X)
        names = get_transformed_feature_names(preprocessor, self.schema)
        self.assertTrue(any("tenure" in n for n in names))
        self.assertTrue(any("Contract" in n for n in names))

    def test_prepare_inference_features(self):
        sample_row = {"tenure": 12, "MonthlyCharges": 75.0, "Contract": "Month-to-month", "PaymentMethod": "Electronic check"}
        inference_df = prepare_inference_features(pd.DataFrame([sample_row]), self.schema)
        self.assertEqual(inference_df.shape[0], 1)
        self.assertIn("tenure", inference_df.columns)
        self.assertIn("Contract", inference_df.columns)


if __name__ == "__main__":
    unittest.main()
