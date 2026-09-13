"""Unit tests for model training, metrics, cross-validation, and champion selection."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import (
    evaluate_predictions,
    get_candidate_models,
    train_and_compare_models,
)
from validation import detect_schema, sanitize_dataframe


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 120
        self.df = pd.DataFrame({
            "customerID": [f"ID{i}" for i in range(n_samples)],
            "tenure": np.random.randint(1, 72, size=n_samples),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Credit card"], size=n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples),
            "Churn": np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65]),
        })
        self.schema = detect_schema(self.df, "test.csv")
        self.clean_df = sanitize_dataframe(self.df, self.schema)

    def test_candidate_models_presence(self):
        candidates = get_candidate_models()
        self.assertIn("logistic_regression", candidates)
        self.assertIn("random_forest", candidates)
        self.assertIn("gradient_boosting", candidates)

    def test_evaluate_predictions_metrics(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.1, 0.85, 0.15])

        metrics = evaluate_predictions(y_true, y_pred, y_proba)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["roc_auc"], 0.5)

    def test_train_and_compare_models_end_to_end(self):
        pipeline, summary = train_and_compare_models(self.clean_df, self.schema, test_size=0.25)
        self.assertIsNotNone(pipeline)
        self.assertIn("best_model", summary)
        self.assertIn("selection_metric", summary)
        self.assertIn("models", summary)
        self.assertTrue(len(summary["models"]) >= 3)

        # Check CV metrics in candidate results
        champ_name = summary["best_model"]
        champ_metrics = summary["models"][champ_name]
        self.assertIn("cv_pr_auc_mean", champ_metrics)
        self.assertIn("cv_roc_auc_mean", champ_metrics)
        self.assertIn("confusion_matrix", champ_metrics)


if __name__ == "__main__":
    unittest.main()
