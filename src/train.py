"""Standalone training script for RetainIQ."""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib

from config import MODELS_DIR, SAMPLE_PATH
from models import train_and_compare_models
from validation import load_file_bytes


def main():
    print(f"Loading data from {SAMPLE_PATH}...")
    with open(SAMPLE_PATH, "rb") as f:
        raw = f.read()

    df, schema, report = load_file_bytes(raw, "Telco-Customer-Churn.csv", is_training=True)
    print(f"Validation status: {report['status']} ({len(df)} rows, {len(schema.all_features)} features)")

    print("Training candidate models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)...")
    pipeline, summary = train_and_compare_models(df, schema)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "churn_model.joblib")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nChampion Model Selected: {summary['best_model_display']}")
    print(f"Selection criteria: {summary['selection_metric']}\n")
    print(f"{'Algorithm':<25} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10} {'Accuracy':<10} {'5-Fold CV PR-AUC':<18}")
    print("-" * 85)
    for name, m in summary["models"].items():
        cv_str = f"{m.get('cv_pr_auc_mean', 0):.4f} ± {m.get('cv_pr_auc_std', 0):.4f}"
        print(f"{name.replace('_', ' ').title():<25} {m['roc_auc']:<10.4f} {m['pr_auc']:<10.4f} {m['f1']:<10.4f} {m['accuracy']:<10.2%} {cv_str:<18}")


if __name__ == "__main__":
    main()
