"""Model training, cross-validation, evaluation metrics, and champion selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from config import CV_FOLDS, RANDOM_STATE, TEST_SPLIT_RATIO, logger
from preprocessing import build_preprocessor, prepare_xy
from validation import Schema

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_candidate_models() -> dict[str, Any]:
    """Return configured classifier algorithms for comparison."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


def evaluate_predictions(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Compute comprehensive classification performance metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)
    total_test = max(1, len(y_true))

    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn_rate": round(tn / max(1, tn + fp), 4),
            "fp_rate": round(fp / max(1, tn + fp), 4),
            "fn_rate": round(fn / max(1, fn + tp), 4),
            "tp_rate": round(tp / max(1, fn + tp), 4),
        },
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def compute_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
) -> dict[str, Any]:
    """Perform Stratified K-Fold cross validation on training set."""
    min_class_count = int(y.value_counts().min())
    actual_splits = max(2, min(n_splits, min_class_count))

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
    }

    try:
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            "cv_folds": actual_splits,
            "cv_roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
            "cv_pr_auc_mean": round(float(np.mean(scores["test_pr_auc"])), 4),
            "cv_pr_auc_std": round(float(np.std(scores["test_pr_auc"])), 4),
            "cv_f1_mean": round(float(np.mean(scores["test_f1"])), 4),
            "cv_f1_std": round(float(np.std(scores["test_f1"])), 4),
        }
    except Exception as exc:
        logger.warning(f"Cross-validation warning: {exc}")
        return {
            "cv_folds": actual_splits,
            "cv_roc_auc_mean": 0.0,
            "cv_roc_auc_std": 0.0,
            "cv_pr_auc_mean": 0.0,
            "cv_pr_auc_std": 0.0,
            "cv_f1_mean": 0.0,
            "cv_f1_std": 0.0,
        }


def train_and_compare_models(
    df: pd.DataFrame,
    schema: Schema,
    test_size: float = TEST_SPLIT_RATIO,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train candidate models with stratified splitting & cross validation, then return champion pipeline + summary metadata."""
    X, y = prepare_xy(df, schema)

    if y.nunique() < 2:
        raise ValueError("Target column must contain both churned and retained customers.")

    min_class_count = int(y.value_counts().min())
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    candidates = get_candidate_models()
    model_results: dict[str, dict[str, Any]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        preprocessor = build_preprocessor(schema)
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        # 1. Stratified Cross-Validation on Training Set
        cv_metrics = compute_cross_validation(pipe, X_train, y_train, n_splits=CV_FOLDS)

        # 2. Fit on Training Split & Evaluate on Holdout Test Split
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_predictions(y_test, y_pred, y_proba)
        test_metrics.update(cv_metrics)
        test_metrics["training_status"] = "Completed"

        model_results[name] = test_metrics
        fitted_pipelines[name] = pipe

    # Champion Model Selection:
    # Use PR-AUC (Average Precision) combined with ROC-AUC as the primary churn metric
    # because churn datasets are inherently imbalanced and PR-AUC highlights positive class identification.
    def champion_score(m_name: str) -> float:
        m = model_results[m_name]
        # Weighted rank score: 0.6 * PR-AUC + 0.4 * ROC-AUC
        return (0.6 * m["pr_auc"]) + (0.4 * m["roc_auc"])

    best_name = max(model_results.keys(), key=champion_score)

    # Train final champion model on full dataset (X, y)
    champion_preprocessor = build_preprocessor(schema)
    champion_estimator = get_candidate_models()[best_name]
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", champion_preprocessor),
            ("classifier", champion_estimator),
        ]
    )
    final_pipeline.fit(X, y)

    now_iso = datetime.now(timezone.utc).isoformat()
    now_formatted = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    summary = {
        "model_version": "v1.2.0",
        "best_model": best_name,
        "best_model_display": best_name.replace("_", " ").title(),
        "selection_metric": "PR-AUC (Precision-Recall AUC) & ROC-AUC Composite",
        "trained_at_iso": now_iso,
        "trained_at_formatted": now_formatted,
        "dataset_filename": schema.filename,
        "rows_total": len(df),
        "rows_train": len(X_train),
        "rows_test": len(X_test),
        "test_churn_count": int(y_test.sum()),
        "test_retained_count": int((y_test == 0).sum()),
        "portfolio_churn_rate": round(float(y.mean()), 4),
        "feature_count": len(schema.all_features),
        "models": model_results,
    }

    return final_pipeline, summary
