"""RetainIQ Unified Analytics Engine (Facade)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import SAMPLE_PATH
from explainability import compute_global_feature_importance, explain_customer_prediction
from models import train_and_compare_models
from playbook import generate_retention_playbook
from preprocessing import build_preprocessor, prepare_inference_features, prepare_xy
from reporting import generate_excel_executive_workbook, generate_html_executive_report
from scoring import (
    load_batch_file_and_score,
    score_batch_dataframe,
    score_single_customer,
)
from validation import (
    Schema,
    detect_schema,
    load_file_bytes,
    normalize_target,
    sanitize_dataframe,
    validate_dataset,
)
from workspace import Workspace, get_workspace


def field_options(df: pd.DataFrame, schema: Schema) -> dict[str, Any]:
    """Generate dynamic form fields and defaults for single customer risk scoring desk."""
    fields = []
    for col in schema.numeric:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        min_val = float(series.min()) if len(series) else 0.0
        max_val = float(series.max()) if len(series) else 100.0
        mean_val = float(series.mean()) if len(series) else 0.0
        fields.append({
            "name": col,
            "type": "number",
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "mean": round(mean_val, 2),
            "default": round(mean_val, 2),
        })

    for col in schema.categorical:
        values = [str(v) for v in df[col].dropna().astype(str).unique() if str(v).lower() not in ("nan", "none", "null")][:50]
        fields.append({
            "name": col,
            "type": "select",
            "options": values,
            "default": values[0] if values else "",
        })

    return {
        "fields": fields,
        "id_column": schema.id_column,
        "feature_count": len(fields),
    }
