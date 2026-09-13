"""Data preprocessing pipelines and feature transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

from config import SAMPLE_PATH
from validation import Schema, load_file_bytes

DATA_PATH = SAMPLE_PATH


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Convenience data loader for scripts and interactive dashboards."""
    target_path = Path(path) if path else SAMPLE_PATH
    with open(target_path, "rb") as f:
        raw = f.read()
    df, _, _ = load_file_bytes(raw, filename=target_path.name, is_training=True)
    return df


def build_preprocessor(schema: Schema) -> ColumnTransformer:
    """Build a leak-free scikit-learn ColumnTransformer for numeric and categorical features."""
    transformers: list[tuple[str, Any, list[str]]] = []

    if schema.numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, schema.numeric))

    if schema.categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, schema.categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_transformed_feature_names(preprocessor: ColumnTransformer, schema: Schema) -> list[str]:
    """Retrieve output feature names from fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is unavailable
        names: list[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                names.extend([f"num__{c}" for c in cols])
            elif name == "cat":
                try:
                    encoder = trans.named_steps["encoder"]
                    encoded_cols = encoder.get_feature_names_out(cols)
                    names.extend([f"cat__{c}" for c in encoded_cols])
                except Exception:
                    names.extend([f"cat__{c}" for c in cols])
        return names


def prepare_xy(df: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from cleaned dataframe."""
    feature_cols = [c for c in schema.all_features if c in df.columns]
    X = df[feature_cols].copy()
    y = df["_y"] if "_y" in df.columns else df[schema.target]
    return X, y


def prepare_inference_features(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Prepare feature matrix for prediction, handling missing columns gracefully."""
    X = pd.DataFrame(index=df.index)
    for col in schema.numeric:
        if col in df.columns:
            X[col] = pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip(),
                errors="coerce",
            )
        else:
            X[col] = np.nan

    for col in schema.categorical:
        if col in df.columns:
            X[col] = df[col].astype(str).str.strip().replace(["nan", "none", "null", ""], np.nan)
        else:
            X[col] = np.nan

    return X[schema.all_features]
