"""Data intake, schema detection, leakage prevention, and validation suite."""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import MIN_TRAIN_ROWS, logger

def _norm(name: str) -> str:
    """Normalize string for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


TARGET_ALIASES = {
    _norm(kw)
    for kw in {
        "churn",
        "exited",
        "attrition",
        "left",
        "is_churn",
        "churned",
        "cancelled",
        "canceled",
        "target",
        "status",
        "churn_label",
        "churn_flag",
        "attrition_flag",
        "customer_status",
        "churn_status",
        "churn_value",
        "churn_indicator",
        "is_attrited",
        "has_churned",
        "did_churn",
        "is_churned",
        "lost_customer",
        "churn_risk",
        "retention_status",
        "retention_flag",
        "attrited",
        "churned_customer",
        "customer_churn",
        "churn_binary",
        "churn_yn",
        "churn_yes_no",
        "cancellation_status",
        "attrition_status",
        "churn_category",
        "churn_class",
        "churn_state",
        "churn_event",
    }
}

ID_ALIASES = {
    _norm(kw)
    for kw in {
        "customerid",
        "customer_id",
        "userid",
        "user_id",
        "id",
        "accountid",
        "account_id",
        "client_id",
        "clientid",
        "subscriber_id",
        "subscriberid",
        "sub_id",
        "cust_id",
        "member_id",
        "memberid",
    }
}

LEAKAGE_KEYWORDS = {
    _norm(kw)
    for kw in {
        "churn_date",
        "cancellation_date",
        "cancel_date",
        "exit_date",
        "churn_reason",
        "cancellation_reason",
        "cancel_reason",
        "exit_reason",
        "refund_after_cancel",
        "termination_code",
        "status_after_churn",
        "account_closed",
        "closed_date",
        "closure_date",
        "deactivation_date",
        "disconnection_date",
        "cancellation",
        "churned_date",
    }
}


@dataclass
class Schema:
    target: str
    id_column: str | None
    numeric: list[str]
    categorical: list[str]
    date_columns: list[str]
    all_features: list[str]
    leakage_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    filename: str = ""
    rows: int = 0
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_target(series: pd.Series) -> tuple[pd.Series, int]:
    """Normalize target to binary 0/1 integer. Returns (normalized_series, invalid_count)."""
    if series.dtype == bool:
        return series.astype(int), 0

    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return series.fillna(0).astype(int), int(series.isna().sum())
        # Multi-valued numeric target
        return (series.fillna(0) > 0).astype(int), 0

    cleaned = series.astype(str).str.strip().str.lower()
    positive_labels = {
        "yes",
        "y",
        "true",
        "1",
        "1.0",
        "churn",
        "churned",
        "left",
        "exited",
        "cancelled",
        "canceled",
        "positive",
        "attrition",
        "attrited",
        "attrited customer",
        "lost",
        "closed",
        "terminated",
        "non-renewed",
        "did churn",
        "t",
    }
    negative_labels = {
        "no",
        "n",
        "false",
        "0",
        "0.0",
        "stay",
        "stayed",
        "active",
        "retained",
        "negative",
        "current",
        "joined",
        "existing customer",
        "existing",
        "renewed",
        "open",
        "f",
    }

    is_pos = cleaned.isin(positive_labels)
    is_neg = cleaned.isin(negative_labels)
    invalid_mask = ~(is_pos | is_neg)
    invalid_count = int(invalid_mask.sum())

    res = np.where(is_pos, 1, 0)
    return pd.Series(res, index=series.index, dtype=int), invalid_count


def _looks_numeric(series: pd.Series) -> bool:
    """Check if series consists predominantly of numeric values."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    converted = pd.to_numeric(
        sample.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(),
        errors="coerce",
    )
    return float(converted.notna().mean()) >= 0.80


def _looks_like_date(col_name: str, series: pd.Series) -> bool:
    """Detect if column represents date/timestamp."""
    norm_name = _norm(col_name)
    if any(k in norm_name for k in ("date", "time", "timestamp", "created", "signup", "joined")):
        return True
    return False


def detect_schema(df: pd.DataFrame, filename: str = "", is_scoring_only: bool = False) -> Schema:
    """Infer dataset schema with column categorization, target, ID, and feature detection."""
    columns = [str(c).strip() for c in df.columns]

    # Target Detection
    target = None
    if not is_scoring_only:
        # Pass 1: Exact normalized alias match
        for col in columns:
            if _norm(col) in TARGET_ALIASES:
                target = col
                break

        # Pass 2: Substring matching for churn/attrition/exited keywords (excluding leakage columns)
        if target is None:
            for col in columns:
                norm = _norm(col)
                if any(x in norm for x in ("date", "reason", "time", "day", "month", "year", "timestamp", "code", "notes", "comment")):
                    continue
                if any(kw in norm for kw in ("churn", "attrit", "exited", "canceld", "cancell")):
                    target = col
                    break

        # Pass 3: Heuristic value-based inspection on binary/2-class columns
        if target is None:
            for col in columns:
                series = df[col].dropna()
                if series.empty:
                    continue
                unique_vals = set(series.astype(str).str.strip().str.lower().unique())
                if 1 <= len(unique_vals) <= 3:
                    pos_match = unique_vals & {
                        "yes", "y", "true", "1", "1.0", "churn", "churned", "attrited",
                        "attrited customer", "exited", "cancelled", "canceled",
                        "left", "lost", "closed", "terminated", "t"
                    }
                    neg_match = unique_vals & {
                        "no", "n", "false", "0", "0.0", "stay", "stayed", "active",
                        "retained", "existing customer", "current", "joined",
                        "renewed", "open", "f"
                    }
                    if pos_match and (neg_match or len(unique_vals) == 1):
                        norm = _norm(col)
                        if not any(g in norm for g in ("gender", "sex", "partner", "depend", "phone", "bill", "senior", "tech", "stream", "backup", "protect", "device", "paper")):
                            target = col
                            break

    # ID Column Detection
    id_column = None
    for col in columns:
        if _norm(col) in ID_ALIASES:
            id_column = col
            break

    if id_column is None:
        # Check for high uniqueness string column with 'id' or 'code'
        for col in columns:
            if col == target:
                continue
            if df[col].nunique(dropna=True) == len(df) and len(df) > 10:
                if any(x in col.lower() for x in ("id", "code", "num", "key", "account")):
                    id_column = col
                    break

    skip_set = set()
    if target:
        skip_set.add(target)
    if id_column:
        skip_set.add(id_column)

    numeric: list[str] = []
    categorical: list[str] = []
    date_columns: list[str] = []
    constant_columns: list[str] = []
    leakage_columns: list[str] = []

    for col in columns:
        if col in skip_set:
            continue

        norm_col = _norm(col)
        # Check leakage keywords
        if any(leak_kw in norm_col for leak_kw in LEAKAGE_KEYWORDS):
            leakage_columns.append(col)
            continue

        series = df[col]
        nunique = int(series.nunique(dropna=True))

        if nunique <= 1:
            constant_columns.append(col)
            continue

        if _looks_like_date(col, series):
            date_columns.append(col)
            continue

        if _looks_numeric(series):
            numeric.append(col)
        else:
            categorical.append(col)

    features = numeric + categorical

    return Schema(
        target=target or "",
        id_column=id_column,
        numeric=numeric,
        categorical=categorical,
        date_columns=date_columns,
        all_features=features,
        leakage_columns=leakage_columns,
        constant_columns=constant_columns,
        filename=filename,
        rows=len(df),
        columns=columns,
    )


def sanitize_dataframe(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Clean and cast dataframe columns according to inferred schema."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    for col in schema.numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
                .replace(["", "nan", "none", "null"], np.nan),
                errors="coerce",
            )

    for col in schema.categorical:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().replace(["nan", "none", "null"], np.nan)

    if schema.target and schema.target in out.columns:
        norm_y, _ = normalize_target(out[schema.target])
        out["_y"] = norm_y

    return out


def validate_dataset(df: pd.DataFrame, filename: str = "", is_training: bool = True) -> dict[str, Any]:
    """Execute complete validation suite and generate a detailed report."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        return {
            "status": "error",
            "errors": ["The uploaded file contains no data rows."],
            "warnings": [],
            "checks": [{"name": "File Content", "status": "fail", "message": "File is empty"}],
            "schema": None,
        }

    schema = detect_schema(df, filename=filename, is_scoring_only=not is_training)
    clean_df = sanitize_dataframe(df, schema)

    # 1. Target Column Check
    if is_training:
        if not schema.target:
            detected_cols_sample = ", ".join(list(df.columns)[:10])
            if len(df.columns) > 10:
                detected_cols_sample += f", ... (+{len(df.columns) - 10} more)"

            errors.append(
                f"No churn target column detected. Detected columns: [{detected_cols_sample}]. "
                "Please ensure your file has a binary churn outcome column (e.g. 'Churn', 'Exited', 'Attrition', 'Status') "
                "with Yes/No or 1/0 values. "
                "Tip: If this file is prospective data meant for scoring without churn labels, "
                "use the 'Risk Scoring' -> 'Batch Scoring' tab instead."
            )
            checks.append({
                "name": "Target Column",
                "status": "fail",
                "message": f"Missing churn outcome column (Detected columns: [{detected_cols_sample}])",
            })
        else:
            y = clean_df["_y"]
            unique_classes = y.unique()
            if len(unique_classes) < 2:
                errors.append(
                    f"Target column '{schema.target}' contains only 1 class ({unique_classes[0]}). Both churned and retained customers are required."
                )
                checks.append({
                    "name": "Target Variance",
                    "status": "fail",
                    "message": "Target column has only one class",
                })
            else:
                churn_count = int(y.sum())
                retained_count = int((y == 0).sum())
                churn_rate = float(y.mean())
                checks.append({
                    "name": "Target Column",
                    "status": "pass",
                    "message": f"Found '{schema.target}' · {churn_rate:.1%} churn rate ({churn_count:,} churned, {retained_count:,} retained)",
                })

                # Class Imbalance Check
                if churn_rate < 0.05 or churn_rate > 0.95:
                    warnings.append(
                        f"Severe class imbalance detected: churn rate is {churn_rate:.1%}. Stratified sampling and cost-sensitive weighting will be applied."
                    )
                    checks.append({
                        "name": "Class Balance",
                        "status": "warning",
                        "message": f"High imbalance ({churn_rate:.1%} churn rate)",
                    })
                else:
                    checks.append({
                        "name": "Class Balance",
                        "status": "pass",
                        "message": f"Healthy distribution ({churn_rate:.1%} churn rate)",
                    })

    # 2. Row Count Verification
    row_count = len(df)
    if is_training and row_count < MIN_TRAIN_ROWS:
        errors.append(f"Insufficient observations: dataset has {row_count} rows. At least {MIN_TRAIN_ROWS} rows are required to train a reliable model.")
        checks.append({
            "name": "Sample Size",
            "status": "fail",
            "message": f"{row_count} rows (minimum {MIN_TRAIN_ROWS} required)",
        })
    elif row_count < 100:
        warnings.append(f"Small dataset ({row_count} rows). Cross-validation will be used, but more data improves model stability.")
        checks.append({
            "name": "Sample Size",
            "status": "warning",
            "message": f"{row_count} rows (usable, small sample)",
        })
    else:
        checks.append({
            "name": "Sample Size",
            "status": "pass",
            "message": f"{row_count:,} customer records",
        })

    # 3. Duplicate ID Check
    if schema.id_column and schema.id_column in df.columns:
        dup_count = int(df[schema.id_column].duplicated().sum())
        if dup_count > 0:
            warnings.append(f"Found {dup_count:,} duplicate customer IDs in column '{schema.id_column}'. Deduplication recommended.")
            checks.append({
                "name": "ID Uniqueness",
                "status": "warning",
                "message": f"{dup_count:,} duplicate IDs found in '{schema.id_column}'",
            })
        else:
            checks.append({
                "name": "ID Uniqueness",
                "status": "pass",
                "message": f"Customer ID '{schema.id_column}' is 100% unique",
            })
    else:
        checks.append({
            "name": "ID Uniqueness",
            "status": "pass",
            "message": "No explicit ID column; row indices used",
        })

    # 4. Feature Adequacy
    if not schema.all_features:
        errors.append("No predictor features detected. Ensure your file contains numeric or categorical attributes.")
        checks.append({
            "name": "Feature Count",
            "status": "fail",
            "message": "No usable features found",
        })
    else:
        checks.append({
            "name": "Feature Count",
            "status": "pass",
            "message": f"{len(schema.all_features)} features ({len(schema.numeric)} numeric, {len(schema.categorical)} categorical)",
        })

    # 5. Missing Values Check
    missing_by_col = df[schema.all_features].isna().mean()
    high_missing_cols = missing_by_col[missing_by_col > 0.40].to_dict()
    if high_missing_cols:
        col_list = ", ".join([f"{c} ({pct:.0%})" for c, pct in high_missing_cols.items()])
        warnings.append(f"High missingness (>40%) in columns: {col_list}. Imputation will be applied.")
        checks.append({
            "name": "Missing Values",
            "status": "warning",
            "message": f"High missing values in {len(high_missing_cols)} column(s)",
        })
    else:
        total_missing = int(df[schema.all_features].isna().sum().sum())
        checks.append({
            "name": "Missing Values",
            "status": "pass",
            "message": f"Missing value rate is low ({total_missing:,} missing entries across features)",
        })

    # 6. High Cardinality Categoricals
    high_cardinality = []
    for col in schema.categorical:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 60 and (n_unique / max(1, len(df))) > 0.30:
            high_cardinality.append(f"{col} ({n_unique} distinct)")

    if high_cardinality:
        warnings.append(f"High-cardinality categorical fields: {', '.join(high_cardinality)}. These may cause sparse representations.")
        checks.append({
            "name": "Categorical Cardinality",
            "status": "warning",
            "message": f"{len(high_cardinality)} high-cardinality column(s)",
        })
    else:
        checks.append({
            "name": "Categorical Cardinality",
            "status": "pass",
            "message": "Categorical column cardinality is appropriate",
        })

    # 7. Constant Columns Check
    if schema.constant_columns:
        warnings.append(f"Detected {len(schema.constant_columns)} constant/zero-variance column(s) (e.g. {', '.join(schema.constant_columns[:3])}). These will be excluded from model training.")
        checks.append({
            "name": "Constant Columns",
            "status": "warning",
            "message": f"{len(schema.constant_columns)} constant column(s) excluded",
        })

    # 8. Data Leakage Detection
    if schema.leakage_columns:
        warnings.append(
            f"Suspicious post-churn leakage columns detected: {', '.join(schema.leakage_columns)}. These contain post-event information and are automatically excluded from training."
        )
        checks.append({
            "name": "Data Leakage Prevention",
            "status": "warning",
            "message": f"Excluded {len(schema.leakage_columns)} potential post-churn column(s)",
        })
    else:
        checks.append({
            "name": "Data Leakage Prevention",
            "status": "pass",
            "message": "No obvious post-churn leakage keywords detected",
        })

    # Perfect correlation check for leakage in numeric features
    if is_training and schema.target and "_y" in clean_df.columns:
        y_vec = clean_df["_y"]
        leakage_numeric = []
        for col in schema.numeric:
            series = clean_df[col].dropna()
            if len(series) == len(y_vec) and series.std() > 0:
                corr = abs(float(np.corrcoef(series, y_vec)[0, 1]))
                if corr > 0.98:
                    leakage_numeric.append(f"{col} (corr={corr:.2f})")
        if leakage_numeric:
            warnings.append(
                f"Numeric features with near-perfect target correlation (>0.98): {', '.join(leakage_numeric)}. Verify these are not post-churn artifacts."
            )

    overall_status = "error" if errors else ("warning" if warnings else "valid")

    return {
        "status": overall_status,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "schema": schema.to_dict(),
        "rows": len(df),
        "columns": list(df.columns),
    }


def load_file_bytes(raw: bytes, filename: str, is_training: bool = True) -> tuple[pd.DataFrame, Schema, dict[str, Any]]:
    """Parse raw bytes into DataFrame, validate, and return (clean_df, schema, validation_report)."""
    buffer = io.BytesIO(raw)
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(buffer)
            df = None
            # Search for first sheet with columns and data
            for sheet_name in excel_file.sheet_names:
                try:
                    candidate = excel_file.parse(sheet_name)
                    if not candidate.empty and len(candidate.columns) > 1 and len(candidate) > 0:
                        df = candidate
                        break
                except Exception:
                    continue
            if df is None:
                buffer.seek(0)
                df = pd.read_excel(buffer)

            # Check if header row is offset (e.g. title in row 0, headers in row 1)
            unnamed_count = sum(str(c).startswith("Unnamed:") for c in df.columns)
            if unnamed_count > len(df.columns) * 0.5 and len(df) > 1:
                try:
                    buffer.seek(0)
                    try_df = pd.read_excel(buffer, header=1)
                    if not try_df.empty and sum(str(c).startswith("Unnamed:") for c in try_df.columns) < unnamed_count:
                        df = try_df
                except Exception:
                    pass
        else:
            df = pd.read_csv(buffer)
    except Exception as exc:
        raise ValueError(f"Could not parse '{filename}': {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    # Clean whitespace and cast column names to string
    df.columns = [str(c).strip() for c in df.columns]

    report = validate_dataset(df, filename=filename, is_training=is_training)
    if not report["is_valid"]:
        raise ValueError("; ".join(report["errors"]))

    schema = detect_schema(df, filename=filename, is_scoring_only=not is_training)
    clean_df = sanitize_dataframe(df, schema)
    return clean_df, schema, report
