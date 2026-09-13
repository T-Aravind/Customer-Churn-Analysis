"""Workspace session isolation, persistence, and audit logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_MED_RISK_THRESHOLD,
    SAMPLE_PATH,
    UPLOADS_DIR,
    logger,
)
from explainability import compute_global_feature_importance
from models import train_and_compare_models
from playbook import generate_retention_playbook
from reporting import generate_excel_executive_workbook, generate_html_executive_report
from scoring import (
    load_batch_file_and_score,
    score_batch_dataframe,
    score_single_customer,
)
from validation import Schema, load_file_bytes, sanitize_dataframe, validate_dataset


class Workspace:
    """Encapsulates a client workspace session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.folder = UPLOADS_DIR / session_id
        self.folder.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame | None = None
        self.schema: Schema | None = None
        self.validation_report: dict[str, Any] | None = None
        self.model: Any = None
        self.metrics: dict[str, Any] | None = None
        self.scored: pd.DataFrame | None = None
        self.high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD
        self.med_threshold: float = DEFAULT_MED_RISK_THRESHOLD

        self._load()

    def _paths(self) -> dict[str, Path]:
        return {
            "data": self.folder / "customers.parquet",
            "schema": self.folder / "schema.json",
            "validation": self.folder / "validation.json",
            "model": self.folder / "model.joblib",
            "metrics": self.folder / "metrics.json",
            "scored": self.folder / "scored.parquet",
            "thresholds": self.folder / "thresholds.json",
            "audit": self.folder / "audit.log",
        }

    def _audit(self, action: str, details: str = "") -> None:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            log_line = f"{timestamp} | {action} | {details}\n"
            paths = self._paths()
            with open(paths["audit"], "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as exc:
            logger.warning(f"Audit log write failed: {exc}")

    def _load(self) -> None:
        paths = self._paths()
        if paths["data"].exists() and paths["schema"].exists():
            try:
                self.df = pd.read_parquet(paths["data"])
                payload = json.loads(paths["schema"].read_text(encoding="utf-8"))
                self.schema = Schema(**payload)
            except Exception as exc:
                logger.error(f"Failed to load cached dataset for {self.session_id}: {exc}")

        if paths["validation"].exists():
            try:
                self.validation_report = json.loads(paths["validation"].read_text(encoding="utf-8"))
            except Exception:
                pass

        if paths["model"].exists():
            try:
                self.model = joblib.load(paths["model"])
            except Exception as exc:
                logger.error(f"Failed to load model for {self.session_id}: {exc}")

        if paths["metrics"].exists():
            try:
                self.metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            except Exception:
                pass

        if paths["scored"].exists():
            try:
                self.scored = pd.read_parquet(paths["scored"])
            except Exception:
                pass

        if paths["thresholds"].exists():
            try:
                t = json.loads(paths["thresholds"].read_text(encoding="utf-8"))
                self.high_threshold = float(t.get("high", DEFAULT_HIGH_RISK_THRESHOLD))
                self.med_threshold = float(t.get("medium", DEFAULT_MED_RISK_THRESHOLD))
            except Exception:
                pass

    def persist(self) -> None:
        paths = self._paths()
        if self.df is not None:
            self.df.to_parquet(paths["data"], index=False)
        if self.schema is not None:
            paths["schema"].write_text(json.dumps(self.schema.to_dict(), indent=2), encoding="utf-8")
        if self.validation_report is not None:
            paths["validation"].write_text(json.dumps(self.validation_report, indent=2), encoding="utf-8")
        if self.model is not None:
            joblib.dump(self.model, paths["model"])
        if self.metrics is not None:
            paths["metrics"].write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")
        if self.scored is not None:
            self.scored.to_parquet(paths["scored"], index=False)
        paths["thresholds"].write_text(
            json.dumps({"high": self.high_threshold, "medium": self.med_threshold}, indent=2),
            encoding="utf-8",
        )

    def ingest(self, raw: bytes, filename: str) -> dict[str, Any]:
        """Ingest new client customer extract and run full validation."""
        clean_df, schema, report = load_file_bytes(raw, filename=filename, is_training=True)

        self.df = clean_df
        self.schema = schema
        self.validation_report = report
        self.model = None
        self.metrics = None
        self.scored = None

        # Clean old model artifacts
        for name in ("model.joblib", "metrics.json", "scored.parquet"):
            p = self.folder / name
            if p.exists():
                p.unlink()

        self.persist()
        self._audit("DATA_INGEST", f"File: {filename} ({len(clean_df)} rows)")

        return {
            "validation": report,
            "schema": schema.to_dict(),
            "kpis": self.get_overview(),
        }

    def require_data(self) -> tuple[pd.DataFrame, Schema]:
        if self.df is None or self.schema is None:
            raise ValueError("Upload a customer file before accessing this section.")
        return self.df, self.schema

    def require_model(self) -> tuple[Any, Schema]:
        if self.model is None or self.schema is None:
            raise ValueError("Train a model before generating churn predictions or risk scoring.")
        return self.model, self.schema

    def set_thresholds(self, high: float, med: float) -> dict[str, Any]:
        """Update risk band cutoffs and refresh scored dataframe if available."""
        if not (0.0 < med < high <= 1.0):
            raise ValueError("Invalid thresholds. Require 0 < Medium < High <= 1.0")
        self.high_threshold = round(high, 2)
        self.med_threshold = round(med, 2)

        if self.model is not None and self.df is not None and self.schema is not None:
            self.scored = score_batch_dataframe(
                self.model,
                self.df,
                self.schema,
                high_threshold=self.high_threshold,
                med_threshold=self.med_threshold,
            )

        self.persist()
        self._audit("THRESHOLD_UPDATE", f"High: {self.high_threshold}, Med: {self.med_threshold}")
        return {"high": self.high_threshold, "medium": self.med_threshold}

    def get_overview(self) -> dict[str, Any]:
        df, schema = self.require_data()
        y = df["_y"]
        churn_rate = float(y.mean())
        churners = int(y.sum())
        retained = int((y == 0).sum())

        monthly_col = next((c for c in schema.numeric if "month" in c.lower() or "charge" in c.lower()), None)
        tenure_col = next((c for c in schema.numeric if "tenure" in c.lower()), None)

        monthly_rev_at_risk = 0.0
        if monthly_col:
            monthly_rev_at_risk = float(df.loc[y == 1, monthly_col].fillna(0).sum())

        avg_tenure = float(df[tenure_col].mean()) if tenure_col and df[tenure_col].notna().any() else None
        avg_monthly = float(df[monthly_col].mean()) if monthly_col and df[monthly_col].notna().any() else None

        return {
            "customers": int(len(df)),
            "churners": churners,
            "retained": retained,
            "churn_rate": round(churn_rate, 4),
            "monthly_revenue_at_risk": round(monthly_rev_at_risk, 2),
            "annual_revenue_at_risk": round(monthly_rev_at_risk * 12, 2),
            "avg_tenure": round(avg_tenure, 1) if avg_tenure else None,
            "avg_monthly": round(avg_monthly, 2) if avg_monthly else None,
            "monthly_column": monthly_col,
            "tenure_column": tenure_col,
        }

    def get_analytics(self) -> dict[str, Any]:
        df, schema = self.require_data()
        y = df["_y"]
        portfolio_churn = float(y.mean())

        churn_mix = [
            {"label": "Active (Retained)", "value": int((y == 0).sum())},
            {"label": "Churned", "value": int(y.sum())},
        ]

        segments = []
        for col in schema.categorical[:12]:
            grouped = (
                df.assign(_y=y)
                .groupby(col, observed=False)
                .agg(customers=("_y", "size"), churners=("_y", "sum"), rate=("_y", "mean"))
                .reset_index()
            )
            grouped["lift"] = grouped["rate"] - portfolio_churn
            grouped = grouped.sort_values("rate", ascending=False)

            rows = []
            for _, r in grouped.iterrows():
                rows.append({
                    "segment": str(r[col]),
                    "customers": int(r["customers"]),
                    "churners": int(r["churners"]),
                    "churn_rate": round(float(r["rate"]), 4),
                    "lift": round(float(r["lift"]), 4),
                    "is_high_risk": bool(r["lift"] > 0.05 and r["customers"] >= 15),
                })

            segments.append({
                "column": col,
                "rows": rows,
            })

        # Tenure bins
        tenure_col = next((c for c in schema.numeric if "tenure" in c.lower()), None)
        tenure_bins = []
        if tenure_col:
            buckets = pd.cut(
                df[tenure_col].fillna(0),
                bins=[-0.1, 6, 12, 24, 36, 48, 10_000],
                labels=["0–6m", "7–12m", "13–24m", "25–36m", "37–48m", "49m+"],
            )
            b_df = df.assign(bucket=buckets, _y=y).groupby("bucket", observed=False)["_y"].agg(["size", "mean"])
            for idx, r in b_df.iterrows():
                tenure_bins.append({
                    "bucket": str(idx),
                    "customers": int(r["size"]),
                    "churn_rate": round(float(r["mean"]), 4),
                })

        # Numeric profiles
        numeric_profile = []
        for col in schema.numeric[:8]:
            numeric_profile.append({
                "column": col,
                "retained_mean": round(float(df.loc[y == 0, col].mean()), 2) if df.loc[y == 0, col].notna().any() else None,
                "churned_mean": round(float(df.loc[y == 1, col].mean()), 2) if df.loc[y == 1, col].notna().any() else None,
            })

        preview_cols = [c for c in df.columns if c != "_y"][:14]

        return {
            "churn_mix": churn_mix,
            "segments": segments,
            "tenure_bins": tenure_bins,
            "numeric_profile": numeric_profile,
            "preview_columns": preview_cols,
            "preview": df.head(15).fillna("").astype(str).to_dict(orient="records"),
        }

    def train(self) -> dict[str, Any]:
        """Train models, evaluate on stratified cross-validation and test holdout, select champion."""
        df, schema = self.require_data()
        pipeline, summary = train_and_compare_models(df, schema)

        self.model = pipeline
        self.metrics = summary
        self.scored = score_batch_dataframe(
            pipeline,
            df,
            schema,
            high_threshold=self.high_threshold,
            med_threshold=self.med_threshold,
        )

        self.persist()
        self._audit("MODEL_TRAIN", f"Champion: {summary['best_model']} (PR-AUC: {summary['models'][summary['best_model']]['pr_auc']})")
        return summary

    def get_insights(self) -> dict[str, Any]:
        """Return global feature drivers, playbook, and governance summary."""
        df, schema = self.require_data()
        global_drivers = []
        if self.model is not None:
            global_drivers = compute_global_feature_importance(self.model, schema)

        playbook = generate_retention_playbook(
            df,
            schema,
            metrics=self.metrics,
            global_drivers=global_drivers,
        )

        return {
            "kpis": self.get_overview(),
            "global_drivers": global_drivers,
            "playbook": playbook,
            "model_governance": self.metrics,
        }

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score single customer with full local feature attribution and action recommendation."""
        pipeline, schema = self.require_model()
        return score_single_customer(
            pipeline,
            payload,
            schema,
            high_threshold=self.high_threshold,
            med_threshold=self.med_threshold,
        )

    def predict_batch(self, raw: bytes, filename: str) -> pd.DataFrame:
        """Batch score prospective customer file (no Churn column required)."""
        pipeline, schema = self.require_model()
        scored_batch = load_batch_file_and_score(
            pipeline,
            raw,
            filename,
            schema,
            high_threshold=self.high_threshold,
            med_threshold=self.med_threshold,
        )
        self._audit("BATCH_SCORE", f"File: {filename} ({len(scored_batch)} scored records)")
        return scored_batch

    def export_excel(self) -> bytes:
        """Generate formatted multi-sheet executive Excel workbook."""
        df, schema = self.require_data()
        return generate_excel_executive_workbook(
            df,
            schema,
            metrics=self.metrics,
            scored_df=self.scored,
            pipeline=self.model,
        )

    def export_html_report(self) -> str:
        """Render printable Executive HTML report."""
        df, schema = self.require_data()
        global_drivers = compute_global_feature_importance(self.model, schema) if self.model else []
        return generate_html_executive_report(
            df,
            schema,
            metrics=self.metrics,
            global_drivers=global_drivers,
        )

    def reset(self) -> dict[str, Any]:
        """Wipe active workspace session files."""
        for path in self.folder.glob("*"):
            if path.is_file():
                try:
                    path.unlink()
                except Exception:
                    pass
        self.df = None
        self.schema = None
        self.validation_report = None
        self.model = None
        self.metrics = None
        self.scored = None
        self._audit("WORKSPACE_RESET", "Full wipe")
        return {"ok": True}

    def status(self) -> dict[str, Any]:
        has_data = self.df is not None
        has_model = self.model is not None
        return {
            "has_data": has_data,
            "has_model": has_model,
            "schema": self.schema.to_dict() if self.schema else None,
            "validation": self.validation_report,
            "kpis": self.get_overview() if has_data else None,
            "metrics": self.metrics,
            "thresholds": {
                "high": self.high_threshold,
                "medium": self.med_threshold,
            },
        }


_workspaces: dict[str, Workspace] = {}


def get_workspace(session_id: str) -> Workspace:
    workspace = _workspaces.get(session_id)
    if workspace is None:
        workspace = Workspace(session_id)
        _workspaces[session_id] = workspace
    return workspace
