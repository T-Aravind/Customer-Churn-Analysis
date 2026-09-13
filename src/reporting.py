"""Enterprise reporting engine: Multi-sheet Excel workbook and printable Executive HTML report."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from explainability import compute_global_feature_importance
from playbook import generate_retention_playbook
from validation import Schema


def generate_excel_executive_workbook(
    df: pd.DataFrame,
    schema: Schema,
    metrics: dict[str, Any] | None = None,
    scored_df: pd.DataFrame | None = None,
    pipeline: Any = None,
) -> bytes:
    """Generate a multi-tab styled executive Excel workbook."""
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    navy_fill = PatternFill(start_color="0B1F3A", end_color="0B1F3A", fill_type="solid")
    blue_header_fill = PatternFill(start_color="1D4ED8", end_color="1D4ED8", fill_type="solid")
    gray_sub_fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
    high_risk_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    med_risk_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    low_risk_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")

    font_title = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    font_subtitle = Font(name="Calibri", size=11, italic=True, color="CBD5E1")
    font_section = Font(name="Calibri", size=13, bold=True, color="0B1F3A")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True)
    font_regular = Font(name="Calibri", size=11)
    font_small = Font(name="Calibri", size=9, color="64748B")

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    y = df["_y"] if "_y" in df.columns else df[schema.target]
    churn_rate = float(y.mean())
    total_customers = len(df)
    churn_count = int(y.sum())
    retained_count = int((y == 0).sum())

    monthly_col = next((c for c in schema.numeric if "month" in c.lower() or "charge" in c.lower()), None)
    tenure_col = next((c for c in schema.numeric if "tenure" in c.lower()), None)

    monthly_at_risk = float(df.loc[y == 1, monthly_col].fillna(0).sum()) if monthly_col else 0.0
    annual_at_risk = monthly_at_risk * 12.0
    avg_tenure = float(df[tenure_col].mean()) if tenure_col and df[tenure_col].notna().any() else 0.0
    avg_monthly = float(df[monthly_col].mean()) if monthly_col and df[monthly_col].notna().any() else 0.0

    # -------------------------------------------------------------
    # Sheet 1: Executive Summary
    # -------------------------------------------------------------
    ws1 = wb.create_sheet(title="Executive Summary")
    ws1.views.sheetView[0].showGridLines = True

    # Title Banner
    ws1.merge_cells("A1:G2")
    ws1["A1"] = "RetainIQ Executive Churn & Revenue Risk Report"
    ws1["A1"].font = font_title
    ws1["A1"].fill = navy_fill
    ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")

    ws1.merge_cells("A3:G3")
    ws1["A3"] = f"Generated: {datetime.now().strftime('%B %d, %Y')} | Confidential Portfolio Intelligence"
    ws1["A3"].font = font_subtitle
    ws1["A3"].fill = navy_fill
    ws1["A3"].alignment = Alignment(horizontal="center", vertical="center")

    # Section 1: KPIs
    ws1["A5"] = "Portfolio Health Indicators"
    ws1["A5"].font = font_section

    kpi_data = [
        ("Total Active Portfolio Accounts", f"{total_customers:,}", "Total customer base in active extract"),
        ("Identified Churned Customers", f"{churn_count:,}", "Historical churn count in baseline"),
        ("Retained Active Customers", f"{retained_count:,}", "Loyal customer cohort"),
        ("Portfolio Churn Rate", f"{churn_rate:.2%}", "Baseline attrition rate"),
        ("Monthly Revenue at Risk", f"₹{monthly_at_risk:,.2f}" if monthly_at_risk > 0 else "N/A", "Direct recurring monthly spend at risk (INR)"),
        ("Annualized Revenue at Risk", f"₹{annual_at_risk:,.2f}" if annual_at_risk > 0 else "N/A", "Projected 12-month revenue exposure (INR)"),
        ("Average Account Tenure", f"{avg_tenure:.1f} months" if avg_tenure > 0 else "N/A", "Mean customer lifetime duration"),
        ("Average Monthly Spend", f"₹{avg_monthly:.2f}" if avg_monthly > 0 else "N/A", "Mean customer monthly recurring spend (INR)"),
    ]

    ws1.cell(row=6, column=1, value="Key Performance Indicator").font = font_header
    ws1.cell(row=6, column=1).fill = blue_header_fill
    ws1.cell(row=6, column=2, value="Observed Value").font = font_header
    ws1.cell(row=6, column=2).fill = blue_header_fill
    ws1.cell(row=6, column=3, value="Operational Definition").font = font_header
    ws1.cell(row=6, column=3).fill = blue_header_fill

    for r_idx, (kpi, val, desc) in enumerate(kpi_data, start=7):
        c1 = ws1.cell(row=r_idx, column=1, value=kpi)
        c2 = ws1.cell(row=r_idx, column=2, value=val)
        c3 = ws1.cell(row=r_idx, column=3, value=desc)
        for c in (c1, c2, c3):
            c.font = font_regular
            c.border = thin_border
        c2.font = font_bold

    # Dataset Profile Section
    start_row = 17
    ws1.cell(row=start_row, column=1, value="Dataset Profile & Intake Metadata").font = font_section

    profile_data = [
        ("Source File", schema.filename or "Active Extract"),
        ("Total Records", f"{schema.rows:,} rows"),
        ("Predictor Features", f"{len(schema.all_features)} features ({len(schema.numeric)} numeric, {len(schema.categorical)} categorical)"),
        ("Target Column", schema.target),
        ("Identifier Column", schema.id_column or "Row Index"),
        ("Data Leakage Columns Excluded", f"{len(schema.leakage_columns)} columns" if schema.leakage_columns else "None detected (Clean)"),
    ]

    for r_idx, (label, val) in enumerate(profile_data, start=start_row + 1):
        c1 = ws1.cell(row=r_idx, column=1, value=label)
        c2 = ws1.cell(row=r_idx, column=2, value=val)
        for c in (c1, c2):
            c.font = font_regular
            c.border = thin_border
        c1.font = font_bold

    # -------------------------------------------------------------
    # Sheet 2: Model Governance & Performance
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Model Governance")
    ws2.views.sheetView[0].showGridLines = True

    ws2.cell(row=1, column=1, value="Model Governance & Benchmark Comparison").font = font_section

    if metrics:
        ws2.cell(row=3, column=1, value="Champion Model:").font = font_bold
        ws2.cell(row=3, column=2, value=metrics.get("best_model_display", metrics.get("best_model", ""))).font = font_bold
        ws2.cell(row=4, column=1, value="Model Version:").font = font_bold
        ws2.cell(row=4, column=2, value=metrics.get("model_version", "v1.0"))
        ws2.cell(row=5, column=1, value="Training Timestamp:").font = font_bold
        ws2.cell(row=5, column=2, value=metrics.get("trained_at_formatted", metrics.get("trained_at_iso", "")))
        ws2.cell(row=6, column=1, value="Selection Strategy:").font = font_bold
        ws2.cell(row=6, column=2, value=metrics.get("selection_metric", "PR-AUC & ROC-AUC Composite"))

        # Model comparison table
        ws2.cell(row=8, column=1, value="Algorithm").font = font_header
        ws2.cell(row=8, column=1).fill = blue_header_fill
        headers = ["Status", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1 Score", "5-Fold CV (PR-AUC)"]
        for c_idx, h in enumerate(headers, start=2):
            cell = ws2.cell(row=8, column=c_idx, value=h)
            cell.font = font_header
            cell.fill = blue_header_fill

        for r_idx, (m_name, m_data) in enumerate(metrics.get("models", {}).items(), start=9):
            ws2.cell(row=r_idx, column=1, value=m_name.replace("_", " ").title()).font = font_bold
            ws2.cell(row=r_idx, column=2, value=m_data.get("training_status", "Completed"))
            ws2.cell(row=r_idx, column=3, value=f"{m_data.get('roc_auc', 0):.4f}")
            ws2.cell(row=r_idx, column=4, value=f"{m_data.get('pr_auc', 0):.4f}")
            ws2.cell(row=r_idx, column=5, value=f"{m_data.get('accuracy', 0):.2%}")
            ws2.cell(row=r_idx, column=6, value=f"{m_data.get('precision', 0):.4f}")
            ws2.cell(row=r_idx, column=7, value=f"{m_data.get('recall', 0):.4f}")
            ws2.cell(row=r_idx, column=8, value=f"{m_data.get('f1', 0):.4f}")
            cv_mean = m_data.get("cv_pr_auc_mean", 0)
            cv_std = m_data.get("cv_pr_auc_std", 0)
            ws2.cell(row=r_idx, column=9, value=f"{cv_mean:.4f} ± {cv_std:.4f}")

            for c in range(1, 10):
                ws2.cell(row=r_idx, column=c).border = thin_border
                ws2.cell(row=r_idx, column=c).font = font_regular
            ws2.cell(row=r_idx, column=1).font = font_bold

            if m_name == metrics.get("best_model"):
                for c in range(1, 10):
                    ws2.cell(row=r_idx, column=c).fill = low_risk_fill
    else:
        ws2.cell(row=3, column=1, value="No model trained yet in this workspace.").font = font_regular

    # -------------------------------------------------------------
    # Sheet 3: Segment Risk Analysis
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Segment Risk Matrix")
    ws3.views.sheetView[0].showGridLines = True

    ws3.cell(row=1, column=1, value="Segment Risk & Churn Lift Matrix").font = font_section

    seg_headers = ["Categorical Feature", "Sub-Segment", "Total Accounts", "Churners", "Segment Churn Rate", "Lift vs Baseline"]
    for c_idx, h in enumerate(seg_headers, start=1):
        cell = ws3.cell(row=3, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = blue_header_fill

    current_r = 4
    for col in schema.categorical[:12]:
        grouped = (
            df.assign(_y=y)
            .groupby(col, observed=False)
            .agg(customers=("_y", "size"), churners=("_y", "sum"), rate=("_y", "mean"))
            .reset_index()
            .sort_values("rate", ascending=False)
        )
        for _, row in grouped.iterrows():
            lift = float(row["rate"]) - churn_rate
            c1 = ws3.cell(row=current_r, column=1, value=col)
            c2 = ws3.cell(row=current_r, column=2, value=str(row[col]))
            c3 = ws3.cell(row=current_r, column=3, value=int(row["customers"]))
            c4 = ws3.cell(row=current_r, column=4, value=int(row["churners"]))
            c5 = ws3.cell(row=current_r, column=5, value=f"{float(row['rate']):.2%}")
            c6 = ws3.cell(row=current_r, column=6, value=f"{lift:+.2%}")

            for c in (c1, c2, c3, c4, c5, c6):
                c.font = font_regular
                c.border = thin_border

            if lift > 0.10:
                c5.fill = high_risk_fill
                c6.fill = high_risk_fill
            elif lift > 0.02:
                c5.fill = med_risk_fill
                c6.fill = med_risk_fill

            current_r += 1

    # -------------------------------------------------------------
    # Sheet 4: Retention Playbook
    # -------------------------------------------------------------
    ws4 = wb.create_sheet(title="Retention Playbook")
    ws4.views.sheetView[0].showGridLines = True

    ws4.cell(row=1, column=1, value="Evidence-Grounded Commercial Retention Playbook").font = font_section

    pb = generate_retention_playbook(df, schema, metrics)
    pb_headers = ["Priority Horizon", "Strategic Pillar", "Target Risk Cohort", "Observed Churn Metric", "Annual Revenue Impact", "Recommended Business Actions"]
    for c_idx, h in enumerate(pb_headers, start=1):
        cell = ws4.cell(row=3, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = blue_header_fill

    for r_idx, pillar in enumerate(pb.get("strategic_pillars", []), start=4):
        c1 = ws4.cell(row=r_idx, column=1, value=pillar["priority"])
        c2 = ws4.cell(row=r_idx, column=2, value=pillar["pillar_name"])
        c3 = ws4.cell(row=r_idx, column=3, value=pillar["target_segment"])
        c4 = ws4.cell(row=r_idx, column=4, value=pillar["observed_metric"])
        c5 = ws4.cell(row=r_idx, column=5, value=pillar["annual_revenue_at_risk"])
        c6 = ws4.cell(row=r_idx, column=6, value=" | ".join(pillar["recommended_actions"]))

        for c in (c1, c2, c3, c4, c5, c6):
            c.font = font_regular
            c.border = thin_border
        c1.font = font_bold
        c2.font = font_bold

    # -------------------------------------------------------------
    # Sheet 5: Top At-Risk Scored Accounts Sample
    # -------------------------------------------------------------
    if scored_df is not None and not scored_df.empty:
        ws5 = wb.create_sheet(title="Scored Accounts Sample")
        ws5.views.sheetView[0].showGridLines = True

        ws5.cell(row=1, column=1, value="Scored Accounts Portfolio Extract (Top 250 Risk Records)").font = font_section

        score_cols = [c for c in scored_df.columns if c in [
            schema.id_column or "customerID",
            "churn_probability",
            "risk_band",
            "predicted_status",
            "retention_priority",
            "monthly_revenue_at_risk",
            "annual_revenue_at_risk",
            "recommended_action",
        ]] + [c for c in schema.all_features[:6] if c in scored_df.columns]

        # Dedup columns
        score_cols = list(dict.fromkeys(score_cols))

        for c_idx, h in enumerate(score_cols, start=1):
            cell = ws5.cell(row=3, column=c_idx, value=h.replace("_", " ").title())
            cell.font = font_header
            cell.fill = blue_header_fill

        sorted_scored = scored_df.sort_values("churn_probability", ascending=False).head(250)
        for r_idx, (_, row) in enumerate(sorted_scored.iterrows(), start=4):
            for c_idx, col_name in enumerate(score_cols, start=1):
                val = row.get(col_name, "")
                cell = ws5.cell(row=r_idx, column=c_idx, value=str(val) if not isinstance(val, (int, float)) else val)
                cell.font = font_regular
                cell.border = thin_border

                if col_name == "risk_band":
                    if str(val) == "High":
                        cell.fill = high_risk_fill
                        cell.font = font_bold
                    elif str(val) == "Medium":
                        cell.fill = med_risk_fill
                    elif str(val) == "Low":
                        cell.fill = low_risk_fill

    # Auto-adjust column widths across all worksheets
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.value:
                    val_str = str(cell.value)
                    # Ignore merged long banner cells
                    if len(val_str) < 60:
                        max_len = max(max_len, len(val_str))
            sheet.column_dimensions[col_letter].width = max(12, min(max_len + 3, 40))

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def generate_html_executive_report(
    df: pd.DataFrame,
    schema: Schema,
    metrics: dict[str, Any] | None = None,
    global_drivers: list[dict[str, Any]] | None = None,
) -> str:
    """Render a standalone, printable Executive Briefing Report formatted for client sign-off and PDF print."""
    y = df["_y"] if "_y" in df.columns else df[schema.target]
    churn_rate = float(y.mean())
    total_customers = len(df)
    churn_count = int(y.sum())
    retained_count = int((y == 0).sum())

    monthly_col = next((c for c in schema.numeric if "month" in c.lower() or "charge" in c.lower()), None)
    tenure_col = next((c for c in schema.numeric if "tenure" in c.lower()), None)

    monthly_at_risk = float(df.loc[y == 1, monthly_col].fillna(0).sum()) if monthly_col else 0.0
    annual_at_risk = monthly_at_risk * 12.0
    avg_tenure = float(df[tenure_col].mean()) if tenure_col and df[tenure_col].notna().any() else 0.0

    pb = generate_retention_playbook(df, schema, metrics, global_drivers)
    current_date = datetime.now().strftime("%B %d, %Y")

    # Driver rows
    driver_html = ""
    if global_drivers:
        for d in global_drivers[:6]:
            driver_html += f"""
            <tr>
              <td><strong>#{d['rank']} {d['feature']}</strong></td>
              <td><span class="badge">{d['feature_type']}</span></td>
              <td><strong>{d['importance_pct']}%</strong></td>
              <td class="muted">{d['description']}</td>
            </tr>"""
    else:
        driver_html = "<tr><td colspan='4' class='muted'>Model drivers will populate upon training.</td></tr>"

    # Model rows
    model_html = ""
    if metrics:
        for m_name, m in metrics.get("models", {}).items():
            is_best = m_name == metrics.get("best_model")
            badge = "<span class='badge best'>Champion</span>" if is_best else "<span class='badge'>Candidate</span>"
            model_html += f"""
            <tr class="{'champion-row' if is_best else ''}">
              <td><strong>{m_name.replace('_', ' ').title()}</strong> {badge}</td>
              <td>{m.get('roc_auc', 0):.4f}</td>
              <td><strong>{m.get('pr_auc', 0):.4f}</strong></td>
              <td>{m.get('accuracy', 0):.1%}</td>
              <td>{m.get('f1', 0):.4f}</td>
              <td>{m.get('cv_pr_auc_mean', 0):.4f} ± {m.get('cv_pr_auc_std', 0):.4f}</td>
            </tr>"""
    else:
        model_html = "<tr><td colspan='6' class='muted'>No models trained yet.</td></tr>"

    # Playbook rows
    playbook_html = ""
    for p in pb.get("strategic_pillars", []):
        playbook_html += f"""
        <div class="card pillar-card">
          <div class="pillar-header">
            <span class="badge priority">{p['priority']}</span>
            <h3>{p['pillar_name']}</h3>
          </div>
          <p><strong>Target Cohort:</strong> {p['target_segment']} · <em>{p['observed_metric']}</em></p>
          <p class="muted" style="margin: 6px 0 12px;">{p['strategic_rationale']}</p>
          <ul class="action-list">
            {''.join([f'<li>{act}</li>' for act in p['recommended_actions']])}
          </ul>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RetainIQ Executive Churn Report · {current_date}</title>
  <style>
    @page {{ size: A4; margin: 16mm; }}
    @media print {{
      body {{ background: #fff !important; color: #000 !important; font-size: 11pt; }}
      .no-print {{ display: none !important; }}
      .card {{ box-shadow: none !important; border: 1px solid #ccc !important; page-break-inside: avoid; }}
      .pillar-card {{ page-break-inside: avoid; }}
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1e293b; background: #f8fafc; line-height: 1.5; padding: 32px; }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0b1f3a; padding-bottom: 20px; margin-bottom: 24px; }}
    .brand-title {{ font-size: 26px; font-weight: 800; color: #0b1f3a; letter-spacing: -0.02em; }}
    .brand-sub {{ font-size: 13px; color: #64748b; margin-top: 2px; }}
    .report-meta {{ text-align: right; font-size: 12px; color: #475569; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; background: #e2e8f0; color: #334155; }}
    .badge.best {{ background: #dcfce7; color: #166534; }}
    .badge.priority {{ background: #fee2e2; color: #991b1b; }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 24px; }}
    .kpi-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700; }}
    .kpi-val {{ font-size: 24px; font-weight: 800; color: #0b1f3a; margin-top: 6px; }}
    h2 {{ font-size: 18px; font-weight: 800; color: #0b1f3a; margin-bottom: 14px; }}
    h3 {{ font-size: 15px; font-weight: 700; color: #0f172a; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f1f5f9; color: #475569; font-weight: 700; text-transform: uppercase; font-size: 11px; }}
    .champion-row {{ background: #f0fdf4; }}
    .muted {{ color: #64748b; font-size: 12px; }}
    .pillar-card {{ margin-bottom: 16px; }}
    .pillar-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .action-list {{ padding-left: 20px; margin-top: 8px; font-size: 13px; color: #334155; }}
    .action-list li {{ margin-bottom: 4px; }}
    .actions-bar {{ display: flex; gap: 12px; justify-content: flex-end; margin-bottom: 20px; }}
    .btn {{ padding: 8px 16px; border-radius: 8px; font-weight: 700; font-size: 13px; cursor: pointer; text-decoration: none; border: 1px solid #cbd5e1; background: #fff; color: #0f172a; }}
    .btn-primary {{ background: #1d4ed8; color: #fff; border-color: #1d4ed8; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="actions-bar no-print">
      <button class="btn btn-primary" onclick="window.print()">🖨️ Print / Save as PDF</button>
      <a class="btn" href="/api/export/excel">📊 Download Excel Workbook</a>
    </div>

    <header class="header">
      <div>
        <div class="brand-title">RetainIQ Executive Churn Intelligence Briefing</div>
        <div class="brand-sub">Portfolio Diagnostic, Predictive Risk Attribution &amp; Strategic Playbook</div>
      </div>
      <div class="report-meta">
        <p><strong>Date:</strong> {current_date}</p>
        <p><strong>Source File:</strong> {schema.filename or 'Active Extract'}</p>
        <p><strong>Classification:</strong> Confidential · Client Delivery</p>
      </div>
    </header>

    <div class="grid-4">
      <div class="card" style="margin-bottom:0">
        <div class="kpi-title">Active Portfolio</div>
        <div class="kpi-val">{total_customers:,}</div>
        <p class="muted">{retained_count:,} active / {churn_count:,} churned</p>
      </div>
      <div class="card" style="margin-bottom:0">
        <div class="kpi-title">Baseline Churn Rate</div>
        <div class="kpi-val">{churn_rate:.1%}</div>
        <p class="muted">Historical benchmark</p>
      </div>
      <div class="card" style="margin-bottom:0">
        <div class="kpi-title">Monthly Rev at Risk</div>
        <div class="kpi-val">{f"₹{monthly_at_risk:,.2f}" if monthly_at_risk > 0 else "—"}</div>
        <p class="muted">Direct recurring exposure (INR)</p>
      </div>
      <div class="card" style="margin-bottom:0">
        <div class="kpi-title">Annualized Rev at Risk</div>
        <div class="kpi-val">{f"₹{annual_at_risk:,.2f}" if annual_at_risk > 0 else "—"}</div>
        <p class="muted">12-month run-rate impact (INR)</p>
      </div>
    </div>

    <section class="card">
      <h2>1. Model Performance &amp; Governance Benchmark</h2>
      <p class="muted" style="margin-bottom:12px;">Models trained with Stratified 5-Fold Cross-Validation on held-out test data. Selected champion optimizes PR-AUC (Precision-Recall trade-off).</p>
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>Accuracy</th>
            <th>F1 Score</th>
            <th>5-Fold CV (PR-AUC)</th>
          </tr>
        </thead>
        <tbody>
          {model_html}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>2. Top Predictive Churn Drivers (Global Attribution)</h2>
      <p class="muted" style="margin-bottom:12px;">Key predictive factors ranked by relative contribution to churn classification.</p>
      <table>
        <thead>
          <tr>
            <th>Predictor</th>
            <th>Type</th>
            <th>Importance Share</th>
            <th>Business Interpretation</th>
          </tr>
        </thead>
        <tbody>
          {driver_html}
        </tbody>
      </table>
    </section>

    <section>
      <h2 style="margin-top: 24px; margin-bottom: 12px;">3. Executive Retention Action Playbook</h2>
      {playbook_html}
    </section>

    <footer style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 11px; color: #64748b; display: flex; justify-content: space-between;">
      <span>RetainIQ Customer Retention Intelligence Platform</span>
      <span>Confidential · For Client Operating Committee</span>
    </footer>
  </div>
</body>
</html>"""
