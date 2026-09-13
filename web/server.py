"""RetainIQ Enterprise FastAPI Application."""

from __future__ import annotations

import io
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response as RawResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    SAMPLE_PATH,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    logger,
)
from engine import get_workspace  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

app = FastAPI(title="RetainIQ", version="2.0.0", docs_url="/api/docs", redoc_url=None)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    """Ensure dynamic responses and static files are never cached by client browsers."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _sanitize_filename(name: str) -> str:
    """Sanitize uploaded file name to avoid directory traversal and unsafe characters."""
    base = Path(name).name
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return cleaned or "uploaded_data.csv"


def _session_id(response: Response, session: str | None) -> str:
    """Retrieve or initialize isolated client workspace session."""
    if session and re.match(r"^[a-zA-Z0-9_-]{16,64}$", session):
        return session
    value = uuid.uuid4().hex
    response.set_cookie(
        SESSION_COOKIE_NAME,
        value,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE_SECONDS,
    )
    return value


def _safe_error(exc: Exception) -> HTTPException:
    """Translate internal errors into safe, professional messages for enterprise clients."""
    msg = str(exc)
    logger.error(f"API Error: {msg}")
    if isinstance(exc, HTTPException):
        return exc
    return HTTPException(status_code=400, detail=msg)


@app.get("/")
def index() -> FileResponse:
    """Serve single-page platform interface."""
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Healthcheck endpoint."""
    return {"ok": True, "product": "RetainIQ", "version": "2.0.0"}


@app.get("/api/status")
def status(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Return workspace health, schema, KPIs, thresholds, and model training status."""
    sid = _session_id(response, retainiq_session)
    return get_workspace(sid).status()


@app.post("/api/upload")
async def upload(
    response: Response,
    file: UploadFile = File(...),
    retainiq_session: str | None = Cookie(default=None),
) -> dict[str, Any]:
    """Ingest new customer extract, execute validation suite, and build schema."""
    sid = _session_id(response, retainiq_session)
    clean_name = _sanitize_filename(file.filename or "customers.csv")
    ext = Path(clean_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) spreadsheet.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed upload size ({MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB).",
        )

    try:
        result = get_workspace(sid).ingest(raw, clean_name)
    except Exception as exc:
        raise _safe_error(exc) from exc

    return result


@app.get("/api/overview")
def overview(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Retrieve executive portfolio KPIs and active schema."""
    sid = _session_id(response, retainiq_session)
    workspace = get_workspace(sid)
    try:
        kpis = workspace.get_overview()
        _, schema = workspace.require_data()
        return {"kpis": kpis, "schema": schema.to_dict()}
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.get("/api/analytics")
def analytics(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Retrieve segment lift diagnostics, tenure distributions, and data preview."""
    sid = _session_id(response, retainiq_session)
    try:
        return get_workspace(sid).get_analytics()
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.get("/api/insights")
def insights(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Retrieve model explainability drivers, governance scorecard, and retention playbook."""
    sid = _session_id(response, retainiq_session)
    try:
        return get_workspace(sid).get_insights()
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.post("/api/train")
def train(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Train candidate models with stratified splitting & cross-validation, selecting champion."""
    sid = _session_id(response, retainiq_session)
    try:
        return get_workspace(sid).train()
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.get("/api/fields")
def fields(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Retrieve dynamic form schema for single customer risk desk."""
    sid = _session_id(response, retainiq_session)
    workspace = get_workspace(sid)
    try:
        df, schema = workspace.require_data()
        from engine import field_options

        return field_options(df, schema)
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.post("/api/predict")
async def predict(
    payload: dict[str, Any],
    response: Response,
    retainiq_session: str | None = Cookie(default=None),
) -> dict[str, Any]:
    """Score individual customer record with feature driver attributions and prescribed action."""
    sid = _session_id(response, retainiq_session)
    try:
        return get_workspace(sid).predict_one(payload)
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.post("/api/predict-batch")
async def predict_batch(
    response: Response,
    file: UploadFile = File(...),
    retainiq_session: str | None = Cookie(default=None),
) -> StreamingResponse:
    """Batch score prospective customer file (no Churn column required)."""
    sid = _session_id(response, retainiq_session)
    clean_name = _sanitize_filename(file.filename or "score.csv")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded batch file is empty.")

    try:
        scored = get_workspace(sid).predict_batch(raw, clean_name)
    except Exception as exc:
        raise _safe_error(exc) from exc

    buffer = io.StringIO()
    scored.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=retainiq_scored_customers.csv"},
    )


@app.post("/api/thresholds")
def set_thresholds(
    payload: dict[str, float],
    response: Response,
    retainiq_session: str | None = Cookie(default=None),
) -> dict[str, Any]:
    """Update risk classification thresholds (High, Medium)."""
    sid = _session_id(response, retainiq_session)
    high = float(payload.get("high", 0.60))
    med = float(payload.get("medium", 0.35))
    try:
        return get_workspace(sid).set_thresholds(high, med)
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.get("/api/export/excel")
def export_excel(response: Response, retainiq_session: str | None = Cookie(default=None)) -> RawResponse:
    """Download executive multi-sheet Excel workbook (.xlsx)."""
    sid = _session_id(response, retainiq_session)
    workspace = get_workspace(sid)
    try:
        excel_bytes = workspace.export_excel()
        return RawResponse(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=retainiq_executive_report.xlsx"},
        )
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.get("/api/export/report")
def export_report(response: Response, retainiq_session: str | None = Cookie(default=None)) -> HTMLResponse:
    """View/print standalone Executive Briefing Report formatted for client delivery and PDF export."""
    sid = _session_id(response, retainiq_session)
    workspace = get_workspace(sid)
    try:
        html_content = workspace.export_html_report()
        return HTMLResponse(content=html_content)
    except Exception as exc:
        raise _safe_error(exc) from exc


@app.get("/api/export/book")
def export_book(response: Response, retainiq_session: str | None = Cookie(default=None)) -> StreamingResponse:
    """Download full customer book with enriched model predictions if trained."""
    sid = _session_id(response, retainiq_session)
    workspace = get_workspace(sid)
    try:
        if workspace.scored is not None:
            frame = workspace.scored
        else:
            df, _ = workspace.require_data()
            frame = df.drop(columns=["_y"], errors="ignore")
    except Exception as exc:
        raise _safe_error(exc) from exc

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=retainiq_customer_book.csv"},
    )


@app.get("/api/sample")
def sample() -> FileResponse:
    """Download approved sample Telco customer dataset."""
    return FileResponse(
        SAMPLE_PATH,
        filename="sample_telco_customers.csv",
        media_type="text/csv",
    )


@app.post("/api/reset")
def reset(response: Response, retainiq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    """Wipe active workspace session and clear state."""
    sid = _session_id(response, retainiq_session)
    return get_workspace(sid).reset()
