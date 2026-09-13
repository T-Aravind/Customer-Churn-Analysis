"""Configuration module for RetainIQ platform."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

# Upload and Security Settings
MAX_UPLOAD_SIZE_MB = int(os.getenv("RETAINIQ_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
SESSION_COOKIE_NAME = "retainiq_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 14  # 14 days

# Default Business Risk Thresholds
DEFAULT_HIGH_RISK_THRESHOLD = float(os.getenv("RETAINIQ_HIGH_THRESHOLD", "0.60"))
DEFAULT_MED_RISK_THRESHOLD = float(os.getenv("RETAINIQ_MED_THRESHOLD", "0.35"))

# Model Training Settings
TEST_SPLIT_RATIO = float(os.getenv("RETAINIQ_TEST_SPLIT", "0.20"))
CV_FOLDS = int(os.getenv("RETAINIQ_CV_FOLDS", "5"))
RANDOM_STATE = 42
MIN_TRAIN_ROWS = 40

# Logging Setup
LOG_LEVEL = os.getenv("RETAINIQ_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retainiq")
