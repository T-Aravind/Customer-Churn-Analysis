"""Streamlit entrypoint for Streamlit Community Cloud and local dashboard runs."""

import sys
from pathlib import Path

# Add project root and src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
