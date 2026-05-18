"""Streamlit Community Cloud entry point (repo root)."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "dashboard" / "app.py"), run_name="__main__")
