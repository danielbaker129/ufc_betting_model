import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")


def _secret(name: str, default: str = "") -> str:
    """Local .env, then Streamlit Cloud secrets, then process env."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


ODDS_API_KEY   = _secret("ODDS_API_KEY")
API_SPORTS_KEY = _secret("API_SPORTS_KEY")

DB_PATH        = ROOT / os.getenv("DB_PATH", "data/ufc.db")
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = ROOT / "models"

for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCRAPE_DELAY  = float(os.getenv("SCRAPE_DELAY", "2.0"))
USER_AGENT    = os.getenv("USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

TRAIN_CUTOFF  = "2024-01-01"
MIN_EDGE      = 0.06    # only bet when model has ≥6% edge over no-vig DK/FD line
MAX_EDGE      = 0.30    # cap at 30% — model well-calibrated up to here; 35%+ slightly overestimates
KELLY_FRAC    = 0.15    # 15% fractional Kelly — conservative sizing
MAX_UNITS     = 5.0     # hard cap 5u ($50) per fight
STARTING_BANK = 1000.0
ELO_START     = 1500.0
ELO_K_EARLY   = 170.0
ELO_K_LATE    = 85.0
ELO_K_CUTOFF  = 5
DECAY_LAMBDA  = 0.85
