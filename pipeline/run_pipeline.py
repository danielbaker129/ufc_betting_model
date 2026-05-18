"""
Full pipeline orchestrator. Run this after ufcstats scraper completes.
Sequence: ELO → Features → Train all models → Backtest → Launch dashboard
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = ROOT / "venv" / "bin" / "python"


def run(script: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([str(PYTHON), script], cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED after {elapsed:.1f}s")
        sys.exit(1)
    print(f"  Done in {elapsed:.1f}s")


def main():
    steps = [
        ("pipeline/elo.py",          "Phase 3: Computing ELO ratings"),
        ("pipeline/features.py",     "Phase 4: Building feature matrix"),
        ("models/moneyline.py",      "Phase 5a: Training moneyline model"),
        ("models/method.py",         "Phase 5b: Training method-of-victory model"),
        ("models/rounds.py",         "Phase 5c: Training rounds model"),
        ("models/props.py",          "Phase 5d: Training props models"),
        ("betting/backtest.py",      "Phase 6: Running backtest"),
        ("scrapers/betmma.py",       "Phase 2b: Scraping betmma.tips odds"),
    ]

    print("UFC Betting Model — Full Pipeline")
    print("="*60)

    for script, desc in steps:
        run(script, desc)

    print("\n" + "="*60)
    print("  ALL PHASES COMPLETE")
    print("="*60)
    print("\nTo launch dashboard:")
    print("  bash run.sh")
    print("  or: source venv/bin/activate && streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
