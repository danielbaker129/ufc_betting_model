# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source venv/bin/activate          # always activate before running anything
python pipeline/config.py         # verify env vars load (.env required)
```

## Running the Project

```bash
bash run.sh                       # launch Streamlit dashboard at localhost:8501
streamlit run dashboard/app.py --server.port 8501
```

## Data Pipeline (run in order)

```bash
# 1. Scrape fight data (takes ~2-4 hours, safe to re-run — uses INSERT OR IGNORE)
python scrapers/ufcstats_fast.py

# 2. Scrape historical odds (requires BETMMA_EMAIL + BETMMA_PASSWORD in .env)
python scrapers/betmma.py

# 3. Build features and train models
python pipeline/run_pipeline.py   # runs ELO → features → all models → backtest
```

Or run stages individually:
```bash
python pipeline/elo.py            # compute ELO ratings (replay all fights chronologically)
python pipeline/features.py       # build leak-free feature matrix → data/processed/feature_matrix.csv
python models/moneyline.py        # LightGBM + Optuna (50 trials) → models/moneyline.pkl
python models/method.py           # XGBoost multi-class (KO/Sub/Dec) → models/method.pkl
python models/rounds.py           # goes-distance classifier → models/rounds.pkl
python models/props.py            # strikes/TDs regression → models/props.pkl
python betting/backtest.py        # simulate bets on test set → data/processed/backtest_results.csv
```

## Architecture

### Data Flow
```
ufcstats.com ──→ scrapers/ufcstats_fast.py ──→ SQLite (data/ufc.db)
betmma.tips  ──→ scrapers/betmma.py        ──→ odds_history table
fightodds.io ──→ scrapers/fightodds.py     ──→ odds_history (multi-book, opening+closing)
                                                      │
                                            pipeline/elo.py (replay fights chronologically)
                                                      │
                                            pipeline/features.py
                                            (CRITICAL: rolling stats computed from fights
                                             strictly BEFORE each fight's date — no leakage)
                                                      │
                                         data/processed/feature_matrix.csv
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                         models/                  models/               models/
                         moneyline.py             method.py             rounds.py
                         (LightGBM)               (XGBoost)             (GBM)
                              └───────────────────────┴───────────────────────┘
                                                      │
                                            betting/backtest.py
                                            (fixed $10/unit sizing, Kelly for units)
                                                      │
                                            dashboard/app.py (Streamlit)
```

### Key Design Decisions

**Leak-free features** (`pipeline/features.py`): For every fight, rolling stats (slpm, td_def, etc.) are computed using only fights that occurred strictly before that fight's date. The feature matrix uses a chronological train/test split (TRAIN_CUTOFF = "2024-01-01") — never random k-fold.

**Differential features**: All model inputs are `fighter_a_stat − fighter_b_stat`. ELO diff is the single most predictive feature.

**Recency weighting**: Two decay levels are used. `DECAY_LAMBDA=0.85` for career averages. `RECENCY_DECAY=0.65` for the `*_recent` features — last fight counts ~10x more than fight #10. Trend features (`slpm_trend = slpm_recent - slpm_career`) explicitly capture declining fighters like aging veterans. `age_decline_diff` = `max(0, age-30)/10 × min(trend, 0)` doubly penalizes old fighters who are also getting worse.

**Betting sizing**: 1 unit = $10. Kelly criterion determines unit count (capped at 5u), applied to a fixed baseline — **not compounding bankroll**. `kelly_units()` in `betting/kelly.py` is the correct sizing function; `kelly_stake()` is legacy only used in backtest dollar tracking.

**Odds alignment**: When matching betmma/fightodds odds to DB fights, fighter order may be reversed. `find_fight_id()` returns `(fight_id, swapped: bool)` — if swapped, odds_a and odds_b must be exchanged before storing.

**Devig**: Shin method (`betting/devig.py:devig_shin`) is used for all edge calculations. Edge = model_prob − no_vig_prob.

### Database Schema (SQLite at data/ufc.db)

- `events`: UFC events (event_id, name, date, location)
- `fighters`: Fighter profiles (physical attributes, career record)
- `fights`: Fight results (fighter_a_id=red corner, fighter_b_id=blue corner, winner_id, method, round)
- `fight_stats`: Per-round striking/grappling stats (2 rows per round: one per fighter)
- `elo_history`: Pre-fight ELO snapshots (elo_before = feature, elo_after = updated rating)
- `odds_history`: Historical odds, multiple rows per fight (one per book/source). `book` field: `betmma`, `fightodds_DK`, `fightodds_Pinnacle_open`, etc.

### Odds Sources

User bets at **DraftKings and FanDuel only**. All edge calculations use best available DK/FD line (line-shopped). The "Book" column in recommendations tells you which to use.

| Source | Book field prefix | Coverage | Notes |
|--------|-------------------|----------|-------|
| betmma.tips | `betmma` | 2012–present | Login required; fallback when DK/FD not available |
| fightodds.io | `fightodds_DraftKings`, `fightodds_FanDuel` | 2018–present | Primary source; opening+closing per book; GraphQL API |
| The Odds API | live only | upcoming | DK+FD requested via `bookmakers=draftkings,fanduel` |

Priority in backtest/edge: `fightodds_DraftKings` → `fightodds_FanDuel` → `betmma` (fallback).

### Model Files

All models saved as `.pkl` in `models/`. Each pickle contains `{"model": ..., "feature_cols": [...], "metrics": {...}}`. Load with `pickle.load()` — no special class needed.

### API Keys (.env)

```
ODDS_API_KEY=...       # The Odds API — live upcoming UFC odds
API_SPORTS_KEY=...     # API-Sports — supplementary
BETMMA_EMAIL=...       # betmma.tips login
BETMMA_PASSWORD=...
```

### Dashboard Tabs

1. **Next Event** — fetches live odds from The Odds API, runs all models, shows plays where model edge > 0 over no-vig line, grouped by event date
2. **Backtest Results** — historical performance on test set (2024+), unit P&L charts
3. **Past Cards** — every historical event with per-fight model predictions, actual P&L per bet (Stake column = amount wagered, P&L column = actual profit/loss at the real odds)
4. **Fighter Lookup** — ELO history chart + recent fight log
