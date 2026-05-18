"""
Backtests the moneyline model on the test set (2024+).
Uses DK/FD closing lines from DB. Fixed $10/unit sizing — Kelly determines units.
Bets only when model has positive edge over no-vig line.

Saves an enriched CSV with ALL fights (bet and pass), including both fighters'
odds, model probs, winner, method, and event name — used directly by Past Cards.
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, PROCESSED_DIR, MODELS_DIR, TRAIN_CUTOFF
from betting.edge import evaluate_fight
from betting.kelly import to_decimal

UNIT = 10.0


def run_backtest():
    feat_path = PROCESSED_DIR / "feature_matrix.csv"
    model_path = MODELS_DIR / "moneyline.pkl"
    if not feat_path.exists() or not model_path.exists():
        print("Run pipeline/features.py and models/moneyline.py first.")
        return

    with open(model_path, "rb") as f:
        m = pickle.load(f)
    model     = m["model"]
    feat_cols = m["feature_cols"]

    feat_df = pd.read_csv(feat_path, parse_dates=["fight_date"])
    test_df = feat_df[feat_df["fight_date"] >= TRAIN_CUTOFF].copy()

    con = sqlite3.connect(DB_PATH)

    # DK/FD closing lines only — betmma excluded because its prices differ from
    # real sportsbooks and inflate apparent edge in backtesting.
    odds_df = pd.read_sql_query(
        """SELECT fight_id,
                  MAX(CASE WHEN book IN ('fightodds_DraftKings','bfo_DraftKings','capture_DraftKings') THEN fighter_a_odds END) as dk_a,
                  MAX(CASE WHEN book IN ('fightodds_DraftKings','bfo_DraftKings','capture_DraftKings') THEN fighter_b_odds END) as dk_b,
                  MAX(CASE WHEN book IN ('fightodds_FanDuel','bfo_FanDuel','capture_FanDuel')          THEN fighter_a_odds END) as fd_a,
                  MAX(CASE WHEN book IN ('fightodds_FanDuel','bfo_FanDuel','capture_FanDuel')          THEN fighter_b_odds END) as fd_b
           FROM odds_history
           WHERE fight_id IS NOT NULL
             AND book IN ('fightodds_DraftKings','fightodds_FanDuel','bfo_DraftKings','bfo_FanDuel',
                          'capture_DraftKings','capture_FanDuel')
           GROUP BY fight_id""",
        con,
    )

    fights_meta = pd.read_sql_query(
        "SELECT fight_id, event_id, method, round FROM fights", con
    )
    events_meta = pd.read_sql_query(
        "SELECT event_id, name as event_name FROM events", con
    )
    con.close()

    def best_dk_fd_odds(row, side):
        col = "a" if side == "a" else "b"
        candidates = [row.get(f"dk_{col}"), row.get(f"fd_{col}")]
        valid = [c for c in candidates if c is not None and not pd.isna(c) and abs(int(c)) < 5000]
        if not valid:
            return None
        return max(int(c) for c in valid)

    merged = test_df.merge(odds_df, on="fight_id", how="left")
    merged = merged.merge(fights_meta, on="fight_id", how="left")
    merged = merged.merge(events_meta, on="event_id", how="left")
    merged["fighter_a_odds"] = merged.apply(lambda r: best_dk_fd_odds(r, "a"), axis=1)
    merged["fighter_b_odds"] = merged.apply(lambda r: best_dk_fd_odds(r, "b"), axis=1)
    merged["has_real_odds"]  = merged["fighter_a_odds"].notna() & merged["fighter_b_odds"].notna()

    has_real = merged["has_real_odds"].sum()
    print(f"Test fights: {len(merged)}  |  With DK/FD odds: {has_real}  |  Skipping {len(merged)-has_real} (no real lines)")

    X       = merged[feat_cols].fillna(0).values
    probs_a = model.predict_proba(X)[:, 1]

    cumulative_units = 0.0
    results = []

    for i, (_, row) in enumerate(merged.iterrows()):
        prob_a = float(probs_a[i])
        prob_b = 1.0 - prob_a
        model_pick = "a" if prob_a >= 0.5 else "b"
        winner     = "a" if row.get("target", 0) == 1 else "b"

        method_str = str(row.get("method", "")).strip()
        rnd        = row.get("round", "")
        result_str = f"{method_str} R{int(rnd)}" if rnd and str(rnd) not in ("nan", "", "None") else method_str

        has_real = bool(row.get("has_real_odds", False))
        odds_a_val = int(row["fighter_a_odds"]) if has_real else None
        odds_b_val = int(row["fighter_b_odds"]) if has_real else None

        # Pre-compute model_edge for display even on non-bet rows
        if has_real:
            from betting.devig import no_vig_probs as _nv
            _nv_a, _nv_b = _nv(odds_a_val, odds_b_val)
            _model_edge = round((prob_a - _nv_a) if model_pick == "a" else (prob_b - _nv_b), 4)
        else:
            _model_edge = None

        base = {
            "fight_id":      row.get("fight_id"),
            "date":          row["fight_date"],
            "event_name":    row.get("event_name") or "",
            "name_a":        row.get("name_a", ""),
            "name_b":        row.get("name_b", ""),
            "has_real_odds": has_real,
            "odds_a":        odds_a_val,
            "odds_b":        odds_b_val,
            "prob_a":        round(prob_a, 4),
            "prob_b":        round(prob_b, 4),
            "model_pick":    model_pick,
            "model_edge":    _model_edge,
            "winner":        winner,
            "model_correct": model_pick == winner,
            "result":        result_str,
            "bet":           False,
            "bet_on":        None,
            "edge":          None,
            "units":         0.0,
            "stake":         0.0,
            "won":           None,
            "pnl_units":     0.0,
            "pnl":           0.0,
            "cumulative_units": round(cumulative_units, 2),
        }

        if not has_real:
            results.append(base)
            continue

        rec = evaluate_fight(prob_a, prob_b, odds_a_val, odds_b_val)

        if not rec["bet"]:
            results.append(base)
            continue

        bet_side  = rec["bet_side"]
        units     = rec["units"]
        dollars   = rec["dollars"]
        odds_taken = rec["odds_taken"]

        dec           = to_decimal(odds_taken)
        won           = winner == bet_side
        pnl_units     = units * (dec - 1) if won else -units
        cumulative_units += pnl_units

        base.update({
            "bet":              True,
            "bet_on":           bet_side,
            "edge":             rec["edge"],
            "units":            round(units, 2),
            "stake":            round(dollars, 2),
            "won":              won,
            "pnl_units":        round(pnl_units, 2),
            "pnl":              round(pnl_units * UNIT, 2),
            "cumulative_units": round(cumulative_units, 2),
        })
        results.append(base)

    results_df = pd.DataFrame(results)
    bet_df     = results_df[results_df["bet"] == True]

    if bet_df.empty:
        print("No bets placed.")
        return results_df

    total_bets   = len(bet_df)
    wins         = bet_df["won"].sum()
    win_rate     = wins / total_bets
    units_staked = bet_df["units"].sum()
    units_pnl    = bet_df["pnl_units"].sum()
    roi          = units_pnl / max(units_staked, 1)
    avg_units    = bet_df["units"].mean()
    avg_edge     = bet_df["edge"].mean()

    print(f"\n{'='*52}")
    print(f"BACKTEST  ({TRAIN_CUTOFF} → present)   1u = ${UNIT:.0f}")
    print(f"{'='*52}")
    print(f"Bets:          {total_bets}")
    print(f"Win rate:      {win_rate*100:.1f}%")
    print(f"Units staked:  {units_staked:.1f}u  (${units_staked*UNIT:,.0f})")
    print(f"P&L:           {units_pnl:+.1f}u  (${units_pnl*UNIT:+,.0f})")
    print(f"ROI:           {roi*100:.1f}%")
    print(f"Avg bet size:  {avg_units:.2f}u  (${avg_units*UNIT:.2f})")
    print(f"Avg edge:      {avg_edge*100:.1f}%")
    print(f"{'='*52}")

    if "date" in bet_df.columns:
        bet_df = bet_df.copy()
        bet_df["year"] = pd.to_datetime(bet_df["date"]).dt.year
        print("\nYearly:")
        for yr, g in bet_df.groupby("year"):
            yr_roi = g["pnl_units"].sum() / max(g["units"].sum(), 1)
            print(f"  {yr}: {len(g)} bets  {g['won'].mean()*100:.0f}% WR  {yr_roi*100:.1f}% ROI  {g['pnl_units'].sum():+.1f}u")

    out = PROCESSED_DIR / "backtest_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    return results_df


if __name__ == "__main__":
    run_backtest()
