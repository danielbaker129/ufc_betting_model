"""
Leak-free feature engineering pipeline.
For each fight, computes rolling stats using ONLY fights strictly before that fight's date.
Outputs data/processed/feature_matrix.csv
"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, PROCESSED_DIR, DECAY_LAMBDA


def load_fights(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT f.fight_id, f.event_id, f.fight_date,
                  f.fighter_a_id, f.fighter_b_id, f.winner_id,
                  f.method, f.round, f.time, f.time_format,
                  f.is_title_fight, f.weight_class,
                  fa.name AS name_a, fa.dob AS dob_a,
                  fa.height_inches AS height_a, fa.reach_inches AS reach_a,
                  fa.stance AS stance_a,
                  fb.name AS name_b, fb.dob AS dob_b,
                  fb.height_inches AS height_b, fb.reach_inches AS reach_b,
                  fb.stance AS stance_b
           FROM fights f
           JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
           JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
           WHERE f.fight_date IS NOT NULL AND f.fight_date != ''
             AND f.winner_id IS NOT NULL
           ORDER BY f.fight_date ASC, f.fight_id ASC""",
        con,
    )


def load_stats(con: sqlite3.Connection) -> pd.DataFrame:
    """Aggregate per-fight totals from per-round stats."""
    return pd.read_sql_query(
        """SELECT fight_id, fighter_id,
                  SUM(sig_str_landed)    AS sig_landed,
                  SUM(sig_str_attempted) AS sig_att,
                  SUM(total_str_landed)  AS tot_landed,
                  SUM(total_str_attempted) AS tot_att,
                  SUM(td_landed)         AS td_landed,
                  SUM(td_attempted)      AS td_att,
                  SUM(sub_attempts)      AS sub_att,
                  SUM(knockdowns)        AS knockdowns,
                  SUM(ctrl_seconds)      AS ctrl_secs,
                  SUM(head_landed)       AS head_landed,
                  SUM(body_landed)       AS body_landed,
                  SUM(leg_landed)        AS leg_landed
           FROM fight_stats
           GROUP BY fight_id, fighter_id""",
        con,
    )


def load_elo(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT fighter_id, fight_id, elo_before FROM elo_history",
        con,
    )


def load_market_odds(con: sqlite3.Connection) -> pd.DataFrame:
    """Load best available DK/FD odds per fight for the market_nv_a feature.

    Real sportsbook lines only (no betmma). market_nv_a = 0.5 when unavailable,
    which acts as a neutral prior. The partial coverage (~28% of all fights, ~83%
    of 2022+ fights) lets the model learn stats patterns from historical data while
    still using the market signal for recent fights where it matters most.
    """
    return pd.read_sql_query(
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


def parse_time_seconds(time_str: str, rnd: int, time_format: str) -> float:
    """Convert fight result time to total seconds fought."""
    if not time_str or not rnd:
        return 0.0
    try:
        m, s = time_str.strip().split(":")
        round_secs = (rnd - 1) * 300 + int(m) * 60 + int(s)
        return float(round_secs)
    except Exception:
        return float((rnd or 1) * 300)


RECENCY_DECAY = 0.65   # steeper than career decay — last 3 fights dominate

def weighted_mean(values: list[float], decay: float = DECAY_LAMBDA) -> float:
    if not values:
        return 0.0
    weights = [decay ** (len(values) - 1 - i) for i in range(len(values))]
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

def recency_mean(values: list[float]) -> float:
    """Aggressive recency weighting (lambda=0.65). Last fight counts 10x fight #10."""
    return weighted_mean(values, decay=RECENCY_DECAY)


def compute_fighter_rolling_stats(
    fighter_id: str,
    before_date: str,
    fights_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict:
    """Compute rolling stats for a fighter using only fights before before_date."""
    prior = fights_df[
        (fights_df["fight_date"] < before_date) &
        ((fights_df["fighter_a_id"] == fighter_id) | (fights_df["fighter_b_id"] == fighter_id))
    ].sort_values("fight_date")

    if prior.empty:
        return {
            "n_fights": 0,
            "slpm_career": 4.0, "str_acc_career": 0.43, "td_avg_career": 1.5,
            "td_acc_career": 0.40, "sub_avg_career": 0.2,
            "slpm_L5": 4.0, "str_acc_L5": 0.43, "td_avg_L5": 1.5,
            "slpm_L3": 4.0, "str_acc_L3": 0.43, "td_avg_L3": 1.5,
            "slpm_recent": 4.0, "str_acc_recent": 0.43, "td_avg_recent": 1.5,
            "avg_fight_secs": 750.0, "ko_finish_rate": 0.25,
            "sub_finish_rate": 0.15, "ko_susceptibility": 0.20,
            "win_streak": 0, "days_since_last": 365.0,
            "recent_win_rate": 0.5, "last_fight_won": 0,
            # Form/trend — 0 means neutral (no trend detected)
            "slpm_trend": 0.0, "str_acc_trend": 0.0, "win_rate_trend": 0.0,
            "age_decline_score": 0.0,
        }

    fight_ids = prior["fight_id"].tolist()
    f_stats = stats_df[stats_df["fighter_id"] == fighter_id]
    f_stats = f_stats[f_stats["fight_id"].isin(fight_ids)]
    f_stats = f_stats.merge(prior[["fight_id", "fight_date", "round", "time", "time_format",
                                    "winner_id", "method"]],
                            on="fight_id", how="left")
    f_stats = f_stats.sort_values("fight_date")

    if f_stats.empty:
        return {
            "n_fights": 0,
            "slpm_career": 4.0, "str_acc_career": 0.43, "td_avg_career": 1.5,
            "td_acc_career": 0.40, "sub_avg_career": 0.2,
            "slpm_L5": 4.0, "str_acc_L5": 0.43, "td_avg_L5": 1.5,
            "slpm_L3": 4.0, "str_acc_L3": 0.43, "td_avg_L3": 1.5,
            "slpm_recent": 4.0, "str_acc_recent": 0.43, "td_avg_recent": 1.5,
            "avg_fight_secs": 750.0, "ko_finish_rate": 0.25,
            "sub_finish_rate": 0.15, "ko_susceptibility": 0.20,
            "win_streak": 0, "days_since_last": 365.0,
            "recent_win_rate": 0.5, "last_fight_won": 0,
            "slpm_trend": 0.0, "str_acc_trend": 0.0, "win_rate_trend": 0.0,
        }

    # Fight time in seconds for each fight
    f_stats["fight_secs"] = f_stats.apply(
        lambda r: parse_time_seconds(r["time"], r["round"], r["time_format"]), axis=1
    )
    f_stats["fight_mins"] = (f_stats["fight_secs"] / 60.0).clip(lower=0.01)

    # Per-minute rates
    f_stats["slpm"]    = f_stats["sig_landed"] / f_stats["fight_mins"]
    f_stats["sapm"]    = 0.0  # filled from opponent stats below
    f_stats["str_acc"] = np.where(f_stats["sig_att"] > 0, f_stats["sig_landed"] / f_stats["sig_att"], 0.0)
    f_stats["td_avg"]  = f_stats["td_landed"] / f_stats["fight_mins"]
    f_stats["td_acc"]  = np.where(f_stats["td_att"] > 0, f_stats["td_landed"] / f_stats["td_att"], 0.0)
    f_stats["sub_avg"] = f_stats["sub_att"] / f_stats["fight_mins"]

    # Win/loss records
    f_stats["won"] = (f_stats["winner_id"] == fighter_id).astype(int)
    f_stats["ko_win"] = ((f_stats["won"] == 1) & f_stats["method"].str.contains("KO|TKO", na=False)).astype(int)
    f_stats["sub_win"] = ((f_stats["won"] == 1) & f_stats["method"].str.contains("Sub", na=False, case=False)).astype(int)
    f_stats["ko_loss"] = ((f_stats["won"] == 0) & f_stats["method"].str.contains("KO|TKO", na=False)).astype(int)

    n = len(f_stats)
    last3 = f_stats.tail(3)
    last5 = f_stats.tail(5)

    def wrm(col, df):
        return weighted_mean(df[col].tolist())

    def rrm(col, df):
        """Aggressive recency-weighted mean (lambda=0.65)."""
        return recency_mean(df[col].tolist())

    # Career weighted averages (lambda=0.85)
    slpm_career    = wrm("slpm", f_stats)
    str_acc_career = wrm("str_acc", f_stats)
    td_avg_career  = wrm("td_avg", f_stats)
    win_rate_career = f_stats["won"].mean()

    # Recent form — aggressive recency weighting (lambda=0.65) on last 5 fights
    slpm_recent    = rrm("slpm", last5)
    str_acc_recent = rrm("str_acc", last5)
    td_avg_recent  = rrm("td_avg", last5)
    win_rate_recent = last5["won"].mean() if len(last5) >= 2 else win_rate_career

    # Trend: recent minus career (negative = declining, positive = improving)
    slpm_trend     = slpm_recent - slpm_career
    str_acc_trend  = str_acc_recent - str_acc_career
    win_rate_trend = win_rate_recent - win_rate_career

    # Last fight result (most recent signal)
    last_fight_won = int(f_stats.iloc[-1]["won"]) if n > 0 else 0

    result = {
        "n_fights": n,
        # Career weighted stats
        "slpm_career":    slpm_career,
        "str_acc_career": str_acc_career,
        "td_avg_career":  td_avg_career,
        "td_acc_career":  wrm("td_acc", f_stats),
        "sub_avg_career": wrm("sub_avg", f_stats),
        # L5 weighted
        "slpm_L5":        wrm("slpm", last5),
        "str_acc_L5":     wrm("str_acc", last5),
        "td_avg_L5":      wrm("td_avg", last5),
        # L3 weighted
        "slpm_L3":        wrm("slpm", last3),
        "str_acc_L3":     wrm("str_acc", last3),
        "td_avg_L3":      wrm("td_avg", last3),
        # Aggressively recency-weighted (lambda=0.65) — captures current form
        "slpm_recent":    slpm_recent,
        "str_acc_recent": str_acc_recent,
        "td_avg_recent":  td_avg_recent,
        # Trend features — most important for detecting decline/improvement
        "slpm_trend":      slpm_trend,       # negative for Burns-type decline
        "str_acc_trend":   str_acc_trend,
        "win_rate_trend":  win_rate_trend,   # losing more recently than career avg
        # Recent win rate (last 5)
        "recent_win_rate": win_rate_recent,
        "last_fight_won":  last_fight_won,
        # Other
        "avg_fight_secs": f_stats["fight_secs"].mean(),
        "ko_finish_rate": f_stats["ko_win"].sum() / max(n, 1),
        "sub_finish_rate": f_stats["sub_win"].sum() / max(n, 1),
        "ko_susceptibility": f_stats["ko_loss"].sum() / max(n, 1),
        "win_streak": _current_streak(f_stats["won"].tolist()),
        "days_since_last": _days_since(f_stats["fight_date"].tolist(), before_date),
    }
    return result


def _current_streak(results: list[int]) -> int:
    """Positive = win streak, negative = loss streak."""
    if not results:
        return 0
    streak = 0
    last = results[-1]
    for r in reversed(results):
        if r == last:
            streak += (1 if last == 1 else -1)
        else:
            break
    return streak


def _days_since(dates: list[str], reference: str) -> float:
    if not dates:
        return 365.0
    try:
        last = datetime.strptime(dates[-1], "%Y-%m-%d")
        ref = datetime.strptime(reference, "%Y-%m-%d")
        return max(0.0, (ref - last).days)
    except Exception:
        return 365.0


def compute_opponent_sapm(
    fighter_id: str,
    before_date: str,
    fights_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> float:
    """Compute average strikes absorbed per minute — from opponent's landed strikes."""
    prior = fights_df[
        (fights_df["fight_date"] < before_date) &
        ((fights_df["fighter_a_id"] == fighter_id) | (fights_df["fighter_b_id"] == fighter_id))
    ].sort_values("fight_date")

    if prior.empty:
        return 4.0  # league average sapm

    sapm_vals = []
    for _, row in prior.iterrows():
        opp_id = row["fighter_b_id"] if row["fighter_a_id"] == fighter_id else row["fighter_a_id"]
        opp_stat = stats_df[(stats_df["fight_id"] == row["fight_id"]) & (stats_df["fighter_id"] == opp_id)]
        if opp_stat.empty:
            continue
        fight_secs = parse_time_seconds(row["time"], row["round"], row["time_format"])
        fight_mins = max(fight_secs / 60.0, 0.01)
        sapm_vals.append(opp_stat["sig_landed"].values[0] / fight_mins)

    return weighted_mean(sapm_vals)


def compute_td_defense(
    fighter_id: str,
    before_date: str,
    fights_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> float:
    """TD defense = 1 - (opp TDs landed / opp TDs attempted)."""
    prior = fights_df[
        (fights_df["fight_date"] < before_date) &
        ((fights_df["fighter_a_id"] == fighter_id) | (fights_df["fighter_b_id"] == fighter_id))
    ]

    if prior.empty:
        return 0.5

    opp_td_land = 0
    opp_td_att = 0
    for _, row in prior.iterrows():
        opp_id = row["fighter_b_id"] if row["fighter_a_id"] == fighter_id else row["fighter_a_id"]
        opp_stat = stats_df[(stats_df["fight_id"] == row["fight_id"]) & (stats_df["fighter_id"] == opp_id)]
        if opp_stat.empty:
            continue
        opp_td_land += opp_stat["td_landed"].values[0]
        opp_td_att += opp_stat["td_att"].values[0]

    return 1.0 - (opp_td_land / max(opp_td_att, 1))


def build_feature_matrix(con: sqlite3.Connection) -> pd.DataFrame:
    print("Loading data from DB...")
    fights = load_fights(con)
    stats = load_stats(con)
    elo = load_elo(con)
    market = load_market_odds(con)

    print(f"  {len(fights)} fights, {len(stats)} stat rows, {len(elo)} ELO rows, {len(market)} with market odds")

    elo_map = {(r["fighter_id"], r["fight_id"]): r["elo_before"] for _, r in elo.iterrows()}

    # Market odds map: fight_id → best no-vig prob for fighter_a
    def best_odds_val(row, col_dk, col_fd):
        dk = row.get(col_dk)
        fd = row.get(col_fd)
        valid = [c for c in [dk, fd] if c is not None and not pd.isna(c) and abs(int(c)) < 5000]
        if not valid: return None
        def dec(o): return int(o)/100+1 if int(o)>0 else 100/abs(int(o))+1
        return max(valid, key=dec)

    def shin_nv_a(odds_a, odds_b):
        """Return Shin-devigged no-vig prob for fighter A. 0.5 if no odds."""
        if odds_a is None or odds_b is None: return 0.5
        try:
            from betting.devig import devig_shin
            pa, _ = devig_shin(int(odds_a), int(odds_b))
            return float(pa)
        except Exception:
            return 0.5

    market_nv_map = {}  # fight_id → market_nv_a
    for _, mrow in market.iterrows():
        fid = mrow["fight_id"]
        oa = best_odds_val(mrow, "dk_a", "fd_a")
        ob = best_odds_val(mrow, "dk_b", "fd_b")
        market_nv_map[fid] = shin_nv_a(oa, ob)

    rows = []
    total = len(fights)

    for i, fight in fights.iterrows():
        if i % 500 == 0:
            print(f"  Processing fight {i}/{total}...")

        fa_id = fight["fighter_a_id"]
        fb_id = fight["fighter_b_id"]
        date = fight["fight_date"]

        fa_stats = compute_fighter_rolling_stats(fa_id, date, fights, stats)
        fb_stats = compute_fighter_rolling_stats(fb_id, date, fights, stats)

        # Both will now always return something (neutral defaults for 0-fight history)

        fa_sapm = compute_opponent_sapm(fa_id, date, fights, stats)
        fb_sapm = compute_opponent_sapm(fb_id, date, fights, stats)
        fa_td_def = compute_td_defense(fa_id, date, fights, stats)
        fb_td_def = compute_td_defense(fb_id, date, fights, stats)

        # Age differential
        def age_at(dob_str, date_str):
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d")
                d = datetime.strptime(date_str, "%Y-%m-%d")
                return (d - dob).days / 365.25
            except Exception:
                return None

        age_a = age_at(fight["dob_a"], date)
        age_b = age_at(fight["dob_b"], date)

        fa_elo = elo_map.get((fa_id, fight["fight_id"]), 1500.0)
        fb_elo = elo_map.get((fb_id, fight["fight_id"]), 1500.0)

        # Stance encoding (southpaw advantage)
        stance_a = str(fight["stance_a"]).lower()
        stance_b = str(fight["stance_b"]).lower()
        both_orthodox = int(stance_a == "orthodox" and stance_b == "orthodox")
        a_southpaw = int(stance_a == "southpaw" and stance_b == "orthodox")
        b_southpaw = int(stance_b == "southpaw" and stance_a == "orthodox")

        # Method encoding for target construction
        method = str(fight.get("method", "")).upper()
        method_ko = int("KO" in method or "TKO" in method)
        method_sub = int("SUB" in method)
        method_dec = int("DEC" in method or "DECISION" in method)

        target = 1 if fight["winner_id"] == fa_id else 0

        # Age × decline interaction — old AND declining = doubly penalized
        def _age_decline(age, trend):
            if not age:
                return 0.0
            age_factor = max(0.0, (age - 30) / 10.0)  # 0 at age 30, 1.0 at age 40
            return age_factor * min(trend, 0.0)        # only fires when declining

        fa_age_decline = _age_decline(age_a, fa_stats.get("slpm_trend", 0.0))
        fb_age_decline = _age_decline(age_b, fb_stats.get("slpm_trend", 0.0))

        row = {
            "fight_id": fight["fight_id"],
            "fight_date": date,
            "name_a": fight["name_a"],
            "name_b": fight["name_b"],
            "weight_class": fight["weight_class"],
            "is_title_fight": fight["is_title_fight"],
            "is_5_round": int(str(fight["time_format"]).startswith("5")),
            "both_orthodox": both_orthodox,
            "a_southpaw_vs_orthodox": a_southpaw,
            "b_southpaw_vs_orthodox": b_southpaw,

            # Differentials (A - B)
            "slpm_career_diff":    fa_stats["slpm_career"] - fb_stats["slpm_career"],
            "slpm_L5_diff":        fa_stats["slpm_L5"] - fb_stats["slpm_L5"],
            "slpm_L3_diff":        fa_stats["slpm_L3"] - fb_stats["slpm_L3"],
            # Aggressively recency-weighted (captures current form best)
            "slpm_recent_diff":    fa_stats["slpm_recent"] - fb_stats["slpm_recent"],
            "str_acc_recent_diff": fa_stats["str_acc_recent"] - fb_stats["str_acc_recent"],
            "td_avg_recent_diff":  fa_stats["td_avg_recent"] - fb_stats["td_avg_recent"],
            # Trend features — negative means declining (key for Burns-type cases)
            "slpm_trend_diff":     fa_stats["slpm_trend"] - fb_stats["slpm_trend"],
            "str_acc_trend_diff":  fa_stats["str_acc_trend"] - fb_stats["str_acc_trend"],
            "win_rate_trend_diff": fa_stats["win_rate_trend"] - fb_stats["win_rate_trend"],
            # Recent win rate
            "recent_win_rate_diff": fa_stats["recent_win_rate"] - fb_stats["recent_win_rate"],
            "last_fight_won_diff": fa_stats["last_fight_won"] - fb_stats["last_fight_won"],
            # Age × decline interaction (most negative = old AND declining)
            "age_decline_diff":    fa_age_decline - fb_age_decline,
            "sapm_diff":           fa_sapm - fb_sapm,
            "net_strike_diff":     (fa_stats["slpm_career"] - fa_sapm) - (fb_stats["slpm_career"] - fb_sapm),
            "str_acc_career_diff": fa_stats["str_acc_career"] - fb_stats["str_acc_career"],
            "str_acc_L5_diff":     fa_stats["str_acc_L5"] - fb_stats["str_acc_L5"],
            "td_avg_career_diff":  fa_stats["td_avg_career"] - fb_stats["td_avg_career"],
            "td_avg_L5_diff":      fa_stats["td_avg_L5"] - fb_stats["td_avg_L5"],
            "td_acc_career_diff":  fa_stats["td_acc_career"] - fb_stats["td_acc_career"],
            "td_def_diff":         fa_td_def - fb_td_def,
            "sub_avg_diff":        fa_stats["sub_avg_career"] - fb_stats["sub_avg_career"],
            "ko_finish_rate_diff": fa_stats["ko_finish_rate"] - fb_stats["ko_finish_rate"],
            "sub_finish_rate_diff": fa_stats["sub_finish_rate"] - fb_stats["sub_finish_rate"],
            "ko_susceptibility_diff": fa_stats["ko_susceptibility"] - fb_stats["ko_susceptibility"],
            "avg_fight_secs_diff": fa_stats["avg_fight_secs"] - fb_stats["avg_fight_secs"],
            "experience_diff":     fa_stats["n_fights"] - fb_stats["n_fights"],
            "win_streak_diff":     fa_stats["win_streak"] - fb_stats["win_streak"],
            "elo_diff":            fa_elo - fb_elo,
            "reach_diff":          (fight["reach_a"] or 0) - (fight["reach_b"] or 0),
            "height_diff":         (fight["height_a"] or 0) - (fight["height_b"] or 0),
            "age_diff":            (age_a or 0) - (age_b or 0) if age_a and age_b else 0,
            "days_since_last_a":   fa_stats["days_since_last"],
            "days_since_last_b":   fb_stats["days_since_last"],

            # Market prior — DK/FD no-vig probability for fighter A
            # When available, this encodes all market information not in stats
            # Model learns RESIDUAL between its prediction and market → captures true edge
            # 0.5 = no market data (pre-2022 fights)
            "market_nv_a": market_nv_map.get(fight["fight_id"], 0.5),

            # Method of victory targets
            "method_ko":  method_ko,
            "method_sub": method_sub,
            "method_dec": method_dec,

            # Primary target
            "target": target,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nFeature matrix: {len(df)} rows x {len(df.columns)} columns")
    print(f"Class balance: {df['target'].mean():.3f} (red corner win rate)")

    out = PROCESSED_DIR / "feature_matrix.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
    return df


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    df = build_feature_matrix(con)
    con.close()
    print(df.head())


if __name__ == "__main__":
    main()
