"""
Leak-free feature engineering pipeline.
For each fight, computes rolling stats using ONLY fights strictly before that fight's date.
Outputs data/processed/feature_matrix.csv
"""
import sqlite3
import sys
from collections import defaultdict
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
    """Aggregate per-fight totals and per-round splits from fight_stats."""
    return pd.read_sql_query(
        """SELECT fight_id, fighter_id,
                  SUM(sig_str_landed)      AS sig_landed,
                  SUM(sig_str_attempted)   AS sig_att,
                  SUM(total_str_landed)    AS tot_landed,
                  SUM(total_str_attempted) AS tot_att,
                  SUM(td_landed)           AS td_landed,
                  SUM(td_attempted)        AS td_att,
                  SUM(sub_attempts)        AS sub_att,
                  SUM(knockdowns)          AS knockdowns,
                  SUM(ctrl_seconds)        AS ctrl_secs,
                  SUM(head_landed)         AS head_landed,
                  SUM(body_landed)         AS body_landed,
                  SUM(leg_landed)          AS leg_landed,
                  -- R1-only output (early-round performance / first-impression signal)
                  SUM(CASE WHEN round = 1 THEN sig_str_landed ELSE 0 END) AS r1_sig_landed,
                  SUM(CASE WHEN round = 1 THEN ctrl_seconds   ELSE 0 END) AS r1_ctrl_secs,
                  SUM(CASE WHEN round = 1 THEN knockdowns     ELSE 0 END) AS r1_knockdowns,
                  -- Late-round output (rounds 3+, cardio / chin proxy)
                  SUM(CASE WHEN round >= 3 THEN sig_str_landed ELSE 0 END) AS late_sig_landed,
                  SUM(CASE WHEN round >= 3 THEN ctrl_seconds   ELSE 0 END) AS late_ctrl_secs,
                  -- Total rounds fought (needed to normalise late-round stats)
                  MAX(round) AS rounds_fought
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
    """Load opening DK/FD lines per fight for the market_nv_a feature.

    Uses OPENING lines (fightodds_*_open) rather than closing lines. Opening lines
    are less efficient — sharp money hasn't moved them yet — so the model learns to
    identify residual edges against the weaker early price rather than the efficient
    closing number. Closing lines are used for edge calculation at bet time via
    evaluate_fight(); training against the open teaches the model what the market
    missed, not what it already corrected.

    market_nv_a = 0.5 when unavailable (neutral prior, pre-2022 fights).
    """
    return pd.read_sql_query(
        """SELECT fight_id,
                  MAX(CASE WHEN book IN ('fightodds_DraftKings_open','bfo_DraftKings','capture_DraftKings') THEN fighter_a_odds END) as dk_a,
                  MAX(CASE WHEN book IN ('fightodds_DraftKings_open','bfo_DraftKings','capture_DraftKings') THEN fighter_b_odds END) as dk_b,
                  MAX(CASE WHEN book IN ('fightodds_FanDuel_open','bfo_FanDuel','capture_FanDuel')          THEN fighter_a_odds END) as fd_a,
                  MAX(CASE WHEN book IN ('fightodds_FanDuel_open','bfo_FanDuel','capture_FanDuel')          THEN fighter_b_odds END) as fd_b
           FROM odds_history
           WHERE fight_id IS NOT NULL
             AND book IN ('fightodds_DraftKings_open','fightodds_FanDuel_open','bfo_DraftKings','bfo_FanDuel',
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


_EMPTY_STATS: dict = {
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
    # Per-round / style features
    "ctrl_per_min": 0.0,        # ground control time per minute fought
    "kd_rate": 0.0,             # knockdowns per minute
    "head_str_pct": 0.50,       # fraction of sig strikes to head
    "leg_str_pct": 0.15,        # fraction of sig strikes to legs
    "fade_rate": 0.0,           # late-round output minus early-round output (neg = fades)
    "late_ctrl_per_min": 0.0,   # control time in rounds 3+ per late-round minute
}


def compute_fighter_rolling_stats(
    fighter_id: str,
    before_date: str,
    prior_fights: list,
    stats_lookup: dict,
) -> dict:
    """Compute rolling stats using pre-filtered, pre-sorted prior fights.

    prior_fights: fight dicts already filtered to before before_date, sorted chronologically.
    stats_lookup: (fight_id, fighter_id) -> stats dict for O(1) access.
    """
    if not prior_fights:
        return dict(_EMPTY_STATS)

    rows = []
    for fight_row in prior_fights:
        fid = fight_row["fight_id"]
        s = stats_lookup.get((fid, fighter_id))
        if s is None:
            continue
        rows.append({
            **s,
            "fight_date":  fight_row["fight_date"],
            "round":       fight_row["round"],
            "time":        fight_row["time"],
            "time_format": fight_row["time_format"],
            "winner_id":   fight_row["winner_id"],
            "method":      fight_row["method"],
        })

    if not rows:
        return dict(_EMPTY_STATS)

    f_stats = pd.DataFrame(rows)

    f_stats["fight_secs"] = f_stats.apply(
        lambda r: parse_time_seconds(r["time"], r["round"], r["time_format"]), axis=1
    )
    f_stats["fight_mins"] = (f_stats["fight_secs"] / 60.0).clip(lower=0.01)

    f_stats["slpm"]    = f_stats["sig_landed"] / f_stats["fight_mins"]
    f_stats["str_acc"] = np.where(f_stats["sig_att"] > 0,
                                   f_stats["sig_landed"] / f_stats["sig_att"], 0.0)
    f_stats["td_avg"]  = f_stats["td_landed"] / f_stats["fight_mins"]
    f_stats["td_acc"]  = np.where(f_stats["td_att"] > 0,
                                   f_stats["td_landed"] / f_stats["td_att"], 0.0)
    f_stats["sub_avg"] = f_stats["sub_att"] / f_stats["fight_mins"]

    f_stats["won"]    = (f_stats["winner_id"] == fighter_id).astype(int)
    f_stats["ko_win"] = ((f_stats["won"] == 1) &
                          f_stats["method"].str.contains("KO|TKO", na=False)).astype(int)
    f_stats["sub_win"] = ((f_stats["won"] == 1) &
                           f_stats["method"].str.contains("Sub", na=False, case=False)).astype(int)
    f_stats["ko_loss"] = ((f_stats["won"] == 0) &
                           f_stats["method"].str.contains("KO|TKO", na=False)).astype(int)

    n     = len(f_stats)
    last3 = f_stats.tail(3)
    last5 = f_stats.tail(5)

    def wrm(col, df): return weighted_mean(df[col].tolist())
    def rrm(col, df): return recency_mean(df[col].tolist())

    slpm_career     = wrm("slpm", f_stats)
    str_acc_career  = wrm("str_acc", f_stats)
    td_avg_career   = wrm("td_avg", f_stats)
    win_rate_career = f_stats["won"].mean()

    slpm_recent     = rrm("slpm", last5)
    str_acc_recent  = rrm("str_acc", last5)
    td_avg_recent   = rrm("td_avg", last5)
    win_rate_recent = last5["won"].mean() if len(last5) >= 2 else win_rate_career

    slpm_trend      = slpm_recent - slpm_career
    str_acc_trend   = str_acc_recent - str_acc_career
    win_rate_trend  = win_rate_recent - win_rate_career

    # --- Style / per-round features ---
    total_sig   = f_stats["sig_landed"].sum()
    total_mins  = f_stats["fight_mins"].sum()
    ctrl_total  = f_stats["ctrl_secs"].sum() if "ctrl_secs" in f_stats.columns else 0
    kd_total    = f_stats["knockdowns"].sum() if "knockdowns" in f_stats.columns else 0
    head_total  = f_stats["head_landed"].sum() if "head_landed" in f_stats.columns else 0
    leg_total   = f_stats["leg_landed"].sum() if "leg_landed" in f_stats.columns else 0

    ctrl_per_min  = ctrl_total / max(total_mins, 0.01) / 60.0
    kd_rate       = kd_total / max(total_mins, 0.01)
    head_str_pct  = head_total / max(total_sig, 1)
    leg_str_pct   = leg_total  / max(total_sig, 1)

    # Fade rate: avg late-round slpm minus avg R1 slpm (negative = fades).
    # Uses per-fight R1 and late-round data from the enriched stats query.
    def _slpm_period(col_landed, col_secs_override=None):
        """slpm for a period given total landed and total seconds columns."""
        landed = f_stats[col_landed].sum() if col_landed in f_stats.columns else 0
        if col_secs_override and col_secs_override in f_stats.columns:
            mins = f_stats[col_secs_override].sum() / 60.0
        else:
            mins = total_mins
        return landed / max(mins, 0.01)

    r1_slpm   = _slpm_period("r1_sig_landed")
    # Late round = R3+; approximate late-round time as fights_with_late_rounds * 5 min/rd
    late_fts  = f_stats[f_stats.get("rounds_fought", pd.Series(dtype=float)) >= 3] \
                if "rounds_fought" in f_stats.columns else pd.DataFrame()
    if len(late_fts) > 0 and "late_sig_landed" in f_stats.columns:
        late_landed = f_stats["late_sig_landed"].sum()
        # Approximate late-round time: each fight that went 3+ rounds contributed roughly
        # (rounds_fought - 2) rounds of late time at 5 min each.
        late_extra_rounds = (f_stats["rounds_fought"].clip(lower=2) - 2).sum() \
                            if "rounds_fought" in f_stats.columns else len(late_fts)
        late_mins = max(late_extra_rounds * 5.0, 0.01)
        late_slpm = late_landed / late_mins
    else:
        late_slpm = r1_slpm  # no late-round data; assume constant output

    fade_rate = late_slpm - r1_slpm   # negative = fades in later rounds

    late_ctrl = f_stats["late_ctrl_secs"].sum() if "late_ctrl_secs" in f_stats.columns else 0
    late_ctrl_per_min = late_ctrl / max(late_mins if len(late_fts) > 0 else 1.0, 0.01) / 60.0

    return {
        "n_fights":        n,
        "slpm_career":     slpm_career,
        "str_acc_career":  str_acc_career,
        "td_avg_career":   td_avg_career,
        "td_acc_career":   wrm("td_acc", f_stats),
        "sub_avg_career":  wrm("sub_avg", f_stats),
        "slpm_L5":         wrm("slpm", last5),
        "str_acc_L5":      wrm("str_acc", last5),
        "td_avg_L5":       wrm("td_avg", last5),
        "slpm_L3":         wrm("slpm", last3),
        "str_acc_L3":      wrm("str_acc", last3),
        "td_avg_L3":       wrm("td_avg", last3),
        "slpm_recent":     slpm_recent,
        "str_acc_recent":  str_acc_recent,
        "td_avg_recent":   td_avg_recent,
        "slpm_trend":      slpm_trend,
        "str_acc_trend":   str_acc_trend,
        "win_rate_trend":  win_rate_trend,
        "recent_win_rate": win_rate_recent,
        "last_fight_won":  int(f_stats.iloc[-1]["won"]) if n > 0 else 0,
        "avg_fight_secs":  f_stats["fight_secs"].mean(),
        "ko_finish_rate":  f_stats["ko_win"].sum() / max(n, 1),
        "sub_finish_rate": f_stats["sub_win"].sum() / max(n, 1),
        "ko_susceptibility": f_stats["ko_loss"].sum() / max(n, 1),
        "win_streak":      _current_streak(f_stats["won"].tolist()),
        "days_since_last": _days_since(f_stats["fight_date"].tolist(), before_date),
        # Per-round / style
        "ctrl_per_min":       ctrl_per_min,
        "kd_rate":            kd_rate,
        "head_str_pct":       head_str_pct,
        "leg_str_pct":        leg_str_pct,
        "fade_rate":          fade_rate,
        "late_ctrl_per_min":  late_ctrl_per_min,
    }


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
        ref  = datetime.strptime(reference, "%Y-%m-%d")
        return max(0.0, (ref - last).days)
    except Exception:
        return 365.0


def compute_opponent_sapm(
    fighter_id: str,
    prior_fights: list,
    stats_lookup: dict,
) -> float:
    """Average strikes absorbed per minute — from opponent's landed strikes."""
    if not prior_fights:
        return 4.0

    sapm_vals = []
    for fight_row in prior_fights:
        fid    = fight_row["fight_id"]
        opp_id = (fight_row["fighter_b_id"] if fight_row["fighter_a_id"] == fighter_id
                  else fight_row["fighter_a_id"])
        opp    = stats_lookup.get((fid, opp_id))
        if opp is None:
            continue
        fight_secs = parse_time_seconds(fight_row["time"], fight_row["round"],
                                        fight_row["time_format"])
        fight_mins = max(fight_secs / 60.0, 0.01)
        sapm_vals.append(opp["sig_landed"] / fight_mins)

    return weighted_mean(sapm_vals) if sapm_vals else 4.0


def compute_td_defense(
    fighter_id: str,
    prior_fights: list,
    stats_lookup: dict,
) -> float:
    """TD defense = 1 - (opp TDs landed / opp TDs attempted)."""
    if not prior_fights:
        return 0.5

    opp_td_land = 0
    opp_td_att  = 0
    for fight_row in prior_fights:
        fid    = fight_row["fight_id"]
        opp_id = (fight_row["fighter_b_id"] if fight_row["fighter_a_id"] == fighter_id
                  else fight_row["fighter_a_id"])
        opp    = stats_lookup.get((fid, opp_id))
        if opp is None:
            continue
        opp_td_land += opp["td_landed"]
        opp_td_att  += opp["td_att"]

    return 1.0 - (opp_td_land / max(opp_td_att, 1))


def compute_avg_opp_elo(
    fighter_id: str,
    prior_fights: list,
    elo_map: dict,
) -> float:
    """Recency-weighted average ELO of opponents faced — proxy for strength of schedule."""
    if not prior_fights:
        return 1500.0
    elos = []
    for fight_row in prior_fights:
        opp_id = (fight_row["fighter_b_id"] if fight_row["fighter_a_id"] == fighter_id
                  else fight_row["fighter_a_id"])
        elos.append(elo_map.get((opp_id, fight_row["fight_id"]), 1500.0))
    return weighted_mean(elos)


def build_feature_matrix(con: sqlite3.Connection) -> pd.DataFrame:
    print("Loading data from DB...")
    fights = load_fights(con)
    stats  = load_stats(con)
    elo    = load_elo(con)
    market = load_market_odds(con)

    print(f"  {len(fights)} fights, {len(stats)} stat rows, "
          f"{len(elo)} ELO rows, {len(market)} with market odds")

    # --- Build O(1) lookup structures upfront ---

    # Per-fighter fight lists sorted chronologically.
    # Each fighter's list covers ALL their appearances (red and blue corner).
    fighter_fights_map: dict[str, list] = defaultdict(list)
    for _, row in fights.iterrows():
        frow = row.to_dict()
        fighter_fights_map[row["fighter_a_id"]].append(frow)
        fighter_fights_map[row["fighter_b_id"]].append(frow)
    for fid in fighter_fights_map:
        fighter_fights_map[fid].sort(key=lambda r: r["fight_date"])

    # Stats lookup: (fight_id, fighter_id) -> stats dict
    stats_lookup: dict[tuple, dict] = {
        (r["fight_id"], r["fighter_id"]): r.to_dict()
        for _, r in stats.iterrows()
    }

    # ELO lookup: (fighter_id, fight_id) -> elo_before
    elo_map: dict[tuple, float] = {
        (r["fighter_id"], r["fight_id"]): r["elo_before"]
        for _, r in elo.iterrows()
    }

    # --- Market prior: devig one book's paired line (never mix books per side) ---
    def best_book_line(row) -> tuple:
        """Return (odds_a, odds_b) from one complete book line.

        Using a single book's paired line ensures the no-vig calculation is correct.
        Mixing the best A-side from DK with the best B-side from FD creates a
        synthetic line that was never priced, biasing the devigged probability.
        DK preferred; FD as fallback.
        """
        def valid(o):
            return o is not None and not pd.isna(o) and abs(int(o)) < 5000
        dk_a, dk_b = row.get("dk_a"), row.get("dk_b")
        if valid(dk_a) and valid(dk_b):
            return int(dk_a), int(dk_b)
        fd_a, fd_b = row.get("fd_a"), row.get("fd_b")
        if valid(fd_a) and valid(fd_b):
            return int(fd_a), int(fd_b)
        return None, None

    def shin_nv_a(odds_a, odds_b) -> float:
        if odds_a is None or odds_b is None:
            return 0.5
        try:
            from betting.devig import devig_shin
            pa, _ = devig_shin(odds_a, odds_b)
            return float(pa)
        except Exception:
            return 0.5

    market_nv_map: dict[str, float] = {}
    for _, mrow in market.iterrows():
        fid    = mrow["fight_id"]
        oa, ob = best_book_line(mrow)
        market_nv_map[fid] = shin_nv_a(oa, ob)

    rows  = []
    total = len(fights)

    for i, fight in fights.iterrows():
        if i % 500 == 0:
            print(f"  Processing fight {i}/{total}...")

        fa_id = fight["fighter_a_id"]
        fb_id = fight["fighter_b_id"]
        date  = fight["fight_date"]

        # Pre-filter prior fights once; reuse for all per-fighter computations.
        fa_prior = [f for f in fighter_fights_map.get(fa_id, []) if f["fight_date"] < date]
        fb_prior = [f for f in fighter_fights_map.get(fb_id, []) if f["fight_date"] < date]

        fa_stats = compute_fighter_rolling_stats(fa_id, date, fa_prior, stats_lookup)
        fb_stats = compute_fighter_rolling_stats(fb_id, date, fb_prior, stats_lookup)

        fa_sapm    = compute_opponent_sapm(fa_id, fa_prior, stats_lookup)
        fb_sapm    = compute_opponent_sapm(fb_id, fb_prior, stats_lookup)
        fa_td_def  = compute_td_defense(fa_id, fa_prior, stats_lookup)
        fb_td_def  = compute_td_defense(fb_id, fb_prior, stats_lookup)
        fa_opp_elo = compute_avg_opp_elo(fa_id, fa_prior, elo_map)
        fb_opp_elo = compute_avg_opp_elo(fb_id, fb_prior, elo_map)

        def age_at(dob_str, date_str):
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d")
                d   = datetime.strptime(date_str, "%Y-%m-%d")
                return (d - dob).days / 365.25
            except Exception:
                return None

        age_a = age_at(fight["dob_a"], date)
        age_b = age_at(fight["dob_b"], date)

        fa_elo = elo_map.get((fa_id, fight["fight_id"]), 1500.0)
        fb_elo = elo_map.get((fb_id, fight["fight_id"]), 1500.0)

        stance_a = str(fight["stance_a"]).lower()
        stance_b = str(fight["stance_b"]).lower()
        both_orthodox     = int(stance_a == "orthodox" and stance_b == "orthodox")
        a_southpaw        = int(stance_a == "southpaw" and stance_b == "orthodox")
        b_southpaw        = int(stance_b == "southpaw" and stance_a == "orthodox")

        method    = str(fight.get("method", "")).upper()
        method_ko  = int("KO" in method or "TKO" in method)
        method_sub = int("SUB" in method)
        method_dec = int("DEC" in method or "DECISION" in method)

        target = 1 if fight["winner_id"] == fa_id else 0

        # Age × decline: fires when a fighter is old AND trending worse on both
        # striking output (slpm_trend) and results (win_rate_trend).
        # win_rate_trend is scaled by 4 to bring it into the same magnitude as slpm_trend
        # before averaging. Using both signals is more robust than slpm alone — it catches
        # wrestlers or grapplers whose striking didn't decline but whose win rate did.
        def _age_decline(age, slpm_trend, win_rate_trend) -> float:
            if not age:
                return 0.0
            age_factor = max(0.0, (age - 30) / 10.0)
            composite  = (slpm_trend + win_rate_trend * 4.0) / 2.0
            return age_factor * min(composite, 0.0)

        fa_age_decline = _age_decline(
            age_a, fa_stats.get("slpm_trend", 0.0), fa_stats.get("win_rate_trend", 0.0)
        )
        fb_age_decline = _age_decline(
            age_b, fb_stats.get("slpm_trend", 0.0), fb_stats.get("win_rate_trend", 0.0)
        )

        row = {
            "fight_id":   fight["fight_id"],
            "fight_date": date,
            "name_a":     fight["name_a"],
            "name_b":     fight["name_b"],
            "weight_class":   fight["weight_class"],
            "is_title_fight": fight["is_title_fight"],
            "is_5_round":     int(str(fight["time_format"]).startswith("5")),
            "both_orthodox":          both_orthodox,
            "a_southpaw_vs_orthodox": a_southpaw,
            "b_southpaw_vs_orthodox": b_southpaw,

            # Differentials (A − B)
            "slpm_career_diff":       fa_stats["slpm_career"]    - fb_stats["slpm_career"],
            "slpm_L5_diff":           fa_stats["slpm_L5"]        - fb_stats["slpm_L5"],
            "slpm_L3_diff":           fa_stats["slpm_L3"]        - fb_stats["slpm_L3"],
            "slpm_recent_diff":       fa_stats["slpm_recent"]    - fb_stats["slpm_recent"],
            "str_acc_recent_diff":    fa_stats["str_acc_recent"] - fb_stats["str_acc_recent"],
            "td_avg_recent_diff":     fa_stats["td_avg_recent"]  - fb_stats["td_avg_recent"],
            "slpm_trend_diff":        fa_stats["slpm_trend"]     - fb_stats["slpm_trend"],
            "str_acc_trend_diff":     fa_stats["str_acc_trend"]  - fb_stats["str_acc_trend"],
            "win_rate_trend_diff":    fa_stats["win_rate_trend"] - fb_stats["win_rate_trend"],
            "recent_win_rate_diff":   fa_stats["recent_win_rate"] - fb_stats["recent_win_rate"],
            "last_fight_won_diff":    fa_stats["last_fight_won"] - fb_stats["last_fight_won"],
            "age_decline_diff":       fa_age_decline - fb_age_decline,
            "sapm_diff":              fa_sapm - fb_sapm,
            "net_strike_diff":        (fa_stats["slpm_career"] - fa_sapm) - (fb_stats["slpm_career"] - fb_sapm),
            "str_acc_career_diff":    fa_stats["str_acc_career"] - fb_stats["str_acc_career"],
            "str_acc_L5_diff":        fa_stats["str_acc_L5"]    - fb_stats["str_acc_L5"],
            "td_avg_career_diff":     fa_stats["td_avg_career"] - fb_stats["td_avg_career"],
            "td_avg_L5_diff":         fa_stats["td_avg_L5"]     - fb_stats["td_avg_L5"],
            "td_acc_career_diff":     fa_stats["td_acc_career"] - fb_stats["td_acc_career"],
            "td_def_diff":            fa_td_def - fb_td_def,
            "sub_avg_diff":           fa_stats["sub_avg_career"] - fb_stats["sub_avg_career"],
            "ko_finish_rate_diff":    fa_stats["ko_finish_rate"]   - fb_stats["ko_finish_rate"],
            "sub_finish_rate_diff":   fa_stats["sub_finish_rate"]  - fb_stats["sub_finish_rate"],
            "ko_susceptibility_diff": fa_stats["ko_susceptibility"] - fb_stats["ko_susceptibility"],
            "avg_fight_secs_diff":    fa_stats["avg_fight_secs"] - fb_stats["avg_fight_secs"],
            "experience_diff":        fa_stats["n_fights"]   - fb_stats["n_fights"],
            "win_streak_diff":        fa_stats["win_streak"] - fb_stats["win_streak"],
            "elo_diff":               fa_elo - fb_elo,
            # Strength of schedule: recency-weighted avg opponent ELO before each fight.
            # Positive = fighter A faced tougher competition on average.
            "avg_opp_elo_diff":       fa_opp_elo - fb_opp_elo,
            "reach_diff":             (fight["reach_a"] or 0) - (fight["reach_b"] or 0),
            "height_diff":            (fight["height_a"] or 0) - (fight["height_b"] or 0),
            "age_diff":               (age_a or 0) - (age_b or 0) if age_a and age_b else 0,
            # Not differenced: a 365-day layoff for A is not equivalent to B having 365 more
            # rest days. The model learns the effect of absolute layoff length independently
            # for each fighter rather than cancelling them out in a differential.
            "days_since_last_a":   fa_stats["days_since_last"],
            "days_since_last_b":   fb_stats["days_since_last"],
            # Per-round and style features
            "ctrl_per_min_diff":      fa_stats["ctrl_per_min"]      - fb_stats["ctrl_per_min"],
            "kd_rate_diff":           fa_stats["kd_rate"]           - fb_stats["kd_rate"],
            "head_str_pct_diff":      fa_stats["head_str_pct"]      - fb_stats["head_str_pct"],
            "leg_str_pct_diff":       fa_stats["leg_str_pct"]       - fb_stats["leg_str_pct"],
            "fade_rate_diff":         fa_stats["fade_rate"]         - fb_stats["fade_rate"],
            "late_ctrl_per_min_diff": fa_stats["late_ctrl_per_min"] - fb_stats["late_ctrl_per_min"],
            # Market prior: Shin-devigged no-vig prob for fighter A from one book's paired
            # line (DK preferred, FD fallback). 0.5 = no market data (pre-2022 fights).
            "market_nv_a": market_nv_map.get(fight["fight_id"], 0.5),

            "method_ko":  method_ko,
            "method_sub": method_sub,
            "method_dec": method_dec,
            "target":     target,
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
