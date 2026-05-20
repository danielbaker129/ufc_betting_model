"""
UFC Betting Model Dashboard — Streamlit app.
Tabs: Next Event | Backtest Results | Fighter Lookup
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, PROCESSED_DIR, MODELS_DIR, MIN_EDGE, MAX_EDGE, KELLY_FRAC, MAX_UNITS
from betting.devig import no_vig_probs, devig_shin
from betting.kelly import bet_summary, to_decimal, kelly_units
from betting.edge import evaluate_fight

st.set_page_config(page_title="UFC Betting Model", page_icon="🥊", layout="wide")

FEAT_COLS = [
    "slpm_career_diff", "slpm_L5_diff", "slpm_L3_diff",
    "sapm_diff", "net_strike_diff",
    "str_acc_career_diff", "str_acc_L5_diff",
    "td_avg_career_diff", "td_avg_L5_diff",
    "td_acc_career_diff", "td_def_diff", "sub_avg_diff",
    "ko_finish_rate_diff", "sub_finish_rate_diff", "ko_susceptibility_diff",
    "avg_fight_secs_diff", "experience_diff", "win_streak_diff",
    "elo_diff", "reach_diff", "height_diff", "age_diff",
    "days_since_last_a", "days_since_last_b",
    "is_title_fight", "is_5_round",
    "both_orthodox", "a_southpaw_vs_orthodox", "b_southpaw_vs_orthodox",
]


def _models_signature() -> str:
    """Cache-bust when .pkl files are added or updated on disk."""
    parts = []
    for name in ("moneyline", "method", "rounds", "props"):
        p = MODELS_DIR / f"{name}.pkl"
        if p.is_file():
            stt = p.stat()
            parts.append(f"{name}:{stt.st_size}:{int(stt.st_mtime)}")
        else:
            parts.append(f"{name}:missing")
    return "|".join(parts)


@st.cache_resource
def load_models(signature: str):
    models = {}
    errors = []
    for name in ("moneyline", "method", "rounds", "props"):
        p = MODELS_DIR / f"{name}.pkl"
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            errors.append(f"{name}: {e}")
    return models, errors


@st.cache_data(ttl=120)
def load_backtest():
    p = PROCESSED_DIR / "backtest_results.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame()


def db_available() -> bool:
    """True only when the scraped UFC database is present (not an empty stub)."""
    if not DB_PATH.is_file() or DB_PATH.stat().st_size < 10_000:
        return False
    try:
        con = sqlite3.connect(DB_PATH)
        ok = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fights'"
        ).fetchone() is not None
        con.close()
        return ok
    except sqlite3.Error:
        return False


def get_db():
    if not db_available():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Scrape locally or include data/ufc.db in the GitHub repo for Cloud deploy."
        )
    return sqlite3.connect(DB_PATH)


def _db_signature() -> str:
    if not db_available():
        return "missing"
    stt = DB_PATH.stat()
    return f"{stt.st_size}:{int(stt.st_mtime)}"


def deployment_missing() -> list[str]:
    missing = []
    if not db_available():
        missing.append("`data/ufc.db`")
    if not any((MODELS_DIR / f"{n}.pkl").exists() for n in ("moneyline", "method", "rounds", "props")):
        missing.append("`models/*.pkl`")
    if not (PROCESSED_DIR / "backtest_results.csv").is_file():
        missing.append("`data/processed/backtest_results.csv`")
    return missing


def _flip_features(features: dict) -> dict:
    """Return a copy of features with A and B sides swapped.

    The model was trained with fighter_a = UFC red corner, who wins 64% of fights.
    For upcoming fights the Odds API assigns fighter_a arbitrarily, so we average
    both orderings to cancel the positional prior.
    """
    flipped = {}
    for k, v in features.items():
        if k.endswith("_diff"):
            flipped[k] = -v
        elif k.endswith("_a"):
            flipped[k[:-2] + "_b"] = v
        elif k.endswith("_b"):
            flipped[k[:-2] + "_a"] = v
        else:
            flipped[k] = v
    # market_nv_a in the flipped view = market prob for the original fighter_b
    if "market_nv_a" in features:
        flipped["market_nv_a"] = 1.0 - features["market_nv_a"]
    return flipped


def predict_fight(features: dict, models: dict) -> dict:
    # Missing feature keys default to 0.0 — for differential features this means
    # "no difference", which matches how the model was trained (0.5 for market_nv_a
    # when odds unavailable, 0 for stats on fighters with no fight history).
    result = {}
    X = np.array([[features.get(c, 0.0) for c in FEAT_COLS]])

    if "moneyline" in models:
        m = models["moneyline"]
        cols = m["feature_cols"]
        Xm = np.array([[features.get(c, 0.0) for c in cols]])
        prob = m["model"].predict_proba(Xm)[0, 1]
        result["prob_a"] = float(prob)
        result["prob_b"] = float(1 - prob)

    if "method" in models:
        m = models["method"]
        cols = m["feature_cols"]
        Xm = np.array([[features.get(c, 0.0) for c in cols]])
        probs = m["model"].predict_proba(Xm)[0]
        le = m["label_encoder"]
        result["method"] = {cls: float(p) for cls, p in zip(le.classes_, probs)}

    if "rounds" in models:
        m = models["rounds"]
        cols = m["feature_cols"]
        Xm = np.array([[features.get(c, 0.0) for c in cols]])
        for tgt, mdl in m["models"].items():
            result[f"rounds_{tgt}"] = float(mdl.predict_proba(Xm)[0, 1])

    return result


# ─── TAB 1: NEXT EVENT ────────────────────────────────────────────────────────

def tab_next_event(models):
    st.header("Upcoming UFC Cards")
    st.caption("All fights grouped by event · 🟢 = model has edge (1u = $10) · ⚪ = no edge / pass")

    try:
        from betting.odds_fetcher import get_upcoming_fights
        with st.spinner("Fetching live odds..."):
            fights = get_upcoming_fights()
    except Exception as e:
        st.warning(f"Could not fetch live odds: {e}")
        fights = []

    if not fights:
        st.info("No upcoming UFC fights found. Either the event is too far out or no odds posted yet.")
        with st.expander("Manual fight entry"):
            manual_fight_form(models)
        return

    # Build full fight data for every fight on every card
    from datetime import timezone, timedelta
    ET = timezone(timedelta(hours=-4))  # EDT (UTC-4); covers May-Nov UFC season

    all_rows = []
    for fight in fights:
        fa, fb = fight["fighter_a"], fight["fighter_b"]
        commence = fight.get("commence_time", "")
        # Convert UTC → ET before extracting date — prevents midnight UTC splits
        # e.g. 2am UTC June 15 = 10pm ET June 14 (same event night)
        try:
            from datetime import datetime as _dt
            utc_dt = _dt.fromisoformat(commence.replace("Z", "+00:00"))
            date = utc_dt.astimezone(ET).strftime("%Y-%m-%d")
        except Exception:
            date = commence[:10]

        # Per-book odds
        dk_a  = fight.get("draftkings_a")
        dk_b  = fight.get("draftkings_b")
        fd_a  = fight.get("fanduel_a")
        fd_b  = fight.get("fanduel_b")

        # Best available line across DK/FD (line-shop)
        odds_a = fight["odds_a"]
        odds_b = fight["odds_b"]
        best_book_a = fight.get("best_book_a", "—")
        best_book_b = fight.get("best_book_b", "—")

        feats = lookup_features(fa, fb, fight_date=date)
        nv_a, nv_b = no_vig_probs(odds_a, odds_b)
        # Inject real market no-vig prob — #1 model feature, was hardcoded to 0.5
        if feats:
            feats["market_nv_a"] = nv_a
        pred  = predict_fight(feats, models) if (feats and models) else {}

        prob_a = pred.get("prob_a", nv_a)
        prob_b = pred.get("prob_b", nv_b)
        method = pred.get("method", {})
        top_method = max(method, key=method.get) if method else "—"

        rec = evaluate_fight(prob_a, prob_b, odds_a, odds_b)

        if rec["bet"]:
            play_side = fa if rec["bet_side"] == "a" else fb
            play_book = best_book_a if rec["bet_side"] == "a" else best_book_b
            play_rec  = rec
        else:
            play_side, play_book, play_rec = None, None, None

        nv_a, nv_b = rec["nv_a"], rec["nv_b"]

        def fmt_odds(o): return f"{o:+d}" if o is not None else "—"

        all_rows.append({
            "_date": date,
            "_commence": commence,
            "Fighter A":  fa,
            "DK A":       fmt_odds(dk_a),
            "FD A":       fmt_odds(fd_a),
            "Model A%":   f"{prob_a*100:.0f}%",
            "Fighter B":  fb,
            "DK B":       fmt_odds(dk_b),
            "FD B":       fmt_odds(fd_b),
            "Model B%":   f"{prob_b*100:.0f}%",
            "Method":     top_method,
            "Play":       play_side or "—",
            "Book":       play_book or "—",
            "Edge":       f"{rec['model_edge']*100:+.1f}%",
            "Units":      f"{play_rec['units']:.1f}u" if play_rec and play_rec["units"] > 0 else "—",
            "Bet":        f"${play_rec['dollars']:.0f}" if play_rec and play_rec["dollars"] > 0 else "PASS",
            "_has_edge":  play_rec is not None,
            "_units":     play_rec["units"] if play_rec else 0,
        })

    # Group by date, merging consecutive dates (handles events that run past midnight)
    from collections import defaultdict
    raw_by_date = defaultdict(list)
    for row in all_rows:
        raw_by_date[row["_date"]].append(row)

    # Merge any date into the previous day if they're adjacent (same event split by midnight)
    by_date = {}
    for date in sorted(raw_by_date.keys()):
        prev = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
        if prev in by_date:
            by_date[prev].extend(raw_by_date[date])
        else:
            by_date[date] = raw_by_date[date]

    # Summary banner
    total_plays = sum(1 for r in all_rows if r["_has_edge"])
    total_units = sum(r["_units"] for r in all_rows if r["_has_edge"])
    if total_plays:
        st.success(f"**{total_plays} recommended play{'s' if total_plays != 1 else ''}** across {len(by_date)} event{'s' if len(by_date) != 1 else ''} · Total exposure: {total_units:.1f}u = **${total_units*10:.0f}**")
    else:
        st.info("No edges found on any upcoming card.")

    display_cols = ["Fighter A", "DK A", "FD A", "Model A%",
                    "Fighter B", "DK B", "FD B", "Model B%",
                    "Method", "Play", "Book", "Edge", "Units", "Bet"]

    def style_next_event(row):
        styles = [""] * len(row)
        idx = list(row.index)
        is_play = row.get("Bet", "PASS") != "PASS"
        if is_play:
            for col in ["Play", "Book", "Edge", "Units", "Bet"]:
                if col in idx:
                    styles[idx.index(col)] = "color: #00c853; font-weight: bold"
        return styles

    for date in sorted(by_date.keys()):
        card_rows = by_date[date]
        card_plays = sum(1 for r in card_rows if r["_has_edge"])
        total_card_units = sum(r["_units"] for r in card_rows if r["_has_edge"])
        label = (f"📅 {date}  —  {len(card_rows)} fights  |  "
                 f"{'🟢 ' + str(card_plays) + ' plays · ' + f'{total_card_units:.1f}u = ${total_card_units*10:.0f}' if card_plays else '⚪ no plays'}")
        with st.expander(label, expanded=True):
            df_card = pd.DataFrame(card_rows)[display_cols]
            st.dataframe(
                df_card.style.apply(style_next_event, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Fighter A": st.column_config.TextColumn("Fighter A", width="medium"),
                    "Fighter B": st.column_config.TextColumn("Fighter B", width="medium"),
                    "Play":      st.column_config.TextColumn("Play",      width="medium"),
                    "DK A":  st.column_config.TextColumn("DK A",  width="small"),
                    "FD A":  st.column_config.TextColumn("FD A",  width="small"),
                    "DK B":  st.column_config.TextColumn("DK B",  width="small"),
                    "FD B":  st.column_config.TextColumn("FD B",  width="small"),
                },
            )


@st.cache_data(ttl=3600)
def _load_pipeline_data(db_signature: str):
    """Load fights, stats, elo and build O(1) lookup structures — cached per DB snapshot."""
    if db_signature == "missing" or not db_available():
        return None, None, None, None, None, None
    from pipeline.features import load_fights, load_stats, load_elo
    from collections import defaultdict
    con = get_db()
    try:
        fights = load_fights(con)
        stats  = load_stats(con)
        elo    = load_elo(con)
    finally:
        con.close()

    # Build once, reuse for every lookup_features call this session.
    fighter_fights_map: dict = defaultdict(list)
    for _, row in fights.iterrows():
        frow = row.to_dict()
        fighter_fights_map[row["fighter_a_id"]].append(frow)
        fighter_fights_map[row["fighter_b_id"]].append(frow)
    for fid in fighter_fights_map:
        fighter_fights_map[fid].sort(key=lambda r: r["fight_date"])

    stats_lookup: dict = {
        (r["fight_id"], r["fighter_id"]): r.to_dict()
        for _, r in stats.iterrows()
    }
    elo_map: dict = {
        (r["fighter_id"], r["fight_id"]): r["elo_before"]
        for _, r in elo.iterrows()
    }

    return fights, stats, elo, fighter_fights_map, stats_lookup, elo_map


def _find_fighter_id(name: str, fights_df: pd.DataFrame) -> str | None:
    """Find fighter_id by matching last name against fights_df."""
    import re
    last = re.sub(r"[^a-z]", "", name.split()[-1].lower())
    for col in ("name_a", "name_b"):
        id_col = "fighter_a_id" if col == "name_a" else "fighter_b_id"
        mask = fights_df[col].str.lower().str.replace(r"[^a-z]", "", regex=True).str.contains(last, na=False)
        if mask.any():
            return fights_df[mask].iloc[-1][id_col]
    return None


def lookup_features(fighter_a: str, fighter_b: str, _feat_df=None,
                    fight_date: str | None = None,
                    is_title: int = 0, is_5rd: int = 0,
                    stance_a: str = "", stance_b: str = "") -> dict:
    """
    Build differential features for an upcoming fight using the same pipeline
    functions as features.py. Works for any two fighters regardless of whether
    they've met before.
    """
    from pipeline.features import (
        compute_fighter_rolling_stats, compute_opponent_sapm,
        compute_td_defense, compute_avg_opp_elo,
    )
    from datetime import datetime as _dt

    today = fight_date or _dt.today().strftime("%Y-%m-%d")
    fights, stats, elo, fighter_fights_map, stats_lookup, elo_map = _load_pipeline_data(_db_signature())
    if fights is None:
        return {}

    fa_id = _find_fighter_id(fighter_a, fights)
    fb_id = _find_fighter_id(fighter_b, fights)
    if not fa_id or not fb_id:
        return {}

    fa_prior = [f for f in fighter_fights_map.get(fa_id, []) if f["fight_date"] < today]
    fb_prior = [f for f in fighter_fights_map.get(fb_id, []) if f["fight_date"] < today]

    fa_s      = compute_fighter_rolling_stats(fa_id, today, fa_prior, stats_lookup)
    fb_s      = compute_fighter_rolling_stats(fb_id, today, fb_prior, stats_lookup)
    fa_sapm   = compute_opponent_sapm(fa_id, fa_prior, stats_lookup)
    fb_sapm   = compute_opponent_sapm(fb_id, fb_prior, stats_lookup)
    fa_td_def = compute_td_defense(fa_id, fa_prior, stats_lookup)
    fb_td_def = compute_td_defense(fb_id, fb_prior, stats_lookup)
    fa_opp_elo = compute_avg_opp_elo(fa_id, fa_prior, elo_map)
    fb_opp_elo = compute_avg_opp_elo(fb_id, fb_prior, elo_map)

    # ELO — latest pre-fight snapshot for each fighter
    fa_elo = elo_map.get((fa_id, fa_prior[-1]["fight_id"]), 1500.0) if fa_prior else 1500.0
    fb_elo = elo_map.get((fb_id, fb_prior[-1]["fight_id"]), 1500.0) if fb_prior else 1500.0

    # Physical attributes
    con = get_db()
    fa_row = pd.read_sql_query("SELECT height_inches, reach_inches, dob, stance FROM fighters WHERE fighter_id=?", con, params=(fa_id,))
    fb_row = pd.read_sql_query("SELECT height_inches, reach_inches, dob, stance FROM fighters WHERE fighter_id=?", con, params=(fb_id,))
    con.close()

    def age_at(dob_str, date_str):
        try:
            dob = _dt.strptime(dob_str, "%Y-%m-%d")
            d   = _dt.strptime(date_str, "%Y-%m-%d")
            return (d - dob).days / 365.25
        except Exception:
            return None

    fa_phys = fa_row.iloc[0] if not fa_row.empty else {}
    fb_phys = fb_row.iloc[0] if not fb_row.empty else {}
    reach_a  = fa_phys.get("reach_inches") or 72
    reach_b  = fb_phys.get("reach_inches") or 72
    height_a = fa_phys.get("height_inches") or 70
    height_b = fb_phys.get("height_inches") or 70
    age_a = age_at(str(fa_phys.get("dob", "")), today)
    age_b = age_at(str(fb_phys.get("dob", "")), today)
    sa = stance_a or str(fa_phys.get("stance", "")).lower()
    sb = stance_b or str(fb_phys.get("stance", "")).lower()

    # Must match features.py: composite of slpm_trend + win_rate_trend
    def age_decline(age, slpm_trend, win_rate_trend):
        if not age:
            return 0.0
        age_factor = max(0.0, (age - 30) / 10.0)
        composite  = (slpm_trend + win_rate_trend * 4.0) / 2.0
        return age_factor * min(composite, 0.0)

    return {
        "slpm_career_diff":       fa_s["slpm_career"] - fb_s["slpm_career"],
        "slpm_L5_diff":           fa_s["slpm_L5"] - fb_s["slpm_L5"],
        "slpm_L3_diff":           fa_s["slpm_L3"] - fb_s["slpm_L3"],
        "str_acc_L5_diff":        fa_s["str_acc_L5"] - fb_s["str_acc_L5"],
        "td_avg_L5_diff":         fa_s["td_avg_L5"] - fb_s["td_avg_L5"],
        "sapm_diff":              fa_sapm - fb_sapm,
        "net_strike_diff":        (fa_s["slpm_career"] - fa_sapm) - (fb_s["slpm_career"] - fb_sapm),
        "str_acc_career_diff":    fa_s["str_acc_career"] - fb_s["str_acc_career"],
        "td_avg_career_diff":     fa_s["td_avg_career"] - fb_s["td_avg_career"],
        "td_acc_career_diff":     fa_s["td_acc_career"] - fb_s["td_acc_career"],
        "td_def_diff":            fa_td_def - fb_td_def,
        "sub_avg_diff":           fa_s["sub_avg_career"] - fb_s["sub_avg_career"],
        "ko_finish_rate_diff":    fa_s["ko_finish_rate"] - fb_s["ko_finish_rate"],
        "sub_finish_rate_diff":   fa_s["sub_finish_rate"] - fb_s["sub_finish_rate"],
        "ko_susceptibility_diff": fa_s["ko_susceptibility"] - fb_s["ko_susceptibility"],
        "avg_fight_secs_diff":    fa_s["avg_fight_secs"] - fb_s["avg_fight_secs"],
        "experience_diff":        fa_s["n_fights"] - fb_s["n_fights"],
        "win_streak_diff":        fa_s["win_streak"] - fb_s["win_streak"],
        "elo_diff":               float(fa_elo) - float(fb_elo),
        "avg_opp_elo_diff":       float(fa_opp_elo) - float(fb_opp_elo),
        "reach_diff":             float(reach_a or 0) - float(reach_b or 0),
        "height_diff":            float(height_a or 0) - float(height_b or 0),
        "age_diff":               ((age_a or 0) - (age_b or 0)) if age_a and age_b else 0.0,
        "days_since_last_a":      fa_s["days_since_last"],
        "days_since_last_b":      fb_s["days_since_last"],
        "is_title_fight":         is_title,
        "is_5_round":             is_5rd,
        "both_orthodox":          int(sa == "orthodox" and sb == "orthodox"),
        "a_southpaw_vs_orthodox": int(sa == "southpaw" and sb == "orthodox"),
        "b_southpaw_vs_orthodox": int(sb == "southpaw" and sa == "orthodox"),
        "slpm_recent_diff":       fa_s["slpm_recent"] - fb_s["slpm_recent"],
        "str_acc_recent_diff":    fa_s["str_acc_recent"] - fb_s["str_acc_recent"],
        "td_avg_recent_diff":     fa_s["td_avg_recent"] - fb_s["td_avg_recent"],
        "slpm_trend_diff":        fa_s["slpm_trend"] - fb_s["slpm_trend"],
        "str_acc_trend_diff":     fa_s["str_acc_trend"] - fb_s["str_acc_trend"],
        "win_rate_trend_diff":    fa_s["win_rate_trend"] - fb_s["win_rate_trend"],
        "recent_win_rate_diff":   fa_s["recent_win_rate"] - fb_s["recent_win_rate"],
        "last_fight_won_diff":    fa_s["last_fight_won"] - fb_s["last_fight_won"],
        "age_decline_diff":       age_decline(age_a, fa_s.get("slpm_trend", 0.0), fa_s.get("win_rate_trend", 0.0))
                                - age_decline(age_b, fb_s.get("slpm_trend", 0.0), fb_s.get("win_rate_trend", 0.0)),
        # Per-round / style features
        "ctrl_per_min_diff":      fa_s.get("ctrl_per_min", 0.0)      - fb_s.get("ctrl_per_min", 0.0),
        "kd_rate_diff":           fa_s.get("kd_rate", 0.0)           - fb_s.get("kd_rate", 0.0),
        "head_str_pct_diff":      fa_s.get("head_str_pct", 0.5)      - fb_s.get("head_str_pct", 0.5),
        "leg_str_pct_diff":       fa_s.get("leg_str_pct", 0.15)      - fb_s.get("leg_str_pct", 0.15),
        "fade_rate_diff":         fa_s.get("fade_rate", 0.0)         - fb_s.get("fade_rate", 0.0),
        "late_ctrl_per_min_diff": fa_s.get("late_ctrl_per_min", 0.0) - fb_s.get("late_ctrl_per_min", 0.0),
        "market_nv_a":            0.5,  # overwritten by caller with live no-vig prob
    }


def manual_fight_form(models):
    col1, col2 = st.columns(2)
    with col1:
        fa = st.text_input("Fighter A name")
        odds_a = st.number_input("Fighter A odds (American)", value=-150)
    with col2:
        fb = st.text_input("Fighter B name")
        odds_b = st.number_input("Fighter B odds (American)", value=130)
    if st.button("Get Prediction") and fa and fb:
        feats = lookup_features(fa, fb)
        if feats:
            pred   = predict_fight(feats, models)
            prob_a = pred.get("prob_a", 0.5)
            rec    = evaluate_fight(prob_a, 1 - prob_a, int(odds_a), int(odds_b))
            st.write(f"**{fa}**: {prob_a*100:.1f}%  edge={rec['nv_a'] - prob_a:+.1%}")
            st.write(f"**{fb}**: {(1-prob_a)*100:.1f}%  edge={rec['nv_b'] - (1-prob_a):+.1%}")
            if rec["bet"]:
                name = fa if rec["bet_side"] == "a" else fb
                st.success(f"Play: **{name}**  {rec['edge']*100:+.1f}% edge  {rec['units']:.1f}u  ${rec['dollars']:.0f}")
            else:
                st.info("No edge — PASS")
        else:
            st.warning("Could not find fighters in the database.")


# ─── TAB 2: BACKTEST RESULTS ──────────────────────────────────────────────────

UNIT_SIZE = 10.0  # $10 per unit

def tab_backtest():
    st.header("Backtest Results — Recommended Plays Only")
    st.caption(f"1 unit = $10 · {int(KELLY_FRAC*100)}% fractional Kelly · Max {int(MAX_UNITS)}u/fight · DK/FD closing lines only · {int(MIN_EDGE*100)}–{int(MAX_EDGE*100)}% edge window")
    df = load_backtest()

    if df.empty:
        st.info("Run `python betting/backtest.py` to generate backtest results.")
        return

    # Filter to only recommended plays (positive edge, stake > 0)
    if "bet" in df.columns:
        bet_df = df[df["bet"] == True].copy()
    else:
        bet_df = df.copy()

    if bet_df.empty:
        st.info("No recommended plays in backtest data.")
        return

    # Convert dollar stake → units ($10 base)
    if "stake" in bet_df.columns:
        bet_df["units"]  = (bet_df["stake"] / UNIT_SIZE).round(2)
        bet_df["pnl_units"] = (bet_df["pnl"] / UNIT_SIZE).round(2) if "pnl" in bet_df.columns else 0

    # Running unit P&L
    if "pnl_units" in bet_df.columns:
        bet_df["cumulative_units"] = bet_df["pnl_units"].cumsum()

    # ── Summary metrics ──────────────────────────────────────────────────────
    total_bets   = len(bet_df)
    wins         = int(bet_df["won"].sum()) if "won" in bet_df.columns else 0
    losses       = total_bets - wins
    win_rate     = bet_df["won"].mean() if "won" in bet_df.columns else 0
    total_units_staked = bet_df["units"].sum() if "units" in bet_df.columns else 0
    total_units_pnl    = bet_df["pnl_units"].sum() if "pnl_units" in bet_df.columns else 0
    roi          = total_units_pnl / max(total_units_staked, 1)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Record", f"{wins}–{losses}")
    col2.metric("Win Rate", f"{win_rate*100:.1f}%")
    col3.metric("Total Units Staked", f"{total_units_staked:.1f}u")
    col4.metric("P&L", f"{total_units_pnl:+.1f}u  (${total_units_pnl*UNIT_SIZE:+.0f})")
    col5.metric("ROI", f"{roi*100:.1f}%")

    # ── Cumulative unit P&L chart ────────────────────────────────────────────
    if "cumulative_units" in bet_df.columns and "date" in bet_df.columns:
        fig = px.line(bet_df, x="date", y="cumulative_units",
                      title="Cumulative P&L (units)",
                      labels={"cumulative_units": "Units P&L", "date": "Date"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_traces(line_color="#00c853")
        st.plotly_chart(fig, use_container_width=True)

    # ── ROI by year ─────────────────────────────────────────────────────────
    if "date" in bet_df.columns and "pnl_units" in bet_df.columns:
        bet_df["year"] = pd.to_datetime(bet_df["date"]).dt.year
        yearly = bet_df.groupby("year").agg(
            bets=("won", "count"),
            win_rate=("won", "mean"),
            units_staked=("units", "sum"),
            units_pnl=("pnl_units", "sum"),
        ).reset_index()
        yearly["roi"] = yearly["units_pnl"] / yearly["units_staked"].clip(lower=0.01)
        fig2 = px.bar(yearly, x="year", y="roi", title="ROI by Year",
                      labels={"roi": "ROI", "year": "Year"},
                      color="roi", color_continuous_scale="RdYlGn",
                      text=yearly["roi"].map(lambda r: f"{r*100:.1f}%"))
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Edge bucket breakdown ────────────────────────────────────────────────
    if "edge" in bet_df.columns:
        _b = bet_df.copy()
        _b["edge_bucket"] = pd.cut(
            _b["edge"],
            bins=[0.05, 0.10, 0.15, 0.20, 0.30, 1.0],
            labels=["5–10%", "10–15%", "15–20%", "20–30%", "30%+"],
        )
        bucket = (
            _b.groupby("edge_bucket", observed=True)
            .agg(Bets=("bet", "count"), Wins=("won", "sum"), PnL_u=("pnl_units", "sum"))
            .reset_index()
        )
        bucket["Win Rate"] = (bucket["Wins"] / bucket["Bets"]).map(lambda v: f"{v*100:.1f}%")
        bucket["P&L (u)"]  = bucket["PnL_u"].map(lambda v: f"{v:+.1f}u")
        bucket = bucket.rename(columns={"edge_bucket": "Edge"}).drop(columns=["PnL_u"])
        st.subheader("Performance by Edge Bucket")
        st.dataframe(bucket[["Edge", "Bets", "Wins", "Win Rate", "P&L (u)"]],
                     use_container_width=True, hide_index=True)

    # ── Monthly P&L bar chart ────────────────────────────────────────────────
    if "date" in bet_df.columns and "pnl_units" in bet_df.columns:
        _m = bet_df.copy()
        _m["month"] = pd.to_datetime(_m["date"]).dt.to_period("M").astype(str)
        monthly = _m.groupby("month").agg(pnl=("pnl_units", "sum")).reset_index()
        monthly["color"] = monthly["pnl"].apply(lambda v: "green" if v >= 0 else "red")
        fig3 = px.bar(
            monthly, x="month", y="pnl",
            title="Monthly P&L (units)",
            labels={"pnl": "Units P&L", "month": "Month"},
            color="color",
            color_discrete_map={"green": "#00c853", "red": "#ff5252"},
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Derive odds for the side that was bet on
    if "odds_a" in bet_df.columns and "odds_b" in bet_df.columns and "bet_on" in bet_df.columns:
        bet_df["odds"] = bet_df.apply(
            lambda r: r["odds_a"] if r["bet_on"] == "a" else r["odds_b"], axis=1
        )

    # ── Bet log ─────────────────────────────────────────────────────────────
    st.subheader("Bet Log")

    # Capture side and outcome before dropping columns
    bet_on_a = (bet_df["bet_on"] == "a") if "bet_on" in bet_df.columns else pd.Series(False, index=bet_df.index)
    bet_won   = bet_df["won"].astype(bool)  if "won"    in bet_df.columns else pd.Series(False, index=bet_df.index)

    rename = {"date": "Date", "name_a": "Fighter A", "name_b": "Fighter B",
              "odds": "Odds", "edge": "Edge",
              "model_prob": "Model %", "units": "Units",
              "won": "Won", "pnl_units": "P&L (u)", "cumulative_units": "Cum. P&L (u)"}
    show = [c for c in rename if c in bet_df.columns]
    display = bet_df[show].rename(columns=rename).sort_values("Date", ascending=False)

    # Format columns
    for col in ["Odds"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"+{int(x)}" if pd.notna(x) and x > 0 else (f"{int(x)}" if pd.notna(x) else ""))
    for col in ["Model %"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
    for col in ["Edge"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"+{x*100:.1f}%" if pd.notna(x) else "")
    for col in ["Units"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.2f}u" if pd.notna(x) else "")
    for col in ["P&L (u)", "Cum. P&L (u)"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:+.2f}u" if pd.notna(x) else "")

    def _highlight_bet(row):
        styles = pd.Series("", index=row.index)
        target = "Fighter A" if bet_on_a.get(row.name, False) else "Fighter B"
        if target in styles.index:
            color = "rgba(0, 200, 83, 0.2)" if bet_won.get(row.name, False) else "rgba(220, 53, 69, 0.2)"
            styles[target] = f"background-color: {color}; font-weight: bold"
        return styles

    st.dataframe(display.style.apply(_highlight_bet, axis=1), use_container_width=True, hide_index=True)


# ─── TAB 3: FIGHTER LOOKUP ────────────────────────────────────────────────────

def tab_fighter_lookup():
    st.header("Fighter Lookup")
    if not db_available():
        st.warning("Fighter lookup requires `data/ufc.db` on the server.")
        return
    query = st.text_input("Search fighter name")

    if not query:
        return

    con = get_db()
    fighters = pd.read_sql_query(
        "SELECT * FROM fighters WHERE name LIKE ? LIMIT 20",
        con, params=(f"%{query}%",)
    )

    if fighters.empty:
        st.warning("No fighters found. The scraper may still be running.")
        con.close()
        return

    fighter_name = st.selectbox("Select fighter", fighters["name"].tolist())
    fighter_row = fighters[fighters["name"] == fighter_name].iloc[0]
    fighter_id = fighter_row["fighter_id"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Profile")
        st.write(f"**DOB:** {fighter_row.get('dob', 'N/A')}")
        st.write(f"**Height:** {fighter_row.get('height_inches', 0):.0f}\" ({fighter_row.get('height_inches', 0)/12:.1f}')")
        st.write(f"**Reach:** {fighter_row.get('reach_inches', 0):.0f}\"")
        st.write(f"**Stance:** {fighter_row.get('stance', 'N/A')}")

    with col2:
        st.subheader("Record")
        wins = fighter_row.get("total_wins", 0)
        losses = fighter_row.get("total_losses", 0)
        draws = fighter_row.get("total_draws", 0)
        st.metric("Record", f"{wins}-{losses}-{draws}")
        st.write(f"KO/TKO wins: {fighter_row.get('ko_wins', 0)}")
        st.write(f"Sub wins: {fighter_row.get('sub_wins', 0)}")
        st.write(f"Dec wins: {fighter_row.get('dec_wins', 0)}")

    # ELO history
    elo_df = pd.read_sql_query(
        "SELECT fight_date, elo_before, elo_after FROM elo_history WHERE fighter_id=? ORDER BY fight_date",
        con, params=(fighter_id,)
    )
    if not elo_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elo_df["fight_date"], y=elo_df["elo_after"],
                                  mode="lines+markers", name="ELO Rating"))
        fig.update_layout(title=f"{fighter_name} ELO Rating History",
                          xaxis_title="Date", yaxis_title="ELO")
        st.plotly_chart(fig, use_container_width=True)

    # Fight history
    fights = pd.read_sql_query(
        """SELECT f.fight_date, opp.name as opponent, f.method, f.round,
                  CASE WHEN f.winner_id = ? THEN 'Win' ELSE 'Loss' END as result
           FROM fights f
           JOIN fighters opp ON opp.fighter_id = CASE
               WHEN f.fighter_a_id = ? THEN f.fighter_b_id ELSE f.fighter_a_id END
           WHERE f.fighter_a_id = ? OR f.fighter_b_id = ?
           ORDER BY f.fight_date DESC LIMIT 20""",
        con, params=(fighter_id, fighter_id, fighter_id, fighter_id)
    )
    if not fights.empty:
        st.subheader("Recent Fights")
        st.dataframe(fights, use_container_width=True)

    con.close()


# ─── TAB 4: PAST CARDS ────────────────────────────────────────────────────────

# load_past_cards_data and _score_fight removed — Past Cards now reads the
# backtest CSV directly so there is one implementation, not two.

def tab_past_cards(_models_unused=None):
    st.header("Past Cards")
    st.caption("Model predictions & recommendations · 1u = $10 · ✅ = correct · ❌ = wrong · Same rules and data as Backtest tab")

    df = load_backtest()
    if df.empty:
        st.info("Run `python betting/backtest.py` to generate results.")
        return

    df["date"] = pd.to_datetime(df["date"])
    min_date = df["date"].min()
    max_date = df["date"].max()

    st.write("**Quick select:**")
    qc1, qc2, qc3 = st.columns(3)
    preset = None
    if qc1.button("2026 YTD"):  preset = ("2026-01-01", str(max_date.date()))
    if qc2.button("2025 full"): preset = ("2025-01-01", "2025-12-31")
    if qc3.button("All"):       preset = (str(min_date.date()), str(max_date.date()))

    if preset:
        default_start = pd.Timestamp(preset[0]).date()
        default_end   = pd.Timestamp(preset[1]).date()
    else:
        default_start = pd.Timestamp("2026-01-01").date()
        default_end   = max_date.date()

    col_l, col_r = st.columns(2)
    with col_l:
        start = st.date_input("From", value=default_start, min_value=min_date.date(), max_value=max_date.date())
    with col_r:
        end = st.date_input("To", value=default_end, min_value=min_date.date(), max_value=max_date.date())

    filtered = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    if filtered.empty:
        st.info("No fights in selected date range.")
        return

    bet_df = filtered[filtered["bet"] == True]

    # ── Summary stats ────────────────────────────────────────────────────────
    # Count events treating consecutive same-name dates as one event
    _event_dates = filtered.groupby("event_name")["date"].min().reset_index()
    n_events      = len(_event_dates)
    n_fights      = len(filtered)
    model_acc     = filtered["model_correct"].mean() if "model_correct" in filtered.columns else 0
    n_plays       = len(bet_df)
    wins          = int(bet_df["won"].sum()) if n_plays else 0
    losses        = n_plays - wins
    units_staked  = bet_df["units"].sum() if n_plays else 0
    total_pnl_u   = bet_df["pnl_units"].sum() if n_plays else 0
    total_pnl_d   = bet_df["pnl"].sum() if n_plays else 0
    roi           = total_pnl_u / units_staked if units_staked > 0 else 0

    st.divider()
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Events",     n_events)
    c2.metric("Fights",     n_fights)
    c3.metric("Model Acc",  f"{model_acc*100:.1f}%")
    c4.metric("Plays",      n_plays)
    c5.metric("Record",     f"{wins}-{losses}" if n_plays else "—")
    c6.metric("P&L (u)",    f"{total_pnl_u:+.1f}u" if n_plays else "—")
    c7.metric("P&L ($)",    f"${total_pnl_d:+.0f}" if n_plays else "—")
    c8.metric("ROI",        f"{roi*100:.1f}%" if n_plays else "—")
    st.divider()

    n_no_odds = int((~filtered["has_real_odds"]).sum())
    if n_no_odds == n_fights:
        st.info("No DK/FD closing lines in DB for this period. Run `python scrapers/fightodds.py` to backfill.")
    elif n_no_odds > 0:
        st.info(f"{n_no_odds}/{n_fights} fights have no DK/FD closing lines — those show model picks only.")

    def fmt_odds(v):
        try:
            return f"{int(v):+d}"
        except (TypeError, ValueError):
            return "—"

    def fmt_pick(row):
        pick_name = row["name_a"] if row["model_pick"] == "a" else row["name_b"]
        mark = "✓" if row.get("model_correct") else "✗"
        return f"{mark} {pick_name}"

    def fmt_play(row):
        if row["bet"]:
            return row["name_a"] if row["bet_on"] == "a" else row["name_b"]
        return "—" if row["has_real_odds"] else "no odds"

    def fmt_winner(row):
        return row["name_a"] if row.get("winner") == "a" else row["name_b"]

    def fmt_check(row):
        if not row["bet"]:
            return "—"
        return "✅" if row["won"] else "❌"

    def fmt_pnl(row):
        if not row["bet"]:
            return "—"
        return f"+${row['pnl']:.0f}" if row["won"] else f"-${abs(row['pnl']):.0f}"

    display_rows = []
    for _, row in filtered.iterrows():
        display_rows.append({
            "Fighter A":  row["name_a"],
            "Odds A":     fmt_odds(row.get("odds_a")),
            "Model A%":   f"{row['prob_a']*100:.0f}%",
            "Fighter B":  row["name_b"],
            "Odds B":     fmt_odds(row.get("odds_b")),
            "Model B%":   f"{row['prob_b']*100:.0f}%",
            "Model Pick": fmt_pick(row),
            "Play":       fmt_play(row),
            "Edge":       (f"{row['edge']*100:+.1f}%" if row["bet"] and pd.notna(row.get("edge"))
                          else f"{row['model_edge']*100:+.1f}%" if pd.notna(row.get("model_edge")) else "—"),
            "Units":      f"{row['units']:.1f}u" if row["bet"] else "—",
            "Stake":      f"${row['stake']:.0f}" if row["bet"] else "PASS",
            "P&L":        fmt_pnl(row),
            "Winner":     fmt_winner(row),
            "Result":     row.get("result") or "—",
            "✓":          fmt_check(row),
            "_date":      str(row["date"])[:10],
            "_event":     row.get("event_name") or "UFC Event",
            "_had_play":  bool(row["bet"]),
            "_pnl_u":     float(row["pnl_units"]) if row["bet"] else 0.0,
            "_won":       bool(row["won"]) if row["bet"] and pd.notna(row.get("won")) else None,
        })

    event_groups: dict = {}
    for r in display_rows:
        key = (r["_date"], r["_event"])
        event_groups.setdefault(key, []).append(r)

    # Merge events split across midnight: same event_name on consecutive dates
    # (e.g. Pimblett prelims on Jan 24, main card runs past midnight into Jan 25)
    merged_groups: dict = {}
    for (date, name), rows in sorted(event_groups.items()):
        prev = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
        if (prev, name) in merged_groups:
            merged_groups[(prev, name)].extend(rows)
        else:
            merged_groups[(date, name)] = rows
    event_groups = merged_groups

    def style_past(row):
        styles = [""] * len(row)
        idx = list(row.index)
        check = row.get("✓", "—")
        if check == "✅":
            for col in ["✓", "P&L", "Play", "Edge", "Units"]:
                if col in idx: styles[idx.index(col)] = "color: #00c853; font-weight: bold"
        elif check == "❌":
            for col in ["✓", "P&L", "Play", "Edge", "Units", "Stake"]:
                if col in idx: styles[idx.index(col)] = "color: #ff5252; font-weight: bold"
        return styles

    display_cols = ["Fighter A", "Odds A", "Model A%", "Fighter B", "Odds B",
                    "Model B%", "Model Pick", "Play", "Edge", "Units", "Stake", "P&L", "Winner", "Result", "✓"]

    for (event_date, event_name) in sorted(event_groups.keys(), reverse=True):
        rows     = event_groups[(event_date, event_name)]
        e_plays  = [r for r in rows if r["_had_play"]]
        e_wins   = sum(1 for r in e_plays if r["_won"])
        e_losses = len(e_plays) - e_wins
        e_pnl    = sum(r["_pnl_u"] for r in e_plays)

        if e_plays:
            label = (f"📅 {event_date} — {event_name}  "
                     f"|  {len(rows)} fights  |  {len(e_plays)} plays ({e_wins}-{e_losses})  "
                     f"|  P&L: {e_pnl:+.1f}u  (${e_pnl*10:+.0f})")
        else:
            label = f"📅 {event_date} — {event_name}  |  {len(rows)} fights  |  No plays"

        with st.expander(label, expanded=False):
            display_df = pd.DataFrame(rows)[display_cols]
            st.dataframe(
                display_df.style.apply(style_past, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            if e_plays:
                c1, c2, c3 = st.columns(3)
                c1.metric("Plays",  f"{len(e_plays)}/{len(rows)}")
                c2.metric("Record", f"{e_wins}-{e_losses}")
                c3.metric("P&L",    f"{e_pnl:+.1f}u  (${e_pnl*10:+.0f})")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.title("🥊 UFC Betting Model")
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", help="Clear cache and reload all data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    missing = deployment_missing()
    if missing:
        st.error(
            "Deploy data missing on this server: "
            + ", ".join(missing)
            + ". Push these files to GitHub (private repo is fine) and redeploy."
        )

    models, model_errors = load_models(_models_signature())

    if not models:
        pkls = list(MODELS_DIR.glob("*.pkl"))
        if model_errors:
            st.warning("Model files found but failed to load:\n\n" + "\n".join(f"- {e}" for e in model_errors))
        elif pkls:
            st.warning(
                "Models did not load (cached empty state). Click **Refresh** above or reboot the app "
                "from Streamlit Cloud settings."
            )
        else:
            st.warning("Models not trained yet. Run `python models/moneyline.py` etc. after scraping completes.")

    tab1, tab2, tab3, tab4 = st.tabs(["Next Event", "Backtest Results", "Past Cards", "Fighter Lookup"])

    with tab1:
        tab_next_event(models)

    with tab2:
        tab_backtest()

    with tab3:
        tab_past_cards()

    with tab4:
        tab_fighter_lookup()


if __name__ == "__main__":
    main()
