"""
Captures DK/FD closing lines from The Odds API and saves to odds_history.
Run this 1-2 hours before each UFC event to store the near-closing lines.
These become the permanent record for backtesting future events.

Usage:
    python scrapers/capture_closing_lines.py

Schedule: Run weekly on UFC event weekends (Friday/Saturday).
"""
import hashlib
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH
from betting.odds_fetcher import get_upcoming_fights
from betting.devig import no_vig_probs

ET = timezone(timedelta(hours=-4))  # EDT


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def find_fight_id(con, name_a: str, name_b: str, date_str: str):
    na, nb = normalize(name_a), normalize(name_b)
    rows = con.execute(
        """SELECT f.fight_id, fa.name, fb.name
           FROM fights f
           JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
           JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
           WHERE f.fight_date BETWEEN date(?, '-3 days') AND date(?, '+3 days')""",
        (date_str, date_str),
    ).fetchall()
    for fight_id, fn_a, fn_b in rows:
        nfa, nfb = normalize(fn_a), normalize(fn_b)
        if (na[:6] in nfa or nfa[:6] in na) and (nb[:6] in nfb or nfb[:6] in nb):
            return fight_id, False
        if (nb[:6] in nfa or nfa[:6] in nb) and (na[:6] in nfb or nfb[:6] in na):
            return fight_id, True
    return None, False


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    print("Fetching DK/FD odds from The Odds API...")
    fights = get_upcoming_fights()
    print(f"  {len(fights)} fights returned")

    saved = 0
    for fight in fights:
        fa = fight["fighter_a"]
        fb = fight["fighter_b"]
        commence = fight.get("commence_time", "")

        # Convert to ET for date
        try:
            utc_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            fight_date = utc_dt.astimezone(ET).strftime("%Y-%m-%d")
        except Exception:
            fight_date = commence[:10]

        fight_id, swapped = find_fight_id(con, fa, fb, fight_date)

        # Store DK odds
        dk_a = fight.get("draftkings_a")
        dk_b = fight.get("draftkings_b")
        if dk_a and dk_b:
            if swapped: dk_a, dk_b = dk_b, dk_a
            oid = uid(fa, fb, fight_date, "capture_dk")
            con.execute(
                """INSERT OR REPLACE INTO odds_history
                   (odds_id, fight_id, fight_date, fighter_a_name, fighter_b_name,
                    fighter_a_odds, fighter_b_odds, book, scraped_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (oid, fight_id, fight_date,
                 fb if swapped else fa, fa if swapped else fb,
                 dk_a, dk_b, "capture_DraftKings", datetime.now().isoformat()),
            )
            saved += 1

        # Store FD odds
        fd_a = fight.get("fanduel_a")
        fd_b = fight.get("fanduel_b")
        if fd_a and fd_b:
            if swapped: fd_a, fd_b = fd_b, fd_a
            oid = uid(fa, fb, fight_date, "capture_fd")
            con.execute(
                """INSERT OR REPLACE INTO odds_history
                   (odds_id, fight_id, fight_date, fighter_a_name, fighter_b_name,
                    fighter_a_odds, fighter_b_odds, book, scraped_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (oid, fight_id, fight_date,
                 fb if swapped else fa, fa if swapped else fb,
                 fd_a, fd_b, "capture_FanDuel", datetime.now().isoformat()),
            )
            saved += 1

    con.commit()

    total = con.execute("SELECT COUNT(*) FROM odds_history WHERE book LIKE 'capture_%'").fetchone()[0]
    matched = con.execute("SELECT COUNT(*) FROM odds_history WHERE book LIKE 'capture_%' AND fight_id IS NOT NULL").fetchone()[0]
    print(f"\nSaved {saved} odds rows this run | Total captured: {total} | Matched: {matched}")

    # Show what was captured
    rows = con.execute(
        """SELECT fighter_a_name, fighter_a_odds, fighter_b_name, fighter_b_odds, book, fight_date
           FROM odds_history WHERE book LIKE 'capture_%'
           ORDER BY scraped_at DESC LIMIT 10"""
    ).fetchall()
    for r in rows:
        print(f"  {r[0]} ({r[1]:+d}) vs {r[2]} ({r[3]:+d}) @ {r[4]} on {r[5]}")

    con.close()


if __name__ == "__main__":
    main()
