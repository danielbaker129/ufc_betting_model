"""
Scrapes historical UFC odds from fightodds.io GraphQL API.
No authentication required — public API with rate limiting.
Provides opening + closing odds per sportsbook per fight.
Event pks range from ~1 (oldest) to ~7000+ (present).
"""
import hashlib
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests
import cloudscraper

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH

API = "https://api.fightodds.io/gql"
HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://fightodds.io",
    "Referer": "https://fightodds.io/",
}
DELAY = 1.5   # seconds between requests

# Only store odds from books the user actually bets at
TARGET_BOOKS = {"DraftKings", "FanDuel"}

# Cloudscraper handles Cloudflare JS challenges
_scraper = cloudscraper.create_scraper()
_scraper.headers.update(HEADERS)


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def gql(query: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            time.sleep(DELAY)
            r = _scraper.post(API, json={"query": query}, timeout=15)
            if r.status_code == 429:
                print(f"    Rate limited — sleeping 30s")
                time.sleep(30)
                continue
            if r.status_code == 403:
                print(f"    Cloudflare block — sleeping 60s")
                time.sleep(60)
                continue
            if r.status_code != 200 or not r.text:
                return None
            data = r.json()
            if "errors" in data:
                return None
            return data.get("data")
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(5)
    return None


EVENT_QUERY = """
{{
  eventOfferTable(pk: {pk}) {{
    name
    date
    slug
    fightOffers {{
      edges {{
        node {{
          fighter1 {{ firstName lastName }}
          fighter2 {{ firstName lastName }}
          bestOdds1
          bestOdds2
          fight {{
            fighterWinner {{ firstName lastName }}
            methodOfVictory1
            round
          }}
          straightOffers {{
            edges {{
              node {{
                sportsbook {{ shortName fullName }}
                outcome1 {{ odds oddsOpen oddsPrev oddsBest oddsWorst }}
                outcome2 {{ odds oddsOpen oddsPrev oddsBest oddsWorst }}
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""


def normalize(name: str) -> str:
    # Decompose unicode (é→e, ã→a, etc.) then strip non-ascii-alpha
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", ascii_only.lower())


def _last(name: str) -> str:
    """Normalize the last word of a name (surname)."""
    parts = name.strip().split()
    return normalize(parts[-1]) if parts else ""


# Known name aliases: fightodds name → DB name (normalized)
_ALIASES = {
    "teciatorres": "teciapennington",   # married name change
    "patrickmix": "patchymix",          # "Patrick" is his real name, UFC uses "Patchy"
}


def _alias(n: str) -> str:
    return _ALIASES.get(n, n)


def _names_match(na: str, nb: str) -> bool:
    """True if two normalized full names are close enough to be the same fighter."""
    na, nb = _alias(na), _alias(nb)
    if na == nb:
        return True
    # 6-char prefix overlap
    if len(na) >= 4 and (na[:6] in nb or nb[:6] in na):
        return True
    # 5-char prefix: catches transliteration variants (Sergey/Sergei, Viacheslav/Viecheslav)
    # and extra middle names (Ateba Abega Gautier / Ateba Gautier)
    if len(na) >= 8 and len(nb) >= 8 and (na[:5] in nb or nb[:5] in na):
        return True
    # Sorted-chars: handles transpositions (Cyril/Ciryl) and reversed word order (Xiao Long/Long Xiao)
    if len(na) >= 5 and len(nb) >= 5 and sorted(na) == sorted(nb):
        return True
    return False


def find_fight_id(con, name_a: str, name_b: str, date_str: str):
    na, nb = normalize(name_a), normalize(name_b)
    la, lb = _last(name_a), _last(name_b)
    if not na or not nb:
        return None, False
    rows = con.execute(
        """SELECT f.fight_id, fa.name, fb.name
           FROM fights f
           JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
           JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
           WHERE f.fight_date BETWEEN date(?, '-5 days') AND date(?, '+5 days')""",
        (date_str, date_str),
    ).fetchall()

    # Pass 1: full-name match
    for fight_id, fn_a, fn_b in rows:
        nfa, nfb = normalize(fn_a), normalize(fn_b)
        if _names_match(na, nfa) and _names_match(nb, nfb):
            return fight_id, False
        if _names_match(nb, nfa) and _names_match(na, nfb):
            return fight_id, True

    # Pass 2: last-name fallback — handles Dan/Daniel, Joe/Joseph, Jim/Jimmy, etc.
    for fight_id, fn_a, fn_b in rows:
        lfa, lfb = _last(fn_a), _last(fn_b)
        if len(la) >= 4 and len(lb) >= 4 and la == lfa and lb == lfb:
            return fight_id, False
        if len(lb) >= 4 and len(la) >= 4 and lb == lfa and la == lfb:
            return fight_id, True

    return None, False


def scrape_event(con, pk: int) -> tuple[int, bool]:
    """Scrape one event. Returns (rows_saved, is_ufc)."""
    data = gql(EVENT_QUERY.format(pk=pk))
    if not data or not data.get("eventOfferTable"):
        return 0, False

    ev = data["eventOfferTable"]
    event_name = ev.get("name", "")
    event_date = ev.get("date", "")

    # Only process UFC events
    is_ufc = "ufc" in event_name.lower()
    if not is_ufc:
        return 0, False

    saved = 0
    for edge in (ev.get("fightOffers") or {}).get("edges", []):
        node = edge["node"]
        f1_data = node.get("fighter1") or {}
        f2_data = node.get("fighter2") or {}
        fa_name = f"{f1_data.get('firstName','')} {f1_data.get('lastName','')}".strip()
        fb_name = f"{f2_data.get('firstName','')} {f2_data.get('lastName','')}".strip()

        if not fa_name or not fb_name:
            continue

        fight_id, swapped = find_fight_id(con, fa_name, fb_name, event_date)

        for offer_edge in (node.get("straightOffers") or {}).get("edges", []):
            offer = offer_edge["node"]
            book = offer["sportsbook"]["shortName"]

            # Only store target books
            if book not in TARGET_BOOKS:
                continue

            o1 = offer["outcome1"]
            o2 = offer["outcome2"]

            # Align to DB fighter order
            if swapped:
                o1, o2 = o2, o1
                fa_stored, fb_stored = fb_name, fa_name
            else:
                fa_stored, fb_stored = fa_name, fb_name

            # Store closing odds
            odds_id = uid(fa_stored, fb_stored, event_date, book, "fightodds")
            con.execute(
                """INSERT OR IGNORE INTO odds_history
                   (odds_id, fight_id, fight_date, fighter_a_name, fighter_b_name,
                    fighter_a_odds, fighter_b_odds, book, scraped_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (odds_id, fight_id, event_date, fa_stored, fb_stored,
                 o1.get("odds"), o2.get("odds"), f"fightodds_{book}", datetime.now().isoformat()),
            )

            # Store opening odds separately
            if o1.get("oddsOpen") and o2.get("oddsOpen"):
                open_id = uid(fa_stored, fb_stored, event_date, book, "fightodds_open")
                con.execute(
                    """INSERT OR IGNORE INTO odds_history
                       (odds_id, fight_id, fight_date, fighter_a_name, fighter_b_name,
                        fighter_a_odds, fighter_b_odds, book, scraped_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (open_id, fight_id, event_date, fa_stored, fb_stored,
                     o1["oddsOpen"], o2["oddsOpen"],
                     f"fightodds_{book}_open", datetime.now().isoformat()),
                )
            saved += 1

    con.commit()
    return saved, True


def find_ufc_pk_range(sample_pks: list[int]) -> tuple[int, int]:
    """Find min/max pk range that includes UFC events."""
    min_ufc, max_ufc = 99999, 0
    for pk in sample_pks:
        data = gql(f'{{ eventOfferTable(pk: {pk}) {{ name }} }}')
        if data and data.get("eventOfferTable"):
            name = data["eventOfferTable"].get("name", "")
            if "ufc" in name.lower():
                min_ufc = min(min_ufc, pk)
                max_ufc = max(max_ufc, pk)
    return min_ufc, max_ufc


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    # Find the pk range — UFC events are scattered among all MMA promotions
    # Scan all pks; skip non-UFC events fast since they return 0 rows
    print("Scraping fightodds.io UFC historical odds...")
    print("(Skipping non-UFC events automatically)")

    total_saved = 0
    ufc_events = 0
    empty_streak = 0

    # UFC events on fightodds span pk=1000-10000+ mixed with other promotions.
    # Scan all pks but skip non-UFC fast (no DB write = instant).
    # Start from 10000 (well above current newest known events ~9000) down to 1.
    for pk in range(10000, 0, -1):
        saved, is_ufc = scrape_event(con, pk)

        if is_ufc:
            ufc_events += 1
            empty_streak = 0
            if saved > 0:
                total_saved += saved
                matched = con.execute(
                    "SELECT COUNT(*) FROM odds_history WHERE fight_id IS NOT NULL AND book LIKE 'fightodds%'"
                ).fetchone()[0]
                print(f"  pk={pk}: {saved} odds rows  |  total={total_saved}  matched={matched}")
        else:
            empty_streak += 1
            # After 500 consecutive non-UFC events in the low-pk range, stop
            if empty_streak > 500 and pk < 1000:
                print(f"  No UFC events found below pk={pk} — stopping")
                break

        if pk % 500 == 0:
            print(f"  ... pk={pk} scanned (UFC events found: {ufc_events})")

    total = con.execute(
        "SELECT COUNT(*) FROM odds_history WHERE book LIKE 'fightodds%'"
    ).fetchone()[0]
    matched = con.execute(
        "SELECT COUNT(*) FROM odds_history WHERE fight_id IS NOT NULL AND book LIKE 'fightodds%'"
    ).fetchone()[0]
    print(f"\nDone. {total} odds rows, {matched} matched to fights. {ufc_events} UFC events scraped.")
    con.close()


if __name__ == "__main__":
    main()
