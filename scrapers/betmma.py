"""
Scrapes historical UFC odds from betmma.tips (login required).
Structure: free_ufc_betting_tips.php?Event=N
Each fight = 4-row table; last row = [odds_a, odds_b].
Fighter names in row[0][1]: "FighterAvssFighterB".
"""
import hashlib
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, USER_AGENT

BASE  = "https://www.betmma.tips"
DELAY = 1.2


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def login(email: str, password: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT,
                       "Referer": BASE,
                       "Accept-Language": "en-US,en;q=0.9"})
    s.post(f"{BASE}/", data={"txt_UName": email, "txt_pword": password, "btnLogin": "Login"},
           timeout=15, allow_redirects=True)
    # Verify
    time.sleep(1)
    r = s.get(f"{BASE}/next_ufc_event.php", timeout=15)
    if "logout" in r.text.lower() or email.lower() in r.text.lower():
        print(f"  ✓ Logged in as {email}")
    else:
        print(f"  ⚠ Login uncertain — proceeding")
    return s


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def find_fight_id(con, name_a: str, name_b: str, date_str: str):
    """Returns (fight_id, swapped) where swapped=True means betmma A=DB's fighter_b."""
    na, nb = normalize(name_a), normalize(name_b)
    if not na or not nb:
        return None, False
    rows = con.execute(
        """SELECT f.fight_id, fa.name, fb.name
           FROM fights f
           JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
           JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
           WHERE f.fight_date BETWEEN date(?, '-10 days') AND date(?, '+10 days')""",
        (date_str, date_str),
    ).fetchall()
    for fight_id, fn_a, fn_b in rows:
        nfa, nfb = normalize(fn_a), normalize(fn_b)
        # Same order: betmma A → DB A, betmma B → DB B
        if (na[:6] in nfa or nfa[:6] in na) and (nb[:6] in nfb or nfb[:6] in nb):
            return fight_id, False
        # Reversed: betmma A → DB B, betmma B → DB A
        if (nb[:6] in nfa or nfa[:6] in nb) and (na[:6] in nfb or nfb[:6] in na):
            return fight_id, True
    return None, False


def parse_odds(s: str):
    s = re.sub(r"[^\d\-+]", "", s.strip())
    try:
        v = int(s)
        return v if abs(v) < 5001 else None
    except Exception:
        return None


def scrape_event(session: requests.Session, event_id: int, con) -> tuple[int, str]:
    """Scrape one event page. Returns (fights_saved, event_date_str)."""
    url = f"{BASE}/free_ufc_betting_tips.php?Event={event_id}"
    try:
        time.sleep(DELAY)
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return 0, ""
    except Exception:
        return 0, ""

    soup = BeautifulSoup(r.text, "lxml")

    # Extract event name + date
    event_name = ""
    event_date = ""
    h = soup.find("h1") or soup.find("h2")
    if h:
        event_name = h.get_text(strip=True)

    # Date is often in a span or p near the title — format: "25th Apr 2026" or "May 3, 2025"
    full_text = soup.get_text(" ")
    date_patterns = [
        r"(\d{1,2})(?:st|nd|rd|th)\s+(\w+)\s+(\d{4})",   # 25th Apr 2026
        r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",                  # April 25, 2026
        r"(\d{4})-(\d{2})-(\d{2})",                         # 2026-04-25
    ]
    for pattern in date_patterns:
        m = re.search(pattern, full_text)
        if m:
            raw = m.group(0)
            raw_clean = re.sub(r"(?<=\d)(st|nd|rd|th)", "", raw)
            for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%B %d, %Y",
                        "%b %d %Y", "%b %d, %Y", "%Y-%m-%d"):
                try:
                    event_date = datetime.strptime(raw_clean.strip(), fmt).strftime("%Y-%m-%d")
                    break
                except Exception:
                    pass
            if event_date:
                break

    # Find fight tables: 4-row tables where last row has two odds values
    tables = soup.find_all("table")
    saved = 0

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) != 4:
            continue

        # Last row should be [odds_a, odds_b]
        last_cells = [td.get_text(strip=True) for td in rows[-1].find_all(["td", "th"])]
        if len(last_cells) < 2:
            continue

        odds_a = parse_odds(last_cells[0])
        odds_b = parse_odds(last_cells[1])
        if odds_a is None or odds_b is None:
            continue

        # Fighter names from row[0], cell[1]: "NameAvsNameB[Picking...]"
        # Use individual <td> elements to avoid text concatenation across cells
        first_row_tds = rows[0].find_all(["td", "th"])
        name_cell = ""
        for td in first_row_tds:
            txt = td.get_text(strip=True)
            if "vs" in txt.lower() and len(txt) > 6:
                name_cell = txt
                break

        if not name_cell:
            continue

        # Split on "vs" (with or without surrounding spaces)
        # Then truncate fighter B name at first known noise keyword
        NOISE = re.compile(r"Picking|Parlay|parlay|straight up|Props|props", re.I)
        parts = re.split(r"\s*vs\.?\s*", name_cell, flags=re.I, maxsplit=1)
        if len(parts) < 2:
            continue
        fa_raw = parts[0].strip()
        fb_part = parts[1]
        m_noise = NOISE.search(fb_part)
        fb_raw = fb_part[:m_noise.start()].strip() if m_noise else fb_part.strip()

        # Final safety: truncate names at 40 chars
        fa_raw = fa_raw[:40].strip()
        fb_raw = fb_raw[:40].strip()

        if len(fa_raw) < 3 or len(fb_raw) < 3:
            continue

        fight_id, swapped = find_fight_id(con, fa_raw, fb_raw, event_date) if event_date else (None, False)

        # If fighters are reversed vs DB order, swap odds so they align correctly
        stored_odds_a = odds_b if swapped else odds_a
        stored_odds_b = odds_a if swapped else odds_b
        stored_name_a = fb_raw if swapped else fa_raw
        stored_name_b = fa_raw if swapped else fb_raw

        odds_id = uid(stored_name_a, stored_name_b, event_date, str(event_id))
        con.execute(
            """INSERT OR IGNORE INTO odds_history
               (odds_id, fight_id, fight_date, fighter_a_name, fighter_b_name,
                fighter_a_odds, fighter_b_odds, book, scraped_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (odds_id, fight_id, event_date, stored_name_a, stored_name_b,
             stored_odds_a, stored_odds_b, "betmma", datetime.now().isoformat()),
        )
        saved += 1

    con.commit()
    return saved, event_date


def find_event_range(session: requests.Session) -> tuple[int, int]:
    """Find the highest valid event ID on betmma."""
    # Start high and binary search downward
    hi = 2100
    for eid in range(hi, hi - 50, -1):
        time.sleep(0.5)
        r = session.get(f"{BASE}/free_ufc_betting_tips.php?Event={eid}", timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        tables_with_odds = [t for t in soup.find_all("table")
                            if len(t.find_all("tr")) == 4 and
                            any(re.match(r'^[+-]\d{2,4}$', td.get_text(strip=True))
                                for td in t.find_all("tr")[-1].find_all("td"))]
        if tables_with_odds:
            return 1, eid
    return 1, 2000


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    email    = os.environ.get("BETMMA_EMAIL", "").strip()
    password = os.environ.get("BETMMA_PASSWORD", "").strip()
    if not email or not password:
        print("ERROR: Set BETMMA_EMAIL and BETMMA_PASSWORD in .env")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    print(f"Logging in to betmma.tips...")
    session = login(email, password)

    print("Finding event range...")
    min_eid, max_eid = find_event_range(session)
    print(f"Scraping events {min_eid} → {max_eid}...")

    total_fights = 0
    empty_streak = 0
    consecutive_empty = 0

    for eid in range(max_eid, min_eid - 1, -1):
        n, edate = scrape_event(session, eid, con)
        if n > 0:
            total_fights += n
            consecutive_empty = 0
            if eid % 50 == 0 or n > 5:
                matched = con.execute("SELECT COUNT(*) FROM odds_history WHERE fight_id IS NOT NULL").fetchone()[0]
                print(f"  Event {eid} ({edate}): {n} fights  |  total={total_fights}  matched={matched}")
        else:
            consecutive_empty += 1
            # Only stop after 100 consecutive empty events (some IDs are unused)
            if consecutive_empty > 100:
                print(f"  100 empty events in a row — stopping at event {eid}")
                break

    total = con.execute("SELECT COUNT(*) FROM odds_history").fetchone()[0]
    matched = con.execute("SELECT COUNT(*) FROM odds_history WHERE fight_id IS NOT NULL").fetchone()[0]
    print(f"\nDone. {total} odds rows, {matched} matched to fights in DB.")
    con.close()


if __name__ == "__main__":
    main()
