"""
Scrapes DraftKings and FanDuel UFC odds from bestfightodds.com.
Static HTML — no auth, no JS required, robots.txt fully open.

Structure:
  <td data-li="[book_id, fighter_pos, matchup_id]">
    <span id="oID{pos}{matchup}{book}">-120</span>
  </td>

  book_id 21 = FanDuel
  book_id 22 = DraftKings
  fighter_pos 1 = fighter A (listed first), 2 = fighter B

Archive URL: https://www.bestfightodds.com/archive
Event URLs:  https://www.bestfightodds.com/events/{slug}-{id}
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

BASE    = "https://www.bestfightodds.com"
DELAY   = 1.5
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE,
}

BOOK_DK = "22"   # DraftKings
BOOK_FD = "21"   # FanDuel
TARGET_BOOKS = {BOOK_DK: "DraftKings", BOOK_FD: "FanDuel"}


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def get(url: str) -> BeautifulSoup:
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def find_fight_id(con, name_a: str, name_b: str, date_str: str):
    na, nb = normalize(name_a), normalize(name_b)
    rows = con.execute(
        """SELECT f.fight_id, fa.name, fb.name
           FROM fights f
           JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
           JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
           WHERE f.fight_date BETWEEN date(?, '-5 days') AND date(?, '+5 days')""",
        (date_str, date_str),
    ).fetchall()
    for fight_id, fn_a, fn_b in rows:
        nfa, nfb = normalize(fn_a), normalize(fn_b)
        if (na[:6] in nfa or nfa[:6] in na) and (nb[:6] in nfb or nfb[:6] in nb):
            return fight_id, False
        if (nb[:6] in nfa or nfa[:6] in nb) and (na[:6] in nfb or nfb[:6] in na):
            return fight_id, True
    return None, False


def parse_event_page(soup: BeautifulSoup, event_date: str, con: sqlite3.Connection) -> int:
    """Parse a BFO event page and extract DK/FD odds per fight.

    BFO structure per matchup:
    - pos=1 row: th has only an admin link (no fighter name visible)
      → fighter 1 name = nearest /fighters/ link BEFORE this row in document order
    - pos=2 row: th has <a href="/fighters/..."><span class="t-b-fcc">Name</span></a>
      → fighter 2 name directly available
    """
    saved = 0
    matchups: dict[str, dict] = {}   # matchup_id → {pos: {name, DraftKings, FanDuel}}

    # Build ordered list of all elements for backward-lookup of fighter names
    all_elements = list(soup.descendants)

    # Index td[data-li] elements with their document position
    td_positions: list[tuple[int, any]] = []
    for i, el in enumerate(all_elements):
        if hasattr(el, "get") and el.get("data-li"):
            td_positions.append((i, el))

    def find_nearest_fighter_link_before(target_idx: int) -> str:
        """Walk backward in document order to find the nearest /fighters/ link."""
        for el in reversed(all_elements[:target_idx]):
            if (hasattr(el, "name") and el.name == "a"
                    and "/fighters/" in str(el.get("href", ""))):
                span = el.find("span", class_="t-b-fcc")
                return span.get_text(strip=True) if span else el.get_text(strip=True)
        return ""

    for doc_idx, td in td_positions:
        try:
            raw = td.get("data-li", "[]")
            parts = re.findall(r"\d+", raw)
            if len(parts) < 3:
                continue
            book_id, pos, matchup_id = parts[0], parts[1], parts[2]

            # Moneyline rows have exactly 3 numbers in data-li: [book_id, pos, matchup_id]
            # Prop rows have 5+: [book_id, pos, matchup_id, prop_id, ...]
            if len(parts) != 3:
                continue
            if pos not in ("1", "2"):
                continue
            if book_id not in TARGET_BOOKS:
                continue

            span = td.select_one("span")
            if not span:
                continue
            odds_text = span.get_text(strip=True).replace(",", "")
            if not re.match(r"^[+-]?\d+$", odds_text):
                continue
            odds = int(odds_text)
            if abs(odds) > 5000:
                continue

            # Get fighter name
            if pos == "2":
                # Name is in th of this row
                row = td.find_parent("tr")
                th = row.find("th") if row else None
                name_span = th.find("span", class_="t-b-fcc") if th else None
                fighter_name = name_span.get_text(strip=True) if name_span else ""
            else:
                # pos=1: name is in nearest fighter link BEFORE this element
                fighter_name = find_nearest_fighter_link_before(doc_idx)

            if not fighter_name or len(fighter_name) < 3:
                continue

            book_name = TARGET_BOOKS[book_id]
            m = matchups.setdefault(matchup_id, {})
            p = m.setdefault(pos, {"name": fighter_name})
            p[book_name] = odds
        except Exception:
            pass

    # Write odds to DB
    for matchup_id, positions in matchups.items():
        p1 = positions.get("1", {})
        p2 = positions.get("2", {})
        name_a = p1.get("name", "")
        name_b = p2.get("name", "")
        if not name_a or not name_b:
            continue

        fight_id, swapped = find_fight_id(con, name_a, name_b, event_date)
        if swapped:
            p1, p2 = p2, p1
            name_a, name_b = name_b, name_a

        for book_name in ("DraftKings", "FanDuel"):
            odds_a = p1.get(book_name)
            odds_b = p2.get(book_name)
            if odds_a is None or odds_b is None:
                continue

            odds_id = uid(name_a, name_b, event_date, book_name, "bfo")
            con.execute(
                """INSERT OR IGNORE INTO odds_history
                   (odds_id, fight_id, fight_date, fighter_a_name, fighter_b_name,
                    fighter_a_odds, fighter_b_odds, book, scraped_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (odds_id, fight_id, event_date, name_a, name_b,
                 odds_a, odds_b, f"bfo_{book_name}", datetime.now().isoformat()),
            )
            saved += 1

    con.commit()
    return saved


def parse_bfo_date(raw: str) -> str:
    """Parse BFO date formats like 'May 16th 2026', 'Jan 3rd 2025'."""
    raw = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw.strip())
    for fmt in ("%B %d %Y", "%b %d %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""


def get_event_list() -> list[tuple[str, str, str]]:
    """Return list of (event_url, event_name, event_date) from BFO archive."""
    print("Fetching BFO event archive...")
    soup = get(f"{BASE}/archive")
    events = []
    seen = set()

    for row in soup.select("table tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        link = row.find("a", href=True)
        if not link or "/events/" not in link["href"]:
            continue

        url = link["href"] if link["href"].startswith("http") else BASE + link["href"]
        name = link.get_text(strip=True)

        # Date is in first cell: "May 16th 2026"
        date_str = parse_bfo_date(cells[0].get_text(strip=True)) if cells else ""

        if url not in seen and name:
            seen.add(url)
            events.append((url, name, date_str))

    print(f"  Found {len(events)} events")
    return events


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    # Check existing BFO rows
    existing = con.execute(
        "SELECT COUNT(*) FROM odds_history WHERE book LIKE 'bfo_%'"
    ).fetchone()[0]
    print(f"Existing BFO rows: {existing}")

    events = get_event_list()
    total = 0

    for url, name, event_date in events:
        # Skip non-UFC events
        if "ufc" not in name.lower() and "ufc" not in url.lower():
            continue

        try:
            soup = get(url)

            # Try to extract date from page if not found in listing
            if not event_date:
                for el in soup.find_all(["h1", "h2", "h3", "span", "p"]):
                    txt = el.get_text(strip=True)
                    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
                        try:
                            event_date = datetime.strptime(txt, fmt).strftime("%Y-%m-%d")
                            break
                        except Exception:
                            pass
                    if event_date:
                        break

            n = parse_event_page(soup, event_date, con)
            if n > 0:
                total += n
                matched = con.execute(
                    "SELECT COUNT(*) FROM odds_history WHERE book LIKE 'bfo_%' AND fight_id IS NOT NULL"
                ).fetchone()[0]
                print(f"  {name} ({event_date}): {n} rows | matched: {matched}")

        except Exception as e:
            print(f"  Error {url}: {e}")

    total_rows = con.execute("SELECT COUNT(*) FROM odds_history WHERE book LIKE 'bfo_%'").fetchone()[0]
    matched = con.execute("SELECT COUNT(*) FROM odds_history WHERE book LIKE 'bfo_%' AND fight_id IS NOT NULL").fetchone()[0]
    print(f"\nDone. {total_rows} BFO rows, {matched} matched to fights.")
    con.close()


if __name__ == "__main__":
    main()
