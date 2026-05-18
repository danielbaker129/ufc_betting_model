"""
Supplements fighter records from Sherdog.com.
Focuses on win/loss method breakdowns and non-UFC fight history.
Updates fighters table with career record details.
"""
import hashlib
import re
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, SCRAPE_DELAY, USER_AGENT

BASE = "https://www.sherdog.com"
SEARCH = f"{BASE}/search/fighters/"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}


def get(url: str) -> BeautifulSoup:
    time.sleep(SCRAPE_DELAY)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def search_fighter(name: str) -> str | None:
    """Search Sherdog for a fighter by name, return profile URL."""
    try:
        soup = get(f"{SEARCH}{requests.utils.quote(name)}")
        result = soup.select_one("table.fightfinder_result tr:nth-child(2) td a")
        if result:
            return BASE + result["href"]
    except Exception:
        pass
    return None


def parse_record_from_profile(soup: BeautifulSoup) -> dict:
    """Extract win/loss breakdown from a Sherdog fighter profile."""
    record = {
        "total_wins": 0, "total_losses": 0, "total_draws": 0,
        "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
        "ko_losses": 0, "sub_losses": 0, "dec_losses": 0,
    }

    try:
        win_section = soup.find("div", class_="wins")
        if win_section:
            spans = win_section.find_all("span")
            for span in spans:
                label = span.get_text(strip=True).lower()
                sib = span.find_next_sibling("span")
                count = int(sib.get_text(strip=True)) if sib else 0
                if "ko" in label or "tko" in label:
                    record["ko_wins"] = count
                elif "sub" in label:
                    record["sub_wins"] = count
                elif "dec" in label:
                    record["dec_wins"] = count
            total_el = soup.find("span", class_="counter")
            if total_el:
                record["total_wins"] = int(total_el.get_text(strip=True))

        loss_section = soup.find("div", class_="loses")
        if loss_section:
            spans = loss_section.find_all("span")
            for span in spans:
                label = span.get_text(strip=True).lower()
                sib = span.find_next_sibling("span")
                count = int(sib.get_text(strip=True)) if sib else 0
                if "ko" in label or "tko" in label:
                    record["ko_losses"] = count
                elif "sub" in label:
                    record["sub_losses"] = count
                elif "dec" in label:
                    record["dec_losses"] = count
    except Exception:
        pass

    # Fallback: parse record string like "25-5-0"
    try:
        rec_el = soup.find("span", class_="record")
        if rec_el:
            m = re.search(r"(\d+)-(\d+)-(\d+)", rec_el.get_text())
            if m:
                record["total_wins"] = int(m.group(1))
                record["total_losses"] = int(m.group(2))
                record["total_draws"] = int(m.group(3))
    except Exception:
        pass

    # Count wins/losses from fight table if still zero
    if record["total_wins"] == 0 and record["total_losses"] == 0:
        try:
            rows = soup.select("table.new_table.fighter tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                result = cells[0].get_text(strip=True).lower()
                method = cells[2].get_text(strip=True).lower() if len(cells) > 2 else ""
                if result == "win":
                    record["total_wins"] += 1
                    if "ko" in method or "tko" in method:
                        record["ko_wins"] += 1
                    elif "sub" in method:
                        record["sub_wins"] += 1
                    else:
                        record["dec_wins"] += 1
                elif result == "loss":
                    record["total_losses"] += 1
                    if "ko" in method or "tko" in method:
                        record["ko_losses"] += 1
                    elif "sub" in method:
                        record["sub_losses"] += 1
                    else:
                        record["dec_losses"] += 1
                elif result in ("draw", "nc"):
                    record["total_draws"] += 1
        except Exception:
            pass

    return record


def update_fighter(con: sqlite3.Connection, fighter_id: str, name: str):
    """Search Sherdog for fighter, scrape record, update DB."""
    url = search_fighter(name)
    if not url:
        return

    try:
        soup = get(url)
        record = parse_record_from_profile(soup)
        con.execute(
            """UPDATE fighters SET
               total_wins=?, total_losses=?, total_draws=?,
               ko_wins=?, sub_wins=?, dec_wins=?,
               ko_losses=?, sub_losses=?, dec_losses=?,
               sherdog_url=?, updated_at=?
               WHERE fighter_id=?""",
            (record["total_wins"], record["total_losses"], record["total_draws"],
             record["ko_wins"], record["sub_wins"], record["dec_wins"],
             record["ko_losses"], record["sub_losses"], record["dec_losses"],
             url, datetime.utcnow().isoformat(), fighter_id),
        )
        con.commit()
    except Exception as e:
        print(f"  Sherdog error for {name}: {e}")


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    fighters = con.execute(
        "SELECT fighter_id, name FROM fighters WHERE total_wins=0 AND total_losses=0"
    ).fetchall()

    print(f"Supplementing {len(fighters)} fighters from Sherdog...")
    for i, (fighter_id, name) in enumerate(fighters):
        if i % 100 == 0:
            print(f"  ...{i}/{len(fighters)}")
        update_fighter(con, fighter_id, name)

    total = con.execute("SELECT COUNT(*) FROM fighters WHERE total_wins > 0").fetchone()[0]
    print(f"Done. {total} fighters with Sherdog records.")
    con.close()


if __name__ == "__main__":
    main()
