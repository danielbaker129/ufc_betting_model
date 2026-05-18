"""
Backfills fight_stats for any fights that are in the DB but have 0 stat rows.
Run once after fixing the stats parser.
"""
import sqlite3
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, SCRAPE_DELAY, USER_AGENT
from scrapers.ufcstats import _scrape_round_stats

HEADERS = {"User-Agent": USER_AGENT}


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    fights_missing_stats = con.execute(
        """SELECT f.fight_id, f.url, f.fighter_a_id, f.fighter_b_id
           FROM fights f
           LEFT JOIN fight_stats s ON s.fight_id = f.fight_id
           WHERE s.stat_id IS NULL AND f.url IS NOT NULL
           GROUP BY f.fight_id"""
    ).fetchall()

    print(f"Backfilling stats for {len(fights_missing_stats)} fights...")

    for i, (fight_id, url, fa_id, fb_id) in enumerate(fights_missing_stats):
        if i % 20 == 0:
            print(f"  ...{i}/{len(fights_missing_stats)}")
        try:
            time.sleep(SCRAPE_DELAY)
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            _scrape_round_stats(con, soup, fight_id, fa_id, fb_id)
            con.commit()
        except Exception as e:
            print(f"    Error {url}: {e}")

    total = con.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
    print(f"Done. {total} total stat rows in DB.")
    con.close()


if __name__ == "__main__":
    main()
