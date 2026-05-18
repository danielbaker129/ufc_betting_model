"""
Fetches live/upcoming UFC odds from The Odds API.
Returns DraftKings and FanDuel odds per fight, line-shops best available.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import ODDS_API_KEY

API_BASE = "https://api.the-odds-api.com/v4"
SPORT    = "mma_mixed_martial_arts"

# Books we care about — map API key → display name
TARGET_BOOKS = {
    "draftkings": "DraftKings",
    "fanduel":    "FanDuel",
}


def fetch_upcoming_odds(regions: str = "us", markets: str = "h2h") -> list[dict]:
    url = f"{API_BASE}/sports/{SPORT}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": ",".join(TARGET_BOOKS.keys()),
    }
    r = requests.get(url, params=params, timeout=15)
    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    print(f"  Odds API: {used} used, {remaining} remaining this month")
    r.raise_for_status()
    return r.json()


def to_decimal(american: int) -> float:
    if american > 0:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def parse_fights(raw: list[dict]) -> list[dict]:
    """
    Parse API response into fight dicts with per-book odds.
    Returns best DK line, best FD line, and best overall line for each fighter.
    """
    fights = []
    for event in raw:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        book_odds: dict[str, dict] = {}   # book_key -> {home_odds, away_odds}

        for book in event.get("bookmakers", []):
            key = book["key"]
            if key not in TARGET_BOOKS:
                continue
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                if home in outcomes and away in outcomes:
                    book_odds[key] = {
                        "name": TARGET_BOOKS[key],
                        "home": outcomes[home],
                        "away": outcomes[away],
                    }

        if not book_odds:
            continue

        # Line-shop: best odds for home = highest decimal odds
        def best_for(fighter_key: str):
            best_odds, best_book = None, None
            for bk, bo in book_odds.items():
                o = bo[fighter_key]
                if best_odds is None or to_decimal(o) > to_decimal(best_odds):
                    best_odds, best_book = o, TARGET_BOOKS[bk]
            return best_odds, best_book

        best_home_odds, best_home_book = best_for("home")
        best_away_odds, best_away_book = best_for("away")

        fights.append({
            "event_id":      event.get("id"),
            "fighter_a":     home,
            "fighter_b":     away,
            "commence_time": commence,
            # Best line available across books
            "odds_a":        best_home_odds,
            "odds_b":        best_away_odds,
            "best_book_a":   best_home_book,
            "best_book_b":   best_away_book,
            # Per-book breakdown
            "draftkings_a":  book_odds.get("draftkings", {}).get("home"),
            "draftkings_b":  book_odds.get("draftkings", {}).get("away"),
            "fanduel_a":     book_odds.get("fanduel", {}).get("home"),
            "fanduel_b":     book_odds.get("fanduel", {}).get("away"),
            "books":         book_odds,
        })

    return fights


def get_upcoming_fights() -> list[dict]:
    raw    = fetch_upcoming_odds()
    fights = parse_fights(raw)
    print(f"  Found {len(fights)} upcoming UFC fights with DK/FD odds")
    return fights


if __name__ == "__main__":
    fights = get_upcoming_fights()
    for f in fights[:5]:
        dk_a = f.get("draftkings_a")
        fd_a = f.get("fanduel_a")
        print(f"  {f['fighter_a']} vs {f['fighter_b']}")
        print(f"    DK:  {f['fighter_a']} {dk_a:+d}  /  {f['fighter_b']} {f.get('draftkings_b'):+d}" if dk_a else "    DK: n/a")
        print(f"    FD:  {f['fighter_a']} {fd_a:+d}  /  {f['fighter_b']} {f.get('fanduel_b'):+d}" if fd_a else "    FD: n/a")
        print(f"    Best A: {f['odds_a']:+d} @ {f['best_book_a']}  |  Best B: {f['odds_b']:+d} @ {f['best_book_b']}")
