"""
Scrapes ufcstats.com: events → fights → per-round stats → fighter profiles.
Saves everything to SQLite. Safe to re-run (upserts on conflict).
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

BASE = "http://ufcstats.com"
HEADERS = {"User-Agent": USER_AGENT}


def get(url: str) -> BeautifulSoup:
    time.sleep(SCRAPE_DELAY)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def parse_time_to_seconds(t: str) -> int:
    t = t.strip()
    if not t or t == "--":
        return 0
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def parse_pct(s: str) -> float:
    s = s.strip().replace("%", "")
    try:
        return float(s) / 100.0
    except Exception:
        return 0.0


def parse_of(s: str):
    s = s.strip()
    if " of " in s:
        parts = s.split(" of ")
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0
    try:
        return int(s), int(s)
    except Exception:
        return 0, 0


def parse_height(h: str) -> float:
    h = h.strip()
    m = re.match(r"(\d+)' ?(\d+)\"?", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return 0.0


def parse_reach(r: str) -> float:
    r = r.strip().replace('"', "").replace("--", "0")
    try:
        return float(r)
    except Exception:
        return 0.0


def scrape_events(con: sqlite3.Connection) -> list[dict]:
    print("Fetching event list...")
    soup = get(f"{BASE}/statistics/events/completed?page=all")
    rows = soup.select("table.b-statistics__table-events tbody tr.b-statistics__table-row")
    events = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        a = cells[0].find("a")
        if not a:
            continue
        url = a["href"].strip()
        name = a.get_text(strip=True)
        date_el = cells[0].find("span", class_="b-statistics__date")
        date_str = date_el.get_text(strip=True) if date_el else ""
        try:
            date_iso = datetime.strptime(date_str, "%B %d, %Y").strftime("%Y-%m-%d")
        except Exception:
            date_iso = date_str
        location = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        event_id = uid(url)
        events.append({"event_id": event_id, "name": name, "date": date_iso, "location": location, "url": url})

    con.executemany(
        """INSERT OR IGNORE INTO events (event_id, name, date, location, url)
           VALUES (:event_id, :name, :date, :location, :url)""",
        events,
    )
    con.commit()
    print(f"  Saved {len(events)} events")
    return events


def scrape_fighters(con: sqlite3.Connection, urls: list[str]) -> dict[str, str]:
    """Scrape fighter profile pages. Returns {url: fighter_id}."""
    url_to_id = {}
    existing = {r[0]: r[1] for r in con.execute("SELECT url, fighter_id FROM fighters").fetchall()}

    to_scrape = [u for u in urls if u not in existing]
    print(f"  Scraping {len(to_scrape)} new fighter profiles (skipping {len(existing)} cached)...")

    for i, url in enumerate(to_scrape):
        if i % 50 == 0 and i > 0:
            print(f"    ...{i}/{len(to_scrape)} fighters done")
        try:
            soup = get(url)
            name_el = soup.find("span", class_="b-content__title-highlight")
            name = name_el.get_text(strip=True) if name_el else "Unknown"
            nickname_el = soup.find("p", class_="b-content__Nickname")
            nickname = nickname_el.get_text(strip=True) if nickname_el else ""

            info = {}
            for li in soup.select("ul.b-list__box-list li.b-list__box-list-item"):
                text = li.get_text(" ", strip=True)
                if ":" in text:
                    k, _, v = text.partition(":")
                    info[k.strip().lower()] = v.strip()

            dob = info.get("dob", "")
            try:
                dob = datetime.strptime(dob, "%b %d, %Y").strftime("%Y-%m-%d")
            except Exception:
                pass

            fighter_id = uid(url)
            row = {
                "fighter_id": fighter_id,
                "name": name,
                "nickname": nickname,
                "dob": dob,
                "height_inches": parse_height(info.get("height", "")),
                "reach_inches": parse_reach(info.get("reach", "")),
                "stance": info.get("stance", ""),
                "weight_class": "",
                "url": url,
                "updated_at": datetime.utcnow().isoformat(),
            }
            con.execute(
                """INSERT OR IGNORE INTO fighters
                   (fighter_id, name, nickname, dob, height_inches, reach_inches,
                    stance, weight_class, url, updated_at)
                   VALUES (:fighter_id,:name,:nickname,:dob,:height_inches,
                           :reach_inches,:stance,:weight_class,:url,:updated_at)""",
                row,
            )
            con.commit()
            url_to_id[url] = fighter_id
        except Exception as e:
            print(f"    Fighter scrape error {url}: {e}")

    for url, fid in existing.items():
        url_to_id[url] = fid

    return url_to_id


def scrape_event_fights(con: sqlite3.Connection, event: dict, fighter_url_map: dict) -> list[str]:
    """Scrape an event page; return list of fight URLs."""
    try:
        soup = get(event["url"])
    except Exception as e:
        print(f"  Event page error {event['url']}: {e}")
        return []

    fight_rows = soup.select("table.b-fight-details__table tbody tr.b-fight-details__table-row")
    fight_urls = []

    for row in fight_rows:
        a = row.find("a")
        if not a or "href" not in a.attrs:
            continue
        fight_url = a["href"].strip()
        fight_urls.append(fight_url)

    return fight_urls


def scrape_fight(con: sqlite3.Connection, fight_url: str, event: dict, fighter_url_map: dict):
    """Scrape a fight detail page and save fight + per-round stats."""
    if con.execute("SELECT 1 FROM fights WHERE url=?", (fight_url,)).fetchone():
        return

    try:
        soup = get(fight_url)
    except Exception as e:
        print(f"    Fight page error {fight_url}: {e}")
        return

    try:
        # Fighter names + links
        fighter_links = soup.select("div.b-fight-details__person a")
        if len(fighter_links) < 2:
            return
        fa_url = fighter_links[0]["href"].strip()
        fb_url = fighter_links[1]["href"].strip()

        # Ensure fighters are scraped
        for fu in [fa_url, fb_url]:
            if fu not in fighter_url_map:
                new_map = scrape_fighters(con, [fu])
                fighter_url_map.update(new_map)

        fa_id = fighter_url_map.get(fa_url)
        fb_id = fighter_url_map.get(fb_url)
        if not fa_id or not fb_id:
            return

        # Winner
        winner_status = [el.get_text(strip=True) for el in soup.select("div.b-fight-details__person i.b-fight-details__person-status")]
        winner_id = None
        if len(winner_status) >= 1 and winner_status[0] == "W":
            winner_id = fa_id
        elif len(winner_status) >= 2 and winner_status[1] == "W":
            winner_id = fb_id

        # Method, round, time
        detail_items = soup.select("div.b-fight-details__content p.b-fight-details__text")
        method, method_detail, rnd, fight_time, time_format = "", "", None, "", ""
        for item in detail_items:
            text = item.get_text(" ", strip=True)
            if text.startswith("Method:"):
                parts = text.replace("Method:", "").strip().split()
                method = parts[0] if parts else ""
                method_detail = " ".join(parts[1:]) if len(parts) > 1 else ""
            elif text.startswith("Round:"):
                try:
                    rnd = int(text.replace("Round:", "").strip())
                except Exception:
                    pass
            elif text.startswith("Time:"):
                fight_time = text.replace("Time:", "").strip()
            elif text.startswith("Time format:"):
                time_format = text.replace("Time format:", "").strip()

        is_title = 1 if soup.find(string=re.compile(r"Title Bout", re.I)) else 0

        # Weight class from fight page
        wc_el = soup.find(string=re.compile(r"Bout", re.I))
        weight_class = ""
        if wc_el:
            parent = wc_el.parent if wc_el.parent else None
            if parent:
                full = parent.get_text(" ", strip=True)
                wc_match = re.search(r"(\w[\w\s]+(?:weight|Heavyweight|Flyweight|Strawweight))", full, re.I)
                if wc_match:
                    weight_class = wc_match.group(1).strip()

        fight_id = uid(fight_url)
        con.execute(
            """INSERT OR IGNORE INTO fights
               (fight_id, event_id, fighter_a_id, fighter_b_id, winner_id,
                method, method_detail, round, time, time_format,
                is_title_fight, weight_class, url, fight_date)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fight_id, event["event_id"], fa_id, fb_id, winner_id,
             method, method_detail, rnd, fight_time, time_format,
             is_title, weight_class, fight_url, event["date"]),
        )

        # Per-round stats
        _scrape_round_stats(con, soup, fight_id, fa_id, fb_id)
        con.commit()

    except Exception as e:
        print(f"    Fight parse error {fight_url}: {e}")


def _split_dual_cell(cell_text: str):
    """
    ufcstats packs both fighters into one cell.
    Formats: "0 0" | "6 of 14 5 of 15" | "42% 33%" | "5:00 3:30"
    Returns (val_a, val_b) as strings.
    """
    t = cell_text.strip()
    # "X of Y A of B"
    m = re.match(r"^(\d+\s+of\s+\d+)\s+(\d+\s+of\s+\d+)$", t)
    if m:
        return m.group(1), m.group(2)
    # "X% Y%"
    m = re.match(r"^(\d+%)\s+(\d+%)$", t)
    if m:
        return m.group(1), m.group(2)
    # "M:SS M:SS"
    m = re.match(r"^(\d+:\d+)\s+(\d+:\d+)$", t)
    if m:
        return m.group(1), m.group(2)
    # "X Y" (two plain integers)
    m = re.match(r"^(\d+)\s+(\d+)$", t)
    if m:
        return m.group(1), m.group(2)
    # Single value (fallback)
    return t, t


def _scrape_round_stats(con, soup, fight_id, fa_id, fb_id):
    """
    Extract per-round stats from a fight detail page.
    ufcstats.com structure:
      Table 0: KD | Sig.Str | Sig.Str% | Total.Str | TD | TD% | Sub | Rev | Ctrl
      Table 1: Sig.Str | Sig.Str% | Head | Body | Leg | Distance | Clinch | Ground
    Each td row = one round (row index 1 → round 1, etc.).
    Both fighters' values are packed in each cell.
    """
    tables = soup.find_all("table", class_="b-fight-details__table")
    round_data: dict[int, dict] = {}

    # ufcstats has 4 tables per fight page: totals + per-round for each of the two stat sections.
    # We only care about the per-round tables (even indices = main stats, odd indices = breakdown).
    # Each td row in a per-round table corresponds to a round (row 0 → round 1, etc.).
    # Table structure: Table 0/1 = totals (skip), Table 2/3 = per-round per-section.
    # Actually all tables alternate: totals then per-round. We detect per-round by row count > 1.

    for table_idx, table in enumerate(tables):
        rows = table.find_all("tr")
        # Skip header row; data rows are td rows only
        td_rows = [r for r in rows if r.find("td")]
        if not td_rows:
            continue

        is_breakdown = table_idx % 2 == 1  # odd = head/body/leg/distance/clinch/ground

        for rnd_idx, row in enumerate(td_rows):
            rnd_num = rnd_idx + 1  # row 0 = round 1
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if not cells or len(cells) < 2:
                continue

            if rnd_num not in round_data:
                round_data[rnd_num] = {"a": {}, "b": {}}
            rd = round_data[rnd_num]

            try:
                if not is_breakdown and len(cells) >= 9:
                    # Main stats: Fighter | KD | Sig.Str | Sig.% | Total.Str | TD | TD% | Sub | Rev | Ctrl
                    kd_a,  kd_b  = _split_dual_cell(cells[1])
                    sig_a, sig_b = _split_dual_cell(cells[2])
                    spct_a,spct_b= _split_dual_cell(cells[3])
                    tot_a, tot_b = _split_dual_cell(cells[4])
                    td_a,  td_b  = _split_dual_cell(cells[5])
                    tpct_a,tpct_b= _split_dual_cell(cells[6])
                    sub_a, sub_b = _split_dual_cell(cells[7])
                    rev_a, rev_b = _split_dual_cell(cells[8])
                    ctrl_a,ctrl_b= _split_dual_cell(cells[9]) if len(cells) > 9 else ("0:00","0:00")

                    sl_a, sa_a = parse_of(sig_a)
                    sl_b, sa_b = parse_of(sig_b)
                    tl_a, ta_a = parse_of(tot_a)
                    tl_b, ta_b = parse_of(tot_b)
                    dl_a, da_a = parse_of(td_a)
                    dl_b, da_b = parse_of(td_b)

                    rd["a"].update({
                        "knockdowns": _int(kd_a),
                        "sig_str_landed": sl_a, "sig_str_attempted": sa_a,
                        "sig_str_pct": parse_pct(spct_a),
                        "total_str_landed": tl_a, "total_str_attempted": ta_a,
                        "td_landed": dl_a, "td_attempted": da_a,
                        "td_pct": parse_pct(tpct_a),
                        "sub_attempts": _int(sub_a), "rev": _int(rev_a),
                        "ctrl_seconds": parse_time_to_seconds(ctrl_a),
                    })
                    rd["b"].update({
                        "knockdowns": _int(kd_b),
                        "sig_str_landed": sl_b, "sig_str_attempted": sa_b,
                        "sig_str_pct": parse_pct(spct_b),
                        "total_str_landed": tl_b, "total_str_attempted": ta_b,
                        "td_landed": dl_b, "td_attempted": da_b,
                        "td_pct": parse_pct(tpct_b),
                        "sub_attempts": _int(sub_b), "rev": _int(rev_b),
                        "ctrl_seconds": parse_time_to_seconds(ctrl_b),
                    })

                elif is_breakdown and len(cells) >= 7:
                    # Breakdown: Fighter | Sig.Str | Sig.% | Head | Body | Leg | Distance | Clinch | Ground
                    head_a,  head_b  = _split_dual_cell(cells[3])
                    body_a,  body_b  = _split_dual_cell(cells[4])
                    leg_a,   leg_b   = _split_dual_cell(cells[5])
                    dist_a,  dist_b  = _split_dual_cell(cells[6]) if len(cells) > 6 else ("0 of 0","0 of 0")
                    clinch_a,clinch_b= _split_dual_cell(cells[7]) if len(cells) > 7 else ("0 of 0","0 of 0")
                    ground_a,ground_b= _split_dual_cell(cells[8]) if len(cells) > 8 else ("0 of 0","0 of 0")

                    hl_a,ha_a = parse_of(head_a);  hl_b,ha_b = parse_of(head_b)
                    bl_a,ba_a = parse_of(body_a);  bl_b,ba_b = parse_of(body_b)
                    ll_a,la_a = parse_of(leg_a);   ll_b,la_b = parse_of(leg_b)
                    dl_a,da_a = parse_of(dist_a);  dl_b,da_b = parse_of(dist_b)
                    cl_a,ca_a = parse_of(clinch_a);cl_b,ca_b = parse_of(clinch_b)
                    gl_a,ga_a = parse_of(ground_a);gl_b,ga_b = parse_of(ground_b)

                    rd["a"].update({
                        "head_landed": hl_a, "head_attempted": ha_a,
                        "body_landed": bl_a, "body_attempted": ba_a,
                        "leg_landed": ll_a, "leg_attempted": la_a,
                        "distance_landed": dl_a, "distance_attempted": da_a,
                        "clinch_landed": cl_a, "clinch_attempted": ca_a,
                        "ground_landed": gl_a, "ground_attempted": ga_a,
                    })
                    rd["b"].update({
                        "head_landed": hl_b, "head_attempted": ha_b,
                        "body_landed": bl_b, "body_attempted": ba_b,
                        "leg_landed": ll_b, "leg_attempted": la_b,
                        "distance_landed": dl_b, "distance_attempted": da_b,
                        "clinch_landed": cl_b, "clinch_attempted": ca_b,
                        "ground_landed": gl_b, "ground_attempted": ga_b,
                    })
            except Exception:
                pass

    # Insert round stats
    for rnd_num, rd in round_data.items():
        for fighter_id, stats in [(fa_id, rd.get("a", {})), (fb_id, rd.get("b", {}))]:
            if not stats:
                continue
            stat_id = uid(fight_id, fighter_id, rnd_num)
            con.execute(
                """INSERT OR IGNORE INTO fight_stats
                   (stat_id, fight_id, fighter_id, round,
                    knockdowns, sig_str_landed, sig_str_attempted, sig_str_pct,
                    total_str_landed, total_str_attempted,
                    td_landed, td_attempted, td_pct,
                    sub_attempts, rev, ctrl_seconds,
                    head_landed, head_attempted, body_landed, body_attempted,
                    leg_landed, leg_attempted, distance_landed, distance_attempted,
                    clinch_landed, clinch_attempted, ground_landed, ground_attempted)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (stat_id, fight_id, fighter_id, rnd_num,
                 stats.get("knockdowns", 0),
                 stats.get("sig_str_landed", 0), stats.get("sig_str_attempted", 0),
                 stats.get("sig_str_pct", 0.0),
                 stats.get("total_str_landed", 0), stats.get("total_str_attempted", 0),
                 stats.get("td_landed", 0), stats.get("td_attempted", 0),
                 stats.get("td_pct", 0.0),
                 stats.get("sub_attempts", 0), stats.get("rev", 0),
                 stats.get("ctrl_seconds", 0),
                 stats.get("head_landed", 0), stats.get("head_attempted", 0),
                 stats.get("body_landed", 0), stats.get("body_attempted", 0),
                 stats.get("leg_landed", 0), stats.get("leg_attempted", 0),
                 stats.get("distance_landed", 0), stats.get("distance_attempted", 0),
                 stats.get("clinch_landed", 0), stats.get("clinch_attempted", 0),
                 stats.get("ground_landed", 0), stats.get("ground_attempted", 0)),
            )


def _int(s: str) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    events = scrape_events(con)
    fighter_url_map: dict[str, str] = {
        r[0]: r[1] for r in con.execute("SELECT url, fighter_id FROM fighters").fetchall()
    }

    print(f"\nProcessing {len(events)} events...")
    for i, event in enumerate(events):
        existing_fights = con.execute(
            "SELECT COUNT(*) FROM fights WHERE event_id=?", (event["event_id"],)
        ).fetchone()[0]
        if existing_fights > 0:
            continue

        print(f"  [{i+1}/{len(events)}] {event['name']} ({event['date']})")
        fight_urls = scrape_event_fights(con, event, fighter_url_map)

        # Collect all fighter URLs from this event first, then batch-scrape
        fighter_urls_needed = []
        for fu in fight_urls:
            try:
                time.sleep(SCRAPE_DELAY)
                soup = get(fu)
                for a in soup.select("div.b-fight-details__person a"):
                    href = a.get("href", "").strip()
                    if href and href not in fighter_url_map:
                        fighter_urls_needed.append(href)
            except Exception:
                pass

        if fighter_urls_needed:
            new_map = scrape_fighters(con, list(set(fighter_urls_needed)))
            fighter_url_map.update(new_map)

        for fu in fight_urls:
            scrape_fight(con, fu, event, fighter_url_map)

        fights_done = con.execute(
            "SELECT COUNT(*) FROM fights WHERE event_id=?", (event["event_id"],)
        ).fetchone()[0]
        print(f"    → {fights_done} fights saved")

    total = con.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    fighters = con.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
    stats = con.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
    print(f"\nDone. {total} fights, {fighters} fighters, {stats} stat rows in DB.")
    con.close()


if __name__ == "__main__":
    main()
