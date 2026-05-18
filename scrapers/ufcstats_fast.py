"""
Fast bulk scraper for ufcstats.com — 3 decoupled phases.
Phase 1: Collect all fight URLs from all event pages   (~774 requests)
Phase 2: Scrape all fight pages for results + stats   (~7000 requests)
Phase 3: Scrape all fighter profile pages              (~3000 requests)

0.5s delay — respectful but ~3x faster than the original.
Safe to re-run: skips already-scraped fights/fighters.
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

DELAY = 0.5
BASE = "http://ufcstats.com"
HDR = {"User-Agent": USER_AGENT}


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def get(url: str) -> BeautifulSoup:
    time.sleep(DELAY)
    r = requests.get(url, headers=HDR, timeout=10)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def parse_time_s(t: str) -> int:
    try:
        m, s = t.strip().split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def parse_pct(s: str) -> float:
    try:
        return float(s.strip().replace("%", "")) / 100.0
    except Exception:
        return 0.0


def parse_of(s: str):
    s = s.strip()
    if " of " in s:
        a, b = s.split(" of ", 1)
        try:
            return int(a.strip()), int(b.strip())
        except Exception:
            return 0, 0
    try:
        v = int(s)
        return v, v
    except Exception:
        return 0, 0


def parse_height(h: str) -> float:
    m = re.match(r"(\d+)'\s*(\d+)", h.strip())
    return int(m.group(1)) * 12 + int(m.group(2)) if m else 0.0


def parse_reach(r: str) -> float:
    try:
        return float(r.strip().replace('"', "").replace("--", "0"))
    except Exception:
        return 0.0


def _int(s) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0


def _split(cell: str):
    """Split a dual-fighter cell into (val_a, val_b)."""
    t = cell.strip()
    m = re.match(r"^(\d+\s+of\s+\d+)\s+(\d+\s+of\s+\d+)$", t)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(\d+%)\s+(\d+%)$", t)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(\d+:\d+)\s+(\d+:\d+)$", t)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(\d+)\s+(\d+)$", t)
    if m:
        return m.group(1), m.group(2)
    return t, t


# ─── PHASE 1: COLLECT EVENT + FIGHT URLS ─────────────────────────────────────

def phase1_collect_fight_urls(con: sqlite3.Connection) -> list[tuple]:
    """Returns list of (fight_url, event_id, event_date)."""
    print("\n=== PHASE 1: Collecting fight URLs from all events ===")
    soup = get(f"{BASE}/statistics/events/completed?page=all")
    event_rows = soup.select("table.b-statistics__table-events tbody tr.b-statistics__table-row")

    events_to_process = []
    for row in event_rows:
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

        con.execute("INSERT OR IGNORE INTO events (event_id, name, date, location, url) VALUES (?,?,?,?,?)",
                    (event_id, name, date_iso, location, url))
        events_to_process.append((event_id, url, date_iso))

    con.commit()
    print(f"  {len(events_to_process)} events in DB")

    # Check which events already have fights
    already_done = {
        r[0] for r in con.execute("SELECT DISTINCT event_id FROM fights").fetchall()
    }

    all_fight_refs = []
    todo = [e for e in events_to_process if e[0] not in already_done]
    print(f"  {len(todo)} events need fight URL collection")

    for i, (event_id, event_url, date_iso) in enumerate(todo):
        if i % 50 == 0 and i > 0:
            print(f"  ...{i}/{len(todo)} events scanned")
        try:
            esoup = get(event_url)
            for row in esoup.select("table.b-fight-details__table tbody tr.b-fight-details__table-row"):
                a = row.find("a")
                if a and "href" in a.attrs:
                    all_fight_refs.append((a["href"].strip(), event_id, date_iso))
        except Exception as e:
            print(f"    Event error {event_url}: {e}")

    print(f"  {len(all_fight_refs)} new fight URLs collected")
    return all_fight_refs


# ─── PHASE 2: SCRAPE FIGHT PAGES ─────────────────────────────────────────────

def phase2_scrape_fights(con: sqlite3.Connection, fight_refs: list[tuple]):
    print(f"\n=== PHASE 2: Scraping {len(fight_refs)} fight pages ===")
    done = {r[0] for r in con.execute("SELECT url FROM fights WHERE url IS NOT NULL").fetchall()}
    todo = [f for f in fight_refs if f[0] not in done]
    print(f"  {len(todo)} fights to scrape (skipping {len(done)} already done)")

    fighter_urls_seen: set[str] = set()

    for i, (fight_url, event_id, fight_date) in enumerate(todo):
        if i % 100 == 0 and i > 0:
            print(f"  ...{i}/{len(todo)} fights done")
        try:
            soup = get(fight_url)
            _process_fight(con, soup, fight_url, event_id, fight_date, fighter_urls_seen)
            con.commit()
        except Exception as e:
            print(f"    Fight error {fight_url}: {e}")

    print(f"  Collected {len(fighter_urls_seen)} unique fighter URLs to scrape in phase 3")
    return fighter_urls_seen


def _process_fight(con, soup, fight_url, event_id, fight_date, fighter_urls_seen: set):
    # Fighter URLs
    fighter_links = soup.select("div.b-fight-details__person a")
    if len(fighter_links) < 2:
        return
    fa_url = fighter_links[0]["href"].strip()
    fb_url = fighter_links[1]["href"].strip()
    fighter_urls_seen.add(fa_url)
    fighter_urls_seen.add(fb_url)

    fa_id = uid(fa_url)
    fb_id = uid(fb_url)

    # Ensure stub fighters exist so FK constraints don't fail
    fa_name = fighter_links[0].get_text(strip=True)
    fb_name = fighter_links[1].get_text(strip=True)
    con.execute("INSERT OR IGNORE INTO fighters (fighter_id, name, url) VALUES (?,?,?)", (fa_id, fa_name, fa_url))
    con.execute("INSERT OR IGNORE INTO fighters (fighter_id, name, url) VALUES (?,?,?)", (fb_id, fb_name, fb_url))

    # Winner
    statuses = [el.get_text(strip=True) for el in soup.select("div.b-fight-details__person i.b-fight-details__person-status")]
    winner_id = None
    if len(statuses) >= 1 and statuses[0] == "W":
        winner_id = fa_id
    elif len(statuses) >= 2 and statuses[1] == "W":
        winner_id = fb_id

    # Method / round / time
    method = method_detail = time_format = fight_time = ""
    rnd = None
    for item in soup.select("div.b-fight-details__content p.b-fight-details__text"):
        text = item.get_text(" ", strip=True)
        if text.startswith("Method:"):
            parts = text.replace("Method:", "").strip().split(None, 1)
            method = parts[0] if parts else ""
            method_detail = parts[1] if len(parts) > 1 else ""
        elif text.startswith("Round:"):
            try:
                rnd = int(text.replace("Round:", "").strip())
            except Exception:
                pass
        elif text.startswith("Time:"):
            fight_time = text.replace("Time:", "").strip()
        elif text.startswith("Time format:"):
            time_format = text.replace("Time format:", "").strip()

    is_title = 1 if soup.find(string=re.compile(r"title bout", re.I)) else 0

    # Weight class
    weight_class = ""
    for el in soup.find_all(string=re.compile(r"(weight|Heavyweight|Flyweight|Strawweight)", re.I)):
        parent_text = el.parent.get_text(" ", strip=True) if el.parent else ""
        wm = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Weight|Heavyweight|Flyweight|Strawweight))", parent_text)
        if wm:
            weight_class = wm.group(1).strip()
            break

    fight_id = uid(fight_url)
    con.execute(
        """INSERT OR IGNORE INTO fights
           (fight_id, event_id, fighter_a_id, fighter_b_id, winner_id,
            method, method_detail, round, time, time_format,
            is_title_fight, weight_class, url, fight_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (fight_id, event_id, fa_id, fb_id, winner_id,
         method, method_detail, rnd, fight_time, time_format,
         is_title, weight_class, fight_url, fight_date),
    )

    # Per-round stats
    _scrape_stats(con, soup, fight_id, fa_id, fb_id)


def _scrape_stats(con, soup, fight_id, fa_id, fb_id):
    tables = soup.find_all("table", class_="b-fight-details__table")
    round_data: dict[int, dict] = {}

    for table_idx, table in enumerate(tables):
        is_breakdown = (table_idx % 2 == 1)
        td_rows = [r for r in table.find_all("tr") if r.find("td")]

        for rnd_idx, row in enumerate(td_rows):
            rnd_num = rnd_idx + 1
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 2:
                continue
            if rnd_num not in round_data:
                round_data[rnd_num] = {"a": {}, "b": {}}
            rd = round_data[rnd_num]
            try:
                if not is_breakdown and len(cells) >= 9:
                    kd_a, kd_b   = _split(cells[1])
                    sig_a, sig_b = _split(cells[2])
                    sp_a, sp_b   = _split(cells[3])
                    tot_a, tot_b = _split(cells[4])
                    td_a, td_b   = _split(cells[5])
                    tp_a, tp_b   = _split(cells[6])
                    sub_a, sub_b = _split(cells[7])
                    rev_a, rev_b = _split(cells[8])
                    ctrl_a, ctrl_b = _split(cells[9]) if len(cells) > 9 else ("0:00", "0:00")

                    sl_a, sa_a = parse_of(sig_a); sl_b, sa_b = parse_of(sig_b)
                    tl_a, ta_a = parse_of(tot_a); tl_b, ta_b = parse_of(tot_b)
                    dl_a, da_a = parse_of(td_a);  dl_b, da_b = parse_of(td_b)

                    rd["a"].update({"knockdowns": _int(kd_a), "sig_str_landed": sl_a, "sig_str_attempted": sa_a,
                                    "sig_str_pct": parse_pct(sp_a), "total_str_landed": tl_a, "total_str_attempted": ta_a,
                                    "td_landed": dl_a, "td_attempted": da_a, "td_pct": parse_pct(tp_a),
                                    "sub_attempts": _int(sub_a), "rev": _int(rev_a), "ctrl_seconds": parse_time_s(ctrl_a)})
                    rd["b"].update({"knockdowns": _int(kd_b), "sig_str_landed": sl_b, "sig_str_attempted": sa_b,
                                    "sig_str_pct": parse_pct(sp_b), "total_str_landed": tl_b, "total_str_attempted": ta_b,
                                    "td_landed": dl_b, "td_attempted": da_b, "td_pct": parse_pct(tp_b),
                                    "sub_attempts": _int(sub_b), "rev": _int(rev_b), "ctrl_seconds": parse_time_s(ctrl_b)})

                elif is_breakdown and len(cells) >= 6:
                    head_a, head_b   = _split(cells[3])
                    body_a, body_b   = _split(cells[4])
                    leg_a,  leg_b    = _split(cells[5])
                    dist_a, dist_b   = _split(cells[6]) if len(cells) > 6 else ("0 of 0", "0 of 0")
                    clinch_a, clinch_b = _split(cells[7]) if len(cells) > 7 else ("0 of 0", "0 of 0")
                    ground_a, ground_b = _split(cells[8]) if len(cells) > 8 else ("0 of 0", "0 of 0")

                    hl_a, ha_a = parse_of(head_a);   hl_b, ha_b = parse_of(head_b)
                    bl_a, ba_a = parse_of(body_a);   bl_b, ba_b = parse_of(body_b)
                    ll_a, la_a = parse_of(leg_a);    ll_b, la_b = parse_of(leg_b)
                    dl_a, da_a = parse_of(dist_a);   dl_b, da_b = parse_of(dist_b)
                    cl_a, ca_a = parse_of(clinch_a); cl_b, ca_b = parse_of(clinch_b)
                    gl_a, ga_a = parse_of(ground_a); gl_b, ga_b = parse_of(ground_b)

                    rd["a"].update({"head_landed": hl_a, "head_attempted": ha_a, "body_landed": bl_a, "body_attempted": ba_a,
                                    "leg_landed": ll_a, "leg_attempted": la_a, "distance_landed": dl_a, "distance_attempted": da_a,
                                    "clinch_landed": cl_a, "clinch_attempted": ca_a, "ground_landed": gl_a, "ground_attempted": ga_a})
                    rd["b"].update({"head_landed": hl_b, "head_attempted": ha_b, "body_landed": bl_b, "body_attempted": ba_b,
                                    "leg_landed": ll_b, "leg_attempted": la_b, "distance_landed": dl_b, "distance_attempted": da_b,
                                    "clinch_landed": cl_b, "clinch_attempted": ca_b, "ground_landed": gl_b, "ground_attempted": ga_b})
            except Exception:
                pass

    for rnd_num, rd in round_data.items():
        for fighter_id, stats in [(fa_id, rd.get("a", {})), (fb_id, rd.get("b", {}))]:
            if not stats:
                continue
            stat_id = uid(fight_id, fighter_id, rnd_num)
            con.execute(
                """INSERT OR IGNORE INTO fight_stats
                   (stat_id, fight_id, fighter_id, round,
                    knockdowns, sig_str_landed, sig_str_attempted, sig_str_pct,
                    total_str_landed, total_str_attempted, td_landed, td_attempted, td_pct,
                    sub_attempts, rev, ctrl_seconds,
                    head_landed, head_attempted, body_landed, body_attempted,
                    leg_landed, leg_attempted, distance_landed, distance_attempted,
                    clinch_landed, clinch_attempted, ground_landed, ground_attempted)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (stat_id, fight_id, fighter_id, rnd_num,
                 stats.get("knockdowns", 0), stats.get("sig_str_landed", 0), stats.get("sig_str_attempted", 0),
                 stats.get("sig_str_pct", 0.0), stats.get("total_str_landed", 0), stats.get("total_str_attempted", 0),
                 stats.get("td_landed", 0), stats.get("td_attempted", 0), stats.get("td_pct", 0.0),
                 stats.get("sub_attempts", 0), stats.get("rev", 0), stats.get("ctrl_seconds", 0),
                 stats.get("head_landed", 0), stats.get("head_attempted", 0),
                 stats.get("body_landed", 0), stats.get("body_attempted", 0),
                 stats.get("leg_landed", 0), stats.get("leg_attempted", 0),
                 stats.get("distance_landed", 0), stats.get("distance_attempted", 0),
                 stats.get("clinch_landed", 0), stats.get("clinch_attempted", 0),
                 stats.get("ground_landed", 0), stats.get("ground_attempted", 0)),
            )


# ─── PHASE 3: SCRAPE FIGHTER PROFILES ────────────────────────────────────────

def phase3_scrape_fighters(con: sqlite3.Connection, fighter_urls: set[str]):
    print(f"\n=== PHASE 3: Scraping fighter profiles ===")
    # Also get any fighters in DB that haven't had their profiles filled
    db_urls = {r[0] for r in con.execute(
        "SELECT url FROM fighters WHERE url IS NOT NULL AND (height_inches IS NULL OR height_inches = 0)"
    ).fetchall()}
    all_urls = (fighter_urls | db_urls) - {None, ""}

    print(f"  {len(all_urls)} fighter profiles to scrape")

    for i, url in enumerate(all_urls):
        if i % 100 == 0 and i > 0:
            already = con.execute("SELECT COUNT(*) FROM fighters WHERE height_inches > 0").fetchone()[0]
            print(f"  ...{i}/{len(all_urls)} fighters done ({already} with full profiles)")
        try:
            soup = get(url)
            _process_fighter(con, soup, url)
            con.commit()
        except Exception as e:
            print(f"    Fighter error {url}: {e}")


def _process_fighter(con, soup, url):
    fighter_id = uid(url)
    name_el = soup.find("span", class_="b-content__title-highlight")
    name = name_el.get_text(strip=True) if name_el else "Unknown"

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

    # Win/loss record from the fights section
    wins = losses = draws = ko_wins = sub_wins = dec_wins = ko_losses = sub_losses = dec_losses = 0
    fight_rows = soup.select("table.b-fight-details__table tbody tr")
    for row in fight_rows:
        cells = [td.get_text(strip=True).lower() for td in row.find_all("td")]
        if not cells:
            continue
        result = cells[0] if cells else ""
        method = cells[7] if len(cells) > 7 else ""
        if result == "win":
            wins += 1
            if "ko" in method or "tko" in method:
                ko_wins += 1
            elif "sub" in method:
                sub_wins += 1
            else:
                dec_wins += 1
        elif result == "loss":
            losses += 1
            if "ko" in method or "tko" in method:
                ko_losses += 1
            elif "sub" in method:
                sub_losses += 1
            else:
                dec_losses += 1
        elif result in ("draw", "nc", "no contest"):
            draws += 1

    con.execute(
        """INSERT OR REPLACE INTO fighters
           (fighter_id, name, dob, height_inches, reach_inches, stance, url,
            total_wins, total_losses, total_draws,
            ko_wins, sub_wins, dec_wins, ko_losses, sub_losses, dec_losses, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (fighter_id, name, dob,
         parse_height(info.get("height", "")),
         parse_reach(info.get("reach", "")),
         info.get("stance", ""),
         url,
         wins, losses, draws,
         ko_wins, sub_wins, dec_wins,
         ko_losses, sub_losses, dec_losses,
         datetime.utcnow().isoformat()),
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

CHECKPOINT = Path(__file__).parent.parent / "data" / "raw" / "fight_urls_checkpoint.txt"


def save_checkpoint(fight_refs: list):
    with open(CHECKPOINT, "w") as f:
        for url, eid, date in fight_refs:
            f.write(f"{url}\t{eid}\t{date}\n")


def load_checkpoint() -> list | None:
    if not CHECKPOINT.exists():
        return None
    refs = []
    with open(CHECKPOINT) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                refs.append((parts[0], parts[1], parts[2]))
    return refs if refs else None


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")

    t0 = time.time()

    cached = load_checkpoint()
    if cached:
        print(f"\n=== Resuming from checkpoint: {len(cached)} fight URLs ===")
        fight_refs = cached
    else:
        fight_refs = phase1_collect_fight_urls(con)
        save_checkpoint(fight_refs)

    fighter_urls = phase2_scrape_fights(con, fight_refs)

    phase3_scrape_fighters(con, fighter_urls)

    total_fights = con.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    total_fighters = con.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
    total_stats = con.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
    elapsed = (time.time() - t0) / 60

    print(f"\n{'='*50}")
    print(f"Scraping complete in {elapsed:.1f} minutes")
    print(f"  {total_fights} fights | {total_fighters} fighters | {total_stats} stat rows")
    print(f"{'='*50}")
    con.close()


if __name__ == "__main__":
    main()
