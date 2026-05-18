"""
Re-links orphaned odds_history rows (fight_id IS NULL) to fights in the DB.

The original scrapers used a strict 6-char prefix match which fails on
nicknames/abbreviations (e.g. "Alex Volkanovski" vs "Alexander Volkanovski").
This script uses difflib for fuzzy matching and a longer common-prefix check.

Run after any scraper that may have left unlinked rows.
"""
import re
import sqlite3
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def name_score(a: str, b: str) -> float:
    """Similarity score between two normalized fighter names, 0-1."""
    if not a or not b:
        return 0.0
    # Exact match
    if a == b:
        return 1.0
    # One is a prefix of the other (handles Alex/Alexander, etc.)
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if long.startswith(short) and len(short) >= 4:
        return 0.95
    # Shared prefix of 6+ chars
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
        else:
            break
    if prefix_len >= 6:
        return 0.85
    # Fallback: difflib ratio
    return SequenceMatcher(None, a, b).ratio()


def last_name(name: str) -> str:
    """Normalized last name — last space-delimited token."""
    parts = name.strip().split()
    return normalize(parts[-1]) if parts else ""


def name_variants(name: str) -> list[str]:
    """Return normalized name plus reversed token order (handles Last First / First Last)."""
    n = normalize(name)
    parts = name.strip().split()
    if len(parts) >= 2:
        rev = normalize(" ".join(reversed(parts)))
        return [n, rev] if rev != n else [n]
    return [n]


def find_fight_id(con, name_a: str, name_b: str, fight_date: str):
    """
    Return (fight_id, swapped) using fuzzy matching with last-name and
    name-reversal fallbacks. Searches fights within ±5 days of fight_date.
    """
    variants_a = name_variants(name_a)
    variants_b = name_variants(name_b)
    la, lb = last_name(name_a), last_name(name_b)
    if not variants_a[0] or not variants_b[0]:
        return None, False

    rows = con.execute(
        """SELECT f.fight_id, fa.name, fb.name
           FROM fights f
           JOIN fighters fa ON fa.fighter_id = f.fighter_a_id
           JOIN fighters fb ON fb.fighter_id = f.fighter_b_id
           WHERE f.fight_date BETWEEN date(?, '-5 days') AND date(?, '+5 days')""",
        (fight_date, fight_date),
    ).fetchall()

    best_score, best_id, best_swapped = 0.0, None, False
    for fight_id, fn_a, fn_b in rows:
        nfa, nfb = normalize(fn_a), normalize(fn_b)
        lfa, lfb = last_name(fn_a), last_name(fn_b)

        # Try all variant combinations for name_a and name_b
        for va in variants_a:
            for vb in variants_b:
                def combined(s1, s2):
                    # Require the stronger match to be high; tolerate a weaker match
                    # on the other fighter (handles nickname/spelling variants)
                    hi, lo = max(s1, s2), min(s1, s2)
                    if hi >= 0.95 and lo >= 0.55:
                        return (hi + lo) / 2
                    return min(s1, s2)
                score_normal  = combined(name_score(va, nfa), name_score(vb, nfb))
                score_swapped = combined(name_score(va, nfb), name_score(vb, nfa))

                # Last-name exact match bonus
                if la == lfa and lb == lfb:
                    score_normal  = max(score_normal, 0.9)
                if la == lfb and lb == lfa:
                    score_swapped = max(score_swapped, 0.9)
                # Reversed last-name match (handles Last First storage)
                la_rev = last_name(" ".join(reversed(name_a.split())))
                lb_rev = last_name(" ".join(reversed(name_b.split())))
                if la_rev == lfa and lb_rev == lfb:
                    score_normal  = max(score_normal, 0.88)
                if la_rev == lfb and lb_rev == lfa:
                    score_swapped = max(score_swapped, 0.88)

                if score_normal >= score_swapped and score_normal > best_score:
                    best_score, best_id, best_swapped = score_normal, fight_id, False
                elif score_swapped > score_normal and score_swapped > best_score:
                    best_score, best_id, best_swapped = score_swapped, fight_id, True

    # Accept if combined score is strong enough.
    # Using 0.72 instead of min-of-two=0.75 to handle cases where one fighter
    # is a perfect match (1.0) but the other is a moderate match (~0.55).
    if best_score >= 0.72:
        return best_id, best_swapped
    return None, False


def relink(dry_run: bool = False):
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")

    unlinked = con.execute(
        """SELECT odds_id, fighter_a_name, fighter_b_name, fight_date, fighter_a_odds, fighter_b_odds, book
           FROM odds_history
           WHERE fight_id IS NULL
             AND fighter_a_name IS NOT NULL
             AND fighter_b_name IS NOT NULL
           ORDER BY fight_date"""
    ).fetchall()

    print(f"Unlinked odds rows: {len(unlinked)}")
    linked = 0
    skipped = 0

    for odds_id, fa_name, fb_name, fight_date, fa_odds, fb_odds, book in unlinked:
        fight_id, swapped = find_fight_id(con, fa_name, fb_name, fight_date)
        if fight_id is None:
            skipped += 1
            continue

        if swapped:
            # Odds are stored for (fa_name, fb_name) but the DB fight has them reversed
            fa_odds, fb_odds = fb_odds, fa_odds

        if not dry_run:
            con.execute(
                """UPDATE odds_history
                   SET fight_id = ?,
                       fighter_a_odds = ?,
                       fighter_b_odds = ?
                   WHERE odds_id = ?""",
                (fight_id, fa_odds, fb_odds, odds_id),
            )
        linked += 1

    if not dry_run:
        con.commit()

    action = "Would link" if dry_run else "Linked"
    print(f"{action}: {linked}  |  Still unmatched: {skipped}")
    con.close()
    return linked


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        print("DRY RUN — no DB changes")
    relink(dry_run=dry)
