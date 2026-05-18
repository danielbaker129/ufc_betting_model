"""
Computes ELO ratings for all fighters by replaying fights chronologically.
Pre-fight ELO (before the result is known) is saved as the feature.
K-factor: 170 for first 5 fights, 85 thereafter.
"""
import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, ELO_START, ELO_K_EARLY, ELO_K_LATE, ELO_K_CUTOFF


def uid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(fight_count: int) -> float:
    return ELO_K_EARLY if fight_count <= ELO_K_CUTOFF else ELO_K_LATE


def compute_elo(con: sqlite3.Connection):
    con.execute("DELETE FROM elo_history")
    con.commit()

    fights = con.execute(
        """SELECT f.fight_id, f.fighter_a_id, f.fighter_b_id,
                  f.winner_id, f.fight_date
           FROM fights f
           WHERE f.fight_date IS NOT NULL AND f.fight_date != ''
           ORDER BY f.fight_date ASC, f.fight_id ASC"""
    ).fetchall()

    ratings: dict[str, float] = {}
    fight_counts: dict[str, int] = {}
    rows = []

    for fight_id, fa_id, fb_id, winner_id, fight_date in fights:
        if not fa_id or not fb_id:
            continue

        ra = ratings.get(fa_id, ELO_START)
        rb = ratings.get(fb_id, ELO_START)
        ca = fight_counts.get(fa_id, 0)
        cb = fight_counts.get(fb_id, 0)

        ea = expected(ra, rb)
        eb = expected(rb, ra)

        if winner_id == fa_id:
            sa, sb = 1.0, 0.0
        elif winner_id == fb_id:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5  # draw or NC

        ka = k_factor(ca)
        kb = k_factor(cb)

        new_ra = ra + ka * (sa - ea)
        new_rb = rb + kb * (sb - eb)

        # Save pre-fight ELO snapshot
        rows.append((uid(fight_id, fa_id), fa_id, fight_id, fight_date, ra, new_ra, fb_id,
                     "win" if winner_id == fa_id else ("loss" if winner_id == fb_id else "draw")))
        rows.append((uid(fight_id, fb_id), fb_id, fight_id, fight_date, rb, new_rb, fa_id,
                     "win" if winner_id == fb_id else ("loss" if winner_id == fa_id else "draw")))

        ratings[fa_id] = new_ra
        ratings[fb_id] = new_rb
        fight_counts[fa_id] = ca + 1
        fight_counts[fb_id] = cb + 1

    con.executemany(
        """INSERT OR REPLACE INTO elo_history
           (elo_id, fighter_id, fight_id, fight_date, elo_before, elo_after, opponent_id, result)
           VALUES (?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()

    total_fighters = len(ratings)
    top5 = sorted(ratings.items(), key=lambda x: -x[1])[:5]
    print(f"ELO computed for {total_fighters} fighters, {len(rows)//2} fights.")
    print("Top 5 current ELO ratings:")
    for fid, elo in top5:
        name = con.execute("SELECT name FROM fighters WHERE fighter_id=?", (fid,)).fetchone()
        print(f"  {name[0] if name else fid}: {elo:.1f}")


def main():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    fights = con.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    print(f"Computing ELO across {fights} fights...")
    compute_elo(con)
    con.close()


if __name__ == "__main__":
    main()
