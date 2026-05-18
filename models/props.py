"""
Fighter props regression models.
Predicts expected significant strikes and takedowns landed per fighter.
Uses expected fight time as a scaling factor.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import PROCESSED_DIR, MODELS_DIR, DB_PATH, TRAIN_CUTOFF

import sqlite3


def load_fight_level_data():
    """Load per-fight stats (not differentials) for prop modeling."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT f.fight_id, f.fight_date, f.fighter_a_id, f.fighter_b_id,
                  f.is_title_fight, f.weight_class,
                  SUM(CASE WHEN s.fighter_id = f.fighter_a_id THEN s.sig_str_landed END) AS a_sig_landed,
                  SUM(CASE WHEN s.fighter_id = f.fighter_b_id THEN s.sig_str_landed END) AS b_sig_landed,
                  SUM(CASE WHEN s.fighter_id = f.fighter_a_id THEN s.td_landed END) AS a_td_landed,
                  SUM(CASE WHEN s.fighter_id = f.fighter_b_id THEN s.td_landed END) AS b_td_landed,
                  f.round, f.time, f.time_format
           FROM fights f
           JOIN fight_stats s ON s.fight_id = f.fight_id
           WHERE f.fight_date IS NOT NULL
           GROUP BY f.fight_id""",
        con,
    )
    con.close()
    return df


def parse_fight_minutes(row) -> float:
    try:
        rnd = int(row["round"]) if row["round"] else 1
        t = str(row["time"]).strip()
        m, s = t.split(":")
        return max(((rnd - 1) * 300 + int(m) * 60 + int(s)) / 60.0, 0.1)
    except Exception:
        rnd = row["round"] or 1
        return float(rnd) * 5.0


def train():
    feat_df = pd.read_csv(PROCESSED_DIR / "feature_matrix.csv", parse_dates=["fight_date"])
    fight_df = load_fight_level_data()
    fight_df["fight_date"] = pd.to_datetime(fight_df["fight_date"])
    fight_df["fight_mins"] = fight_df.apply(parse_fight_minutes, axis=1)

    merged = feat_df.merge(fight_df[["fight_id", "fight_mins", "a_sig_landed", "b_sig_landed",
                                      "a_td_landed", "b_td_landed"]], on="fight_id", how="inner")

    # Per-minute targets
    merged["a_slpm_actual"] = merged["a_sig_landed"] / merged["fight_mins"]
    merged["b_slpm_actual"] = merged["b_sig_landed"] / merged["fight_mins"]
    merged["a_tdpm_actual"] = merged["a_td_landed"] / merged["fight_mins"]

    FEATURE_COLS = [
        "slpm_career_diff", "slpm_L5_diff", "sapm_diff", "str_acc_career_diff",
        "td_avg_career_diff", "td_def_diff", "sub_avg_diff",
        "ko_finish_rate_diff", "avg_fight_secs_diff",
        "experience_diff", "elo_diff", "is_title_fight", "is_5_round",
    ]
    merged = merged.dropna(subset=FEATURE_COLS)

    train_df = merged[merged["fight_date"] < TRAIN_CUTOFF]
    test_df  = merged[merged["fight_date"] >= TRAIN_CUTOFF]

    models = {}
    for target, desc in [
        ("a_slpm_actual", "Fighter A sig strikes per minute"),
        ("b_slpm_actual", "Fighter B sig strikes per minute"),
        ("a_tdpm_actual", "Fighter A takedowns per minute"),
    ]:
        y_train = train_df[target].clip(0, 20).values
        y_test  = test_df[target].clip(0, 20).values
        X_train = train_df[FEATURE_COLS].values
        X_test  = test_df[FEATURE_COLS].values

        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test).clip(0)

        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        print(f"{desc}: MAE={mae:.3f} R²={r2:.3f}")

        models[target] = model

    out = MODELS_DIR / "props.pkl"
    with open(out, "wb") as f:
        pickle.dump({"models": models, "feature_cols": FEATURE_COLS}, f)
    print(f"\nProps models saved to {out}")


def predict(features_dict: dict, expected_fight_mins: float = 12.5, model_path=None) -> dict:
    """Returns expected strikes/TDs for the fight given expected duration."""
    path = model_path or MODELS_DIR / "props.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    models = data["models"]
    cols = data["feature_cols"]
    X = np.array([[features_dict.get(c, 0.0) for c in cols]])

    result = {}
    for name, model in models.items():
        rate = max(float(model.predict(X)[0]), 0.0)
        total = rate * expected_fight_mins
        result[name.replace("_actual", "_per_min")] = rate
        result[name.replace("_actual", "_expected_total")] = total

    return result


if __name__ == "__main__":
    train()
