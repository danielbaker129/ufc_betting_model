"""
Total rounds model: predict over/under on fight length.
3-round fights: over/under 1.5 and 2.5 rounds.
5-round fights: over/under 2.5 and 4.5 rounds.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import PROCESSED_DIR, MODELS_DIR, TRAIN_CUTOFF

FEATURE_COLS = [
    "slpm_career_diff", "sapm_diff", "net_strike_diff",
    "str_acc_career_diff", "td_avg_career_diff", "td_def_diff",
    "ko_finish_rate_diff", "sub_finish_rate_diff", "avg_fight_secs_diff",
    "experience_diff", "elo_diff", "is_title_fight", "is_5_round",
    "days_since_last_a", "days_since_last_b",
]

# Absolute versions for total-based features
ABS_COLS = [
    "slpm_career_diff", "ko_finish_rate_diff", "sub_finish_rate_diff", "avg_fight_secs_diff",
]


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "feature_matrix.csv", parse_dates=["fight_date"])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    df = df.dropna(subset=["target"])
    return df


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create over/under targets from method + round columns."""
    df = df.copy()
    # fight goes the distance = Decision
    df["goes_distance"] = df["method_dec"].astype(int)
    # over 2.5 rounds for 3-round fights
    df["over_2_5_3rnd"] = (df["method_dec"] == 1).astype(int)  # dec = definitely over 2.5
    # over 4.5 rounds for 5-round fights — only decisions go 5 rounds
    df["over_4_5_5rnd"] = df["method_dec"].astype(int)
    return df


def train():
    df = load_data()
    df = make_targets(df)

    models = {}
    for target_name, desc in [
        ("goes_distance", "Goes the distance (any format)"),
        ("over_2_5_3rnd", "Over 2.5 rounds"),
    ]:
        train_df = df[df["fight_date"] < TRAIN_CUTOFF]
        test_df  = df[df["fight_date"] >= TRAIN_CUTOFF]

        X_train, y_train = train_df[FEATURE_COLS].values, train_df[target_name].values
        X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df[target_name].values

        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, probs)

        print(f"\n=== Rounds: {desc} ===")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        print(f"Base rate: {y_train.mean():.3f}")

        models[target_name] = model

    out = MODELS_DIR / "rounds.pkl"
    with open(out, "wb") as f:
        pickle.dump({"models": models, "feature_cols": FEATURE_COLS}, f)
    print(f"\nRounds models saved to {out}")


def predict(features_dict: dict, is_5_round: bool = False, model_path=None) -> dict:
    path = model_path or MODELS_DIR / "rounds.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    models = data["models"]
    cols = data["feature_cols"]
    X = np.array([[features_dict.get(c, 0.0) for c in cols]])
    result = {}
    for name, model in models.items():
        prob = model.predict_proba(X)[0, 1]
        result[name] = float(prob)
    return result


if __name__ == "__main__":
    train()
