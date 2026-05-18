"""
Moneyline model: predict fight winner (red corner = 1).
LightGBM + Optuna tuning + isotonic calibration.
Chronological train/test split — no data leakage.
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH, PROCESSED_DIR, MODELS_DIR, TRAIN_CUTOFF

FEATURE_COLS = [
    # Career stats
    "slpm_career_diff", "slpm_L5_diff", "slpm_L3_diff",
    "sapm_diff", "net_strike_diff",
    "str_acc_career_diff", "str_acc_L5_diff",
    "td_avg_career_diff", "td_avg_L5_diff",
    "td_acc_career_diff", "td_def_diff", "sub_avg_diff",
    "ko_finish_rate_diff", "sub_finish_rate_diff", "ko_susceptibility_diff",
    "avg_fight_secs_diff", "experience_diff", "win_streak_diff",
    "elo_diff", "reach_diff", "height_diff", "age_diff",
    "days_since_last_a", "days_since_last_b",
    # Recent form (lambda=0.65 — last 3 fights dominate)
    "slpm_recent_diff", "str_acc_recent_diff", "td_avg_recent_diff",
    # Trend / trajectory — negative = declining
    "slpm_trend_diff", "str_acc_trend_diff", "win_rate_trend_diff",
    "recent_win_rate_diff", "last_fight_won_diff",
    # Age × decline interaction (old + declining = large negative)
    "age_decline_diff",
    # Contextual
    "is_title_fight", "is_5_round",
    "both_orthodox", "a_southpaw_vs_orthodox", "b_southpaw_vs_orthodox",
    # Market prior — 0.5 for pre-2022 fights (no data), real DK/FD no-vig for 2022+.
    "market_nv_a",
]


def load_data():
    path = PROCESSED_DIR / "feature_matrix.csv"
    df = pd.read_csv(path, parse_dates=["fight_date"])
    df = df.dropna(subset=["target"])          # only require the label
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)   # 0 = neutral for differentials
    return df


def train():
    df = load_data()
    train_df = df[df["fight_date"] < TRAIN_CUTOFF]
    test_df  = df[df["fight_date"] >= TRAIN_CUTOFF]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["target"].values

    print(f"Train: {len(X_train)} fights | Test: {len(X_test)} fights")
    print(f"Train date range: {train_df['fight_date'].min().date()} → {train_df['fight_date'].max().date()}")
    print(f"Test  date range: {test_df['fight_date'].min().date()} → {test_df['fight_date'].max().date()}")

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_train[train_idx], y_train[train_idx],
                  eval_set=[(X_train[val_idx], y_train[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            prob = m.predict_proba(X_train[val_idx])[:, 1]
            scores.append(log_loss(y_train[val_idx], prob))
        return np.mean(scores)

    print("\nRunning Optuna tuning (50 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best = study.best_params
    print(f"Best params: {best}")

    base_model = lgb.LGBMClassifier(**best, verbosity=-1)
    base_model.fit(X_train, y_train)

    print("Calibrating probabilities (isotonic, 5-fold CV on training set)...")
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    ll    = log_loss(y_test, probs)

    print(f"\n=== Moneyline Model — Test Set Results ===")
    print(f"Accuracy:    {acc:.4f}  ({acc*100:.1f}%)")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Brier Score: {brier:.4f}  (lower=better)")
    print(f"Log-Loss:    {ll:.4f}  (lower=better)")

    # Feature importance
    fi = pd.Series(base_model.feature_importances_, index=FEATURE_COLS)
    print("\nTop 10 features:")
    print(fi.nlargest(10).to_string())

    out = MODELS_DIR / "moneyline.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS,
                     "metrics": {"accuracy": acc, "auc": auc, "brier": brier, "log_loss": ll}}, f)
    print(f"\nModel saved to {out}")
    return model


def predict(features_dict: dict, model_path=None) -> dict:
    path = model_path or MODELS_DIR / "moneyline.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    cols = data["feature_cols"]
    X = np.array([[features_dict.get(c, 0.0) for c in cols]])
    prob = model.predict_proba(X)[0, 1]
    return {"prob_a_wins": float(prob), "prob_b_wins": float(1 - prob)}


if __name__ == "__main__":
    train()
