"""
Method-of-victory model: KO/TKO, Submission, Decision.
XGBoost multi-class classifier, chronological split.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import PROCESSED_DIR, MODELS_DIR, TRAIN_CUTOFF

FEATURE_COLS = [
    "slpm_career_diff", "slpm_L5_diff", "sapm_diff", "net_strike_diff",
    "str_acc_career_diff", "td_avg_career_diff", "td_def_diff", "sub_avg_diff",
    "ko_finish_rate_diff", "sub_finish_rate_diff", "ko_susceptibility_diff",
    "avg_fight_secs_diff", "experience_diff", "elo_diff", "reach_diff",
    "is_title_fight", "is_5_round",
]


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "feature_matrix.csv", parse_dates=["fight_date"])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    df = df.dropna(subset=["target"])
    df["method_class"] = np.where(df["method_ko"] == 1, "KO/TKO",
                          np.where(df["method_sub"] == 1, "Submission", "Decision"))
    return df


def train():
    df = load_data()
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["method_class"])

    train_df = df[df["fight_date"] < TRAIN_CUTOFF]
    test_df  = df[df["fight_date"] >= TRAIN_CUTOFF]

    X_train, y_train = train_df[FEATURE_COLS].values, train_df["label"].values
    X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df["label"].values

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    ll  = log_loss(y_test, probs)

    print(f"\n=== Method Model — Test Set Results ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"Log-Loss: {ll:.4f}")
    print(f"Classes:  {le.classes_}")

    # Per-class accuracy
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        cls_acc = (preds[mask] == i).mean() if mask.sum() > 0 else 0.0
        print(f"  {cls}: {cls_acc:.3f} acc ({mask.sum()} fights)")

    out = MODELS_DIR / "method.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le,
                     "feature_cols": FEATURE_COLS,
                     "metrics": {"accuracy": acc, "log_loss": ll}}, f)
    print(f"\nModel saved to {out}")


def predict(features_dict: dict, model_path=None) -> dict:
    path = model_path or MODELS_DIR / "method.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    le = data["label_encoder"]
    cols = data["feature_cols"]
    X = np.array([[features_dict.get(c, 0.0) for c in cols]])
    probs = model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(le.classes_, probs)}


if __name__ == "__main__":
    train()
