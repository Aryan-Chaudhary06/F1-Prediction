import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit

from app.models.qualifying_feature_engineering import build_qualifying_training_features

# ── Path resolution ──────────────────────────────────────────────────────────
# Same HF Spaces-safe pattern as race_predictor.py — /tmp is writable there,
# data/ is read-only. Set MODEL_DIR=/tmp in HF Space secrets to activate it.
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../data")
MODEL_DIR = os.getenv("MODEL_DIR", _DEFAULT_MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "qualifying_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "qualifying_model.meta.json")

FEATURES = [
    "driver_5race_avg_quali_pos", "driver_quali_pos_std",
    "constructor_quali_pace_score", "circuit_type_code", "is_wet",
    "round", "year",
]

# Reuse the same regulation-era weighting rationale as race_predictor.py —
# 2026's new aero/PU rules likely shifted single-lap pace hierarchies too,
# not just race results, so the same weighting scheme applies here.
# IMPORTANT: for XGBRanker, weights are per ranking GROUP (one qualifying
# session = one group), not per row — see compute_group_weights() below.
REGULATION_ERA_WEIGHTS = {
    2022: 0.6,
    2023: 0.7,
    2024: 0.85,
    2025: 1.0,
    2026: 3.0,
}
DEFAULT_ERA_WEIGHT = 1.0


def _build_groups(df: pd.DataFrame) -> np.ndarray:
    """
    XGBRanker needs the data sorted so that rows belonging to the same
    ranking group (one qualifying session = one (year, round) pair) are
    contiguous, plus an array of group sizes. Returns the group-size array;
    mutates nothing — caller must sort `df` the same way first (see
    train_qualifying_model below, which sorts immediately before this is
    called and keeps that order all the way through .fit()).
    """
    return df.groupby(["year", "round"], sort=False).size().values


def compute_group_weights(df: pd.DataFrame, era_weights: dict = None) -> np.ndarray:
    """
    Returns one weight per ranking GROUP (i.e. one per (year, round)
    qualifying session), in the same (year, round) order _build_groups()
    will produce — NOT one weight per row. Passing a per-row array to
    XGBRanker.fit(sample_weight=...) silently corrupts training or crashes;
    it must be exactly len(groups) long.
    """
    weights = era_weights or REGULATION_ERA_WEIGHTS
    session_years = (
        df.groupby(["year", "round"], sort=False)["year"]
        .first()
    )
    return session_years.map(weights).fillna(DEFAULT_ERA_WEIGHT).values


def train_qualifying_model(historical_quali_df: pd.DataFrame,
                           use_era_weighting: bool = True,
                           era_weights: dict = None) -> XGBRanker:
    print("Building qualifying features...")
    df = build_qualifying_training_features(historical_quali_df)
    df = df.dropna(subset=FEATURES)

    # XGBRanker requires rows for the same group to be contiguous — sort by
    # (year, round) first and keep this exact order through fit().
    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    X = df[FEATURES].values
    y = df["position"].max() - df["position"].values + 1  # invert: higher = better (pole = best)
    groups = _build_groups(df)

    n_groups = len(groups)
    if n_groups < 10:
        raise ValueError(
            f"Only {n_groups} qualifying sessions in training data — "
            "need at least 10 for a meaningful train/test split. "
            "Check the year range passed to get_cached_historical_qualifying()."
        )

    group_weights = compute_group_weights(df, era_weights) if use_era_weighting else np.ones(n_groups)

    # Split at the GROUP level (whole sessions go to train or test, never
    # split mid-session) using GroupShuffleSplit, then re-expand back to
    # row-level groups/weights for whichever split each side lands in.
    group_ids = np.repeat(np.arange(n_groups), groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=group_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_group_ids = group_ids[train_idx]
    test_group_ids = group_ids[test_idx]

    # Recompute contiguous group sizes for each split, in first-seen order
    # (np.unique with return_counts sorts by value, which matches since
    # group_ids increase monotonically with the original sort order).
    _, train_groups = np.unique(train_group_ids, return_counts=True)
    _, test_groups = np.unique(test_group_ids, return_counts=True)

    train_unique_ids = np.unique(train_group_ids)
    test_unique_ids = np.unique(test_group_ids)
    train_weights = group_weights[train_unique_ids]
    test_weights = group_weights[test_unique_ids]

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="ndcg",
        verbosity=0,
    )

    print(f"Training XGBRanker on {n_groups} qualifying sessions "
          f"({len(train_groups)} train / {len(test_groups)} test)...")
    model.fit(
        X_train, y_train,
        group=train_groups,
        sample_weight=train_weights,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        sample_weight_eval_set=[test_weights],
        verbose=False,
    )

    # Evaluate: for each test session, did the model rank the actual pole
    # sitter (and top 3) correctly? A simple, interpretable accuracy proxy
    # since NDCG alone doesn't translate intuitively to "did it get pole right".
    test_df = df.iloc[test_idx].copy()
    test_df["predicted_score"] = model.predict(X_test)
    pole_correct, top3_correct, total_sessions = 0, 0, 0
    for (yr, rnd), session in test_df.groupby(["year", "round"]):
        total_sessions += 1
        predicted_order = session.sort_values("predicted_score", ascending=False)
        actual_pole = session.loc[session["position"] == 1, "driver"]
        if len(actual_pole) and predicted_order.iloc[0]["driver"] == actual_pole.iloc[0]:
            pole_correct += 1
        actual_top3 = set(session.loc[session["position"] <= 3, "driver"])
        predicted_top3 = set(predicted_order.head(3)["driver"])
        if len(actual_top3 & predicted_top3) >= 2:  # at least 2 of 3 right
            top3_correct += 1

    pole_accuracy = pole_correct / total_sessions if total_sessions else 0.0
    top3_accuracy = top3_correct / total_sessions if total_sessions else 0.0
    print(f"Pole prediction accuracy: {pole_accuracy:.1%} "
          f"({pole_correct}/{total_sessions} sessions)")
    print(f"Top-3 overlap (>=2 of 3 correct): {top3_accuracy:.1%}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    years_in_training = sorted(df["year"].unique().tolist())
    with open(MODEL_META_PATH, "w") as f:
        json.dump({
            "trained_at": datetime.datetime.now().isoformat(),
            "years_trained_on": years_in_training,
            "sessions_trained_on": n_groups,
            "rows_trained_on": len(df),
            "pole_accuracy": round(pole_accuracy, 4),
            "top3_accuracy": round(top3_accuracy, 4),
            "era_weighting_used": use_era_weighting,
            "era_weights": era_weights or REGULATION_ERA_WEIGHTS,
        }, f, indent=2)

    print(f"Qualifying model saved to {MODEL_PATH}")
    return model


def load_qualifying_model() -> XGBRanker:
    """Loads the trained qualifying ranker from disk. Raises a clear
    FileNotFoundError if missing (e.g. first run on HF Spaces)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained qualifying model found at {MODEL_PATH}. "
            "Train it first from the Qualifying Predictor page."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_or_train_qualifying_model(historical_quali_df: pd.DataFrame = None) -> XGBRanker:
    """
    Same never-crashes-on-missing-model pattern as
    race_predictor.load_or_train_model() — loads the saved ranker if it
    exists, otherwise trains a fresh one from historical_quali_df.
    """
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    if historical_quali_df is None:
        raise RuntimeError(
            "No trained qualifying model found and no training data was "
            "provided. Pass historical_quali_df, or train from the UI."
        )

    print(f"No qualifying model found at {MODEL_PATH} — training now...")
    return train_qualifying_model(historical_quali_df)


def qualifying_model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def load_qualifying_model_metadata() -> dict | None:
    if not os.path.exists(MODEL_META_PATH):
        return None
    with open(MODEL_META_PATH) as f:
        return json.load(f)


def qualifying_model_is_stale(max_age_days: int = 7) -> bool:
    meta = load_qualifying_model_metadata()
    if meta is None:
        return True
    trained_at = datetime.datetime.fromisoformat(meta["trained_at"])
    age = datetime.datetime.now() - trained_at
    return age > datetime.timedelta(days=max_age_days)


def predict_qualifying_order(model: XGBRanker,
                             session_features: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a feature dataframe for ONE upcoming qualifying session (one row
    per driver — all drivers competing in that single session, i.e. one
    ranking group) and returns it sorted into predicted qualifying order,
    P1 first, with a confidence score per driver.
    """
    df = session_features.copy()
    df = df.fillna(0)
    X = df[FEATURES].values
    scores = model.predict(X)
    df["ranker_score"] = scores

    # Confidence score: normalize ranker scores within this session to a
    # 0-100 scale, purely for display — XGBRanker's raw scores are
    # relative/unbounded, not probabilities, so this is NOT a calibrated
    # probability of pole, just a relative-confidence readout.
    score_range = df["ranker_score"].max() - df["ranker_score"].min()
    if score_range > 0:
        df["confidence"] = ((df["ranker_score"] - df["ranker_score"].min()) / score_range * 100).round(1)
    else:
        df["confidence"] = 50.0

    df = df.sort_values("ranker_score", ascending=False).reset_index(drop=True)
    df["predicted_quali_position"] = range(1, len(df) + 1)
    return df[["driver", "predicted_quali_position", "confidence", "ranker_score"] + FEATURES]


def get_qualifying_feature_importance(model: XGBRanker) -> pd.DataFrame:
    importance = model.feature_importances_
    n = min(len(importance), len(FEATURES))
    return pd.DataFrame({
        "feature": FEATURES[:n],
        "importance": importance[:n]
    }).sort_values("importance", ascending=False).reset_index(drop=True)
