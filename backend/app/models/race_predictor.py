import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from app.models.feature_engineering import build_training_features

# ── Path resolution ──────────────────────────────────────────────────────────
# On Hugging Face Spaces, /tmp is writable but data/ is read-only (no .pkl
# can be committed due to HF binary file restrictions). We store the trained
# model in /tmp on HF and fall back to the local data/ path for development.
# Set MODEL_DIR=/tmp in HF Space secrets to activate the HF path.
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../data")
MODEL_DIR = os.getenv("MODEL_DIR", _DEFAULT_MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "race_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "race_model.meta.json")

FEATURES = [
    "grid", "grid_squared",
    "driver_rolling_points", "driver_rolling_wins", "driver_rolling_podiums",
    "driver_circuit_avg_pos", "constructor_avg_points",
    "constructor_dnf_rate", "circuit_type_code",
    "round", "year",
]

# One entry per FEATURES column, in the same order. XGBoost enforces these
# as a hard constraint on the trained trees, not a hint:
#   -1  →  prediction must be non-increasing as the feature increases
#   +1  →  prediction must be non-decreasing as the feature increases
#    0  →  no constraint (categorical / no physical monotonic meaning)
#
# Why this exists: regularization alone (min_child_weight, reg_lambda, the
# lower 2026 era-weight) reduced overfitting but didn't fix the underlying
# issue — XGBoost is free to decide grid position is "optional" if
# driver-identity-correlated rolling-form features are more separable, and
# a pole-sitter can end up predicted below a driver starting P8 purely
# because that driver's rolling stats dominated the splits. These
# constraints make that structurally impossible: worse grid can never
# increase podium probability, better recent form can never decrease it,
# etc. — regardless of how the trees would otherwise prefer to split.
MONOTONE_CONSTRAINTS = (
    -1,  # grid — worse (higher number) starting position never helps
    -1,  # grid_squared — same direction, penalizes deep grid slots more
    1,   # driver_rolling_points — more recent points never hurts
    1,   # driver_rolling_wins — more recent wins never hurts
    1,   # driver_rolling_podiums — more recent podiums never hurts
    -1,  # driver_circuit_avg_pos — worse (higher number) avg finish never helps
    1,   # constructor_avg_points — stronger team pace never hurts
    -1,  # constructor_dnf_rate — more unreliable car never helps
    0,   # circuit_type_code — categorical, no monotonic meaning
    0,   # round — no physical monotonic relationship
    0,   # year — no physical monotonic relationship
)

# ── Regulation-era sample weighting ─────────────────────────────────────────
# 2026 introduced F1's biggest regulation overhaul in over a decade (new
# aero rules, new power-unit format). Grid-position-to-result relationships,
# constructor pace hierarchies, and DNF patterns from 2022-2025 (the
# previous regulation era) are a real but noisier signal for predicting
# 2026+ races than 2026 data itself, because car characteristics changed
# significantly. Rather than drop the old-era data entirely (which would
# leave very little training data early in the new era), each row is
# weighted by which regulation era it belongs to — 2026 races count several
# times more than a single equivalent 2022-2025 race when XGBoost computes
# its loss gradient.
#
# These are deliberately simple, hand-set multipliers rather than something
# tuned via cross-validation — treat them as a reasonable starting point,
# not a finely calibrated constant. Revisit once a full 2026 season exists.
REGULATION_ERA_WEIGHTS = {
    2022: 0.6,
    2023: 0.7,
    2024: 0.85,
    2025: 1.0,    # last season under the previous regs — most comparable of the "old era"
    2026: 1.5,    # current regulations — still favored, but 3.0 let a small
                  # handful of early-2026 races (where the same few drivers
                  # both started AND finished well) dominate training enough
                  # that the model could fit them via driver-identity-
                  # correlated rolling-form features alone, without needing
                  # to learn a real grid→result relationship. See the
                  # `min_child_weight` / `reg_lambda` additions below for
                  # the other half of this fix.
}
DEFAULT_ERA_WEIGHT = 1.0  # fallback for any year not listed above


def compute_sample_weights(df: pd.DataFrame,
                           era_weights: dict = None) -> np.ndarray:
    """
    Returns a per-row weight array aligned to df's index, based on each
    row's `year` column and REGULATION_ERA_WEIGHTS (or a custom override).

    Uses numpy vectorize instead of pandas .map() — pandas map behaviour
    with dict arguments changed across 2.x versions and produced doubled
    arrays in some configurations. numpy vectorize is stable across all versions.
    """
    weights = era_weights or REGULATION_ERA_WEIGHTS
    years = df["year"].to_numpy(dtype=int)
    get_weight = np.vectorize(lambda y: weights.get(int(y), DEFAULT_ERA_WEIGHT))
    return get_weight(years)  # guaranteed 1D, same length as df


def train_model(historical_df: pd.DataFrame,
                use_era_weighting: bool = True,
                era_weights: dict = None) -> XGBClassifier:
    print("Building features...")
    df = build_training_features(historical_df)
    df = df.dropna(subset=FEATURES)

    X = df[FEATURES].values
    y = df["podium"].values
    sw = compute_sample_weights(df, era_weights) if use_era_weighting else np.ones(len(df))

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sw, test_size=0.2, random_state=42, stratify=y
    )
    # sw is already guaranteed 1D by compute_sample_weights (numpy vectorize)
    # but squeeze here as a final safety net
    sw_train = np.asarray(sw_train, dtype=float).ravel()
    sw_test  = np.asarray(sw_test,  dtype=float).ravel()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,           # back to 5 — a shallower tree (4) actively
                                # worked against grid getting its own split
                                # once rolling-form features claimed the
                                # early ones; monotone_constraints below is
                                # the real fix for the overfitting concern,
                                # not depth.
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.5,  # was 6 — that matched the raw ~5.6:1 class
                                # imbalance, but combined with the monotonic
                                # constraints it pushed recall to a perfect
                                # 1.00 at the cost of precision (0.74) — the
                                # model was flagging real podium finishers
                                # correctly every time, but also over-calling
                                # ~26% of its "podium" predictions on drivers
                                # who didn't actually podium. Lowering this
                                # trades some of that excess recall back for
                                # tighter, more trustworthy probabilities.
                                # If precision still isn't where you want it
                                # after retraining, drop this further (e.g.
                                # to 2); if recall drops too much (missing
                                # real podium finishers), bring it back up.
        min_child_weight=5,    # requires more real evidence per split — makes it
                                # harder to carve out a leaf for "just this driver"
        reg_lambda=2.0,        # L2 regularization — shrinks leaf weights, so no
                                # single feature can dominate a prediction as
                                # completely as before
        monotone_constraints=MONOTONE_CONSTRAINTS,
        tree_method="hist",    # required by XGBoost for monotone_constraints
        early_stopping_rounds=20,  # stop once held-out logloss stops improving,
                                    # instead of using all 300 rounds regardless
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train,
              sample_weight=sw_train,
              eval_set=[(X_test, y_test)],
              sample_weight_eval_set=[sw_test],
              verbose=False)

    print(f"Best iteration: {model.best_iteration} / {model.n_estimators} "
          "(if this is close to n_estimators, early stopping barely engaged)")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["No podium", "Podium"]))

    # Feature importance — printed at train time so it's easy to check
    # whether `grid`/`grid_squared` are actually influencing predictions
    # or being drowned out by driver-identity-correlated rolling-form
    # features (the failure mode that prompted the era-weight/
    # regularization changes above).
    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("Feature importance (highest to lowest):")
    for name, imp in importances:
        print(f"  {name:28s} {imp:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    years_in_training = sorted(df["year"].unique().tolist())
    with open(MODEL_META_PATH, "w") as f:
        json.dump({
            "trained_at": datetime.datetime.now().isoformat(),
            "years_trained_on": years_in_training,
            "rows_trained_on": len(df),
            "accuracy": round(float(acc), 4),
            "era_weighting_used": use_era_weighting,
            "era_weights": era_weights or REGULATION_ERA_WEIGHTS,
        }, f, indent=2)

    print(f"Model saved to {MODEL_PATH} — n_features: {model.n_features_in_}")
    return model


def load_model() -> XGBClassifier:
    """
    Load the trained model from disk. If no model file exists (e.g. first
    run on Hugging Face Spaces where .pkl files cannot be committed), raises
    a clear FileNotFoundError so the caller can trigger a retrain.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Click 'Train / Refresh Model' to train it now."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_or_train_model(historical_df: pd.DataFrame = None) -> XGBClassifier:
    """
    Try to load the model. If it doesn't exist (first run on HF Spaces or
    after cache cleared), train it automatically using historical_df.

    This is the preferred entry point for the Streamlit app — it never
    crashes on a missing model file.

    Args:
        historical_df: Raw historical race results DataFrame. Only needed
                       if the model file is missing. If None and the model
                       is missing, raises RuntimeError with a user-friendly
                       message.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Model missing — happens on first HF Spaces run or after /tmp reset
    if historical_df is None:
        raise RuntimeError(
            "No trained model found and no training data was provided. "
            "Pass historical_df to load_or_train_model() or click "
            "'Train / Refresh Model' in the UI."
        )

    print(f"No model found at {MODEL_PATH} — training now...")
    return train_model(historical_df)


def load_model_metadata() -> dict | None:
    """Returns the metadata saved alongside the model at last train time
    (trained_at, years_trained_on, accuracy, etc.), or None if the model
    was trained before this metadata file existed."""
    if not os.path.exists(MODEL_META_PATH):
        return None
    with open(MODEL_META_PATH) as f:
        return json.load(f)


def model_exists() -> bool:
    """Returns True if a trained model file exists on disk."""
    return os.path.exists(MODEL_PATH)


def model_is_stale(max_age_days: int = 7) -> bool:
    """
    Returns True if the saved model is older than max_age_days, OR if no
    metadata exists at all (e.g. very first run, or a model trained before
    this staleness tracking was added). Used to show a "data may be
    outdated" banner / offer a one-click retrain in the UI, without forcing
    a retrain on every single page load.
    """
    meta = load_model_metadata()
    if meta is None:
        return True
    trained_at = datetime.datetime.fromisoformat(meta["trained_at"])
    age = datetime.datetime.now() - trained_at
    return age > datetime.timedelta(days=max_age_days)


def predict_race(model: XGBClassifier,
                 race_features: pd.DataFrame) -> pd.DataFrame:
    df = race_features.copy()
    df = df.fillna(0)
    n = model.n_features_in_
    X = df[FEATURES].values
    if X.shape[1] < n:
        X = np.hstack([X, np.zeros((X.shape[0], n - X.shape[1]))])
    elif X.shape[1] > n:
        X = X[:, :n]
    probs = model.predict_proba(X)[:, 1]
    df["podium_probability"] = probs
    df = df.sort_values("podium_probability", ascending=False)
    df["predicted_position"] = range(1, len(df) + 1)
    return df[["driver", "podium_probability", "predicted_position"] + FEATURES].reset_index(drop=True)


def get_feature_importance(model: XGBClassifier) -> pd.DataFrame:
    importance = model.feature_importances_
    n = min(len(importance), len(FEATURES))
    return pd.DataFrame({
        "feature": FEATURES[:n],
        "importance": importance[:n]
    }).sort_values("importance", ascending=False).reset_index(drop=True)
