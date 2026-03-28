import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from app.models.feature_engineering import build_training_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../data/race_model.pkl")

FEATURES = [
    "grid", "grid_squared",
    "driver_rolling_points", "driver_rolling_wins", "driver_rolling_podiums",
    "driver_circuit_avg_pos", "constructor_avg_points",
    "constructor_dnf_rate", "circuit_type_code",
    "round", "year",
]

def train_model(historical_df: pd.DataFrame) -> XGBClassifier:
    print("Building features...")
    df = build_training_features(historical_df)
    df = df.dropna(subset=FEATURES)

    X = df[FEATURES].values
    y = df["podium"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=6,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["No podium", "Podium"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved — n_features: {model.n_features_in_}")
    return model

def load_model() -> XGBClassifier:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

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
    return df[["driver","podium_probability","predicted_position"] + FEATURES].reset_index(drop=True)

def get_feature_importance(model: XGBClassifier) -> pd.DataFrame:
    importance = model.feature_importances_
    n = min(len(importance), len(FEATURES))
    return pd.DataFrame({
        "feature": FEATURES[:n],
        "importance": importance[:n]
    }).sort_values("importance", ascending=False).reset_index(drop=True)