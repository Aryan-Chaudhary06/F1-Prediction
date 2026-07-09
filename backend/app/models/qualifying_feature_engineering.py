"""
app/models/qualifying_feature_engineering.py
───────────────────────────────────────────────
Feature pipeline for the Qualifying Predictor — a separate model from the
race podium predictor (race_predictor.py). Qualifying is a single-lap pace
contest, not a race-craft/strategy contest, so it gets its own feature set
built mostly from rolling QUALIFYING history rather than race results.

Mirrors the structure of app/models/feature_engineering.py (same column
naming conventions, same new-constructor fallback pattern) so the two
pipelines stay easy to read side by side, but the two are intentionally
NOT merged into one shared function — race position and qualifying
position are different targets with different signal, and forcing them
through one function would make both harder to follow.
"""

import pandas as pd
import numpy as np

from app.models.feature_engineering import CIRCUIT_TYPE

# Re-export the same circuit-type fallback constants used for race
# predictions, so "new constructor" means the same thing in both models.
NEW_CONSTRUCTORS_2026 = {"Audi", "Cadillac"}
MIN_RACES_FOR_OWN_DATA = 3


def _lap_time_to_seconds(t) -> float:
    """Converts a Jolpica lap-time string like '1:23.456' to seconds
    (83.456). Returns NaN for missing/unparseable values (e.g. a driver
    who didn't set a time in Q2/Q3 because they were knocked out)."""
    if pd.isna(t) or t in (None, ""):
        return np.nan
    try:
        if ":" in str(t):
            mins, secs = str(t).split(":")
            return int(mins) * 60 + float(secs)
        return float(t)
    except (ValueError, TypeError):
        return np.nan


def build_qualifying_training_features(historical_quali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the output of ergast_client.get_cached_historical_qualifying()
    (one row per driver per session: year, round, circuit, driver,
    constructor, position, q1, q2, q3) and returns a feature dataframe
    ready for training the qualifying ranker.

    Target: `position` (final qualifying position, 1 = pole).
    """
    df = historical_quali_df.copy()
    df = df.dropna(subset=["position"])
    df["position"] = df["position"].astype(int)

    # Best lap time across Q1/Q2/Q3 — whichever session a driver's best lap
    # came from. Drivers knocked out in Q1 only ever set a Q1 time, so this
    # naturally handles missing Q2/Q3 values.
    for q in ["q1", "q2", "q3"]:
        df[f"{q}_seconds"] = df[q].apply(_lap_time_to_seconds)
    df["best_lap_seconds"] = df[["q1_seconds", "q2_seconds", "q3_seconds"]].min(axis=1)

    # Gap to the session's pole time, as a percentage — a normalized pace
    # measure that's comparable across different circuits (a 0.5s gap means
    # very different things at Monaco vs Monza in absolute terms).
    pole_time = df.groupby(["year", "round"])["best_lap_seconds"].transform("min")
    df["gap_to_pole_pct"] = ((df["best_lap_seconds"] - pole_time) / pole_time * 100).fillna(0)

    df = df.sort_values(["driver", "year", "round"])

    # Rolling qualifying form — the core signal for this model. Shifted by
    # 1 so a driver's CURRENT result never leaks into their own feature row.
    df["driver_5race_avg_quali_pos"] = (
        df.groupby("driver")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["driver_quali_pos_std"] = (
        df.groupby("driver")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )

    # Constructor single-lap pace — average gap-to-pole for the team,
    # independent of which of its two drivers set the time.
    constructor_pace = (
        df.groupby(["constructor", "year", "round"])["gap_to_pole_pct"]
        .mean()
        .reset_index()
        .rename(columns={"gap_to_pole_pct": "constructor_quali_pace_score"})
    )
    df = df.merge(constructor_pace, on=["constructor", "year", "round"], how="left")
    # Use only PRIOR races for this feature too — merge above pulls in the
    # same-race value, so roll it backward per constructor afterward.
    df = df.sort_values(["constructor", "year", "round"])
    df["constructor_quali_pace_score"] = (
        df.groupby("constructor")["constructor_quali_pace_score"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df = df.sort_values(["driver", "year", "round"])
    df["circuit_type"] = df["circuit"].map(CIRCUIT_TYPE).fillna("unknown")
    df["circuit_type_code"] = pd.Categorical(df["circuit_type"]).codes

    is_wet_placeholder = 0  # see note in qualifying_predictor.py docstring —
    # historical wet/dry flag isn't available from Jolpica; left as a
    # constant column here so the FEATURES list stays stable even though
    # at the moment it carries no signal. The Streamlit page still lets
    # the user pick a weather condition for clarity/future use.
    df["is_wet"] = is_wet_placeholder

    features = [
        "driver_5race_avg_quali_pos", "driver_quali_pos_std",
        "constructor_quali_pace_score", "circuit_type_code", "is_wet",
        "round", "year",
    ]
    return df[features + ["position", "driver", "constructor", "circuit"]].copy()


def apply_qualifying_new_constructor_fallback(row: dict, constructor: str,
                                              historical_features_df: pd.DataFrame) -> dict:
    """
    Same idea as feature_engineering.apply_new_constructor_fallback(), but
    for the qualifying pace feature: a brand-new constructor (or one with
    fewer than MIN_RACES_FOR_OWN_DATA qualifying sessions on record) gets
    the midfield-average gap-to-pole instead of a missing/zero value.
    """
    races_for_constructor = (
        historical_features_df.loc[historical_features_df["constructor"] == constructor,
                                    ["year", "round"]]
        .drop_duplicates()
        .shape[0]
    )

    if constructor in NEW_CONSTRUCTORS_2026 or races_for_constructor < MIN_RACES_FOR_OWN_DATA:
        # New constructor fallback
        pace_by_constructor = (
            historical_features_df.groupby("constructor")["constructor_quali_pace_score"]
            .mean()
            .sort_values()  # ascending — smaller gap-to-pole % = faster
        )
        if len(pace_by_constructor) >= 6:
            midfield_pace = pace_by_constructor.iloc[5]
        else:
            midfield_pace = pace_by_constructor.mean() if len(pace_by_constructor) else 1.5

        row["constructor_quali_pace_score"] = midfield_pace

    return row


def apply_rookie_fallback(row: dict, driver_name: str, rookie_names: set,
                          historical_features_df: pd.DataFrame) -> dict:
    """
    A driver with no F1 qualifying history at all (a rookie, by car number
    or by name) has no rolling-form features to draw on. Rather than feed
    the model zeros (which would read as "an extremely fast, perfectly
    consistent driver" — zero gap, zero std dev — the opposite of reality),
    fall back to the FIELD-WIDE rolling average, which represents "unknown,
    assume average" rather than "known to be exceptional."

    This is deliberately a blunter fallback than the constructor one above —
    there's no good proxy for an individual driver's pace before they've
    driven the car, so "assume average" is the honest answer. The UI is
    responsible for labeling these predictions as projected/low-confidence
    (see the 🆕 rookie badge in the Qualifying Predictor page).
    """
    if driver_name in rookie_names:
        row["driver_5race_avg_quali_pos"] = historical_features_df["driver_5race_avg_quali_pos"].mean()
        row["driver_quali_pos_std"] = historical_features_df["driver_quali_pos_std"].mean()
    return row
