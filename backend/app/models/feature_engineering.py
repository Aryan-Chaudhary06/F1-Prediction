import pandas as pd
import numpy as np

POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

CIRCUIT_TYPE = {
    "Bahrain International Circuit": "high_downforce",
    "Jeddah Corniche Circuit": "street",
    "Albert Park Grand Prix Circuit": "street",
    "Suzuka Circuit": "technical",
    "Shanghai International Circuit": "technical",
    "Miami International Autodrome": "street",
    "Autodromo Enzo e Dino Ferrari": "technical",
    "Circuit de Monaco": "street",
    "Circuit de Barcelona-Catalunya": "high_downforce",
    "Circuit Gilles Villeneuve": "street",
    "Red Bull Ring": "power",
    "Silverstone Circuit": "power",
    "Hungaroring": "high_downforce",
    "Circuit de Spa-Francorchamps": "power",
    "Circuit Zandvoort": "high_downforce",
    "Autodromo Nazionale di Monza": "power",
    "Baku City Circuit": "street",
    "Marina Bay Street Circuit": "street",
    "Circuit of the Americas": "technical",
    "Autodromo Hermanos Rodriguez": "high_downforce",
    "Autodromo Jose Carlos Pace": "technical",
    "Las Vegas Strip Street Circuit": "street",
    "Lusail International Circuit": "high_downforce",
    "Yas Marina Circuit": "high_downforce",
}

def build_training_features(historical_df: pd.DataFrame) -> pd.DataFrame:
    df = historical_df.copy()
    df = df.dropna(subset=["position", "grid"])
    df["position"] = df["position"].astype(int)
    df["grid"] = df["grid"].astype(int)
    df["won"] = (df["position"] == 1).astype(int)
    df["podium"] = (df["position"] <= 3).astype(int)
    df["points_scored"] = df["position"].map(POINTS_MAP).fillna(0)

    df = df.sort_values(["driver", "year", "round"])

    df["driver_rolling_points"] = (
        df.groupby("driver")["points_scored"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["driver_rolling_wins"] = (
        df.groupby("driver")["won"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    df["driver_rolling_podiums"] = (
        df.groupby("driver")["podium"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    circuit_avg = (
        df.groupby(["driver", "circuit"])["position"]
        .mean()
        .reset_index()
        .rename(columns={"position": "driver_circuit_avg_pos"})
    )
    df = df.merge(circuit_avg, on=["driver", "circuit"], how="left")

    constructor_avg = (
        df.groupby(["constructor", "year", "round"])["points_scored"]
        .mean()
        .reset_index()
        .rename(columns={"points_scored": "constructor_avg_points"})
    )
    df = df.merge(constructor_avg, on=["constructor", "year", "round"], how="left")

    df["grid_squared"] = df["grid"] ** 2
    df["circuit_type"] = df["circuit"].map(CIRCUIT_TYPE).fillna("unknown")
    df["circuit_type_code"] = pd.Categorical(df["circuit_type"]).codes

    df["dnf"] = (~df["status"].str.contains("Finished|Lap", na=False)).astype(int)
    df["constructor_dnf_rate"] = (
        df.groupby("constructor")["dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    features = [
        "grid", "grid_squared",
        "driver_rolling_points", "driver_rolling_wins", "driver_rolling_podiums",
        "driver_circuit_avg_pos", "constructor_avg_points",
        "constructor_dnf_rate", "circuit_type_code",
        "round", "year",
    ]
    return df[features + ["won", "podium", "position", "driver",
                          "constructor", "circuit"]].copy()


# ── New-constructor / no-history fallback ───────────────────────────────────
# Audi and Cadillac entered F1 for the 2026 season with zero prior race
# history under those constructor names (Audi inherits Sauber's grid slot
# but not its constructor identity in Ergast's data model; Cadillac is a
# brand-new 11th team). Drivers on these teams therefore have NO rows in
# build_training_features()'s output for "constructor_avg_points" or
# "constructor_dnf_rate" until the team accumulates its own race history.
#
# This is used at INFERENCE time (predicting an upcoming 2026 race), not
# during training — build_training_features() above is only ever run on
# completed historical seasons, where this situation can't yet occur for
# a team that hasn't raced. Call this from the prediction-feature-row
# builder (see app.py Race Predictor page) whenever a driver's constructor
# has fewer than MIN_RACES_FOR_OWN_DATA races on record.

NEW_CONSTRUCTORS_2026 = {"Audi", "Cadillac"}
MIN_RACES_FOR_OWN_DATA = 3


def apply_new_constructor_fallback(row: dict, constructor: str,
                                    historical_features_df: pd.DataFrame) -> dict:
    """
    Mutates and returns `row` (a single inference-time feature dict) so that
    constructor-pace features fall back to the midfield average when the
    constructor has fewer than MIN_RACES_FOR_OWN_DATA races of history.

    `historical_features_df` should be the output of build_training_features()
    over whatever seasons are available (e.g. 2022-2024), used only to
    compute what "midfield average" means.
    """
    races_for_constructor = (
        historical_features_df.loc[historical_features_df["constructor"] == constructor,
                                    ["year", "round"]]
        .drop_duplicates()
        .shape[0]
    )

    if constructor in NEW_CONSTRUCTORS_2026 or races_for_constructor < MIN_RACES_FOR_OWN_DATA:
        # New constructor fallback
        midfield = (
            historical_features_df.groupby("constructor")["constructor_avg_points"]
            .mean()
            .sort_values(ascending=False)
        )
        if len(midfield) >= 6:
            midfield_avg_points = midfield.iloc[5]  # rank 6 of field ≈ midfield
        else:
            midfield_avg_points = midfield.mean() if len(midfield) else 0.0

        dnf_rates = historical_features_df.groupby("constructor")["constructor_dnf_rate"].mean()
        midfield_dnf_rate = dnf_rates.median() if len(dnf_rates) else 0.10

        row["constructor_avg_points"] = midfield_avg_points
        row["constructor_dnf_rate"] = midfield_dnf_rate

    return row


# ── Rookie / no-history fallback ─────────────────────────────────────────
# A driver with zero prior race rows in historical_features_df has no
# rolling-form or circuit-history features to draw on. The naive fallback
# of filling every feature with 0.0 is actively harmful, not just
# uninformative:
#
#   driver_circuit_avg_pos = 0.0  →  "averages a P0 finish" — a value
#   that's literally better than winning every race, and one real data
#   never produces (positions start at 1). Since the model correctly
#   learned "lower avg position → more likely podium" from real data, a
#   fabricated 0.0 reads as maximum confidence, and because it's the same
#   value on every single row for that driver regardless of round or
#   circuit, the model outputs the same near-certain prediction every
#   single time — exactly the "100% for the same driver every race" bug.
#
# Falling back to FIELD-WIDE rolling averages instead represents "unknown,
# assume roughly average" rather than "known to be superhuman". This
# mirrors qualifying_feature_engineering.apply_rookie_fallback() — keep
# both in sync if this logic changes.
def apply_rookie_fallback(row: dict, driver: str,
                          historical_features_df: pd.DataFrame) -> dict:
    """
    Mutates and returns `row` (a single inference-time feature dict) so
    that a driver with no historical rows falls back to field-wide average
    rolling-form and circuit-history features instead of fabricated zeros.

    Call this whenever driver_hist (rows for this specific driver in
    historical_features_df) is empty — i.e. a rookie with zero prior F1
    races on record — BEFORE the grid/circuit/round/year overrides are
    applied on top.
    """
    rookie_fields = [
        "driver_rolling_points", "driver_rolling_wins", "driver_rolling_podiums",
        "driver_circuit_avg_pos",
    ]
    for field in rookie_fields:
        if field not in historical_features_df.columns:
            continue
        avg = historical_features_df[field].mean()
        row[field] = float(avg) if pd.notna(avg) else 0.0

    return row