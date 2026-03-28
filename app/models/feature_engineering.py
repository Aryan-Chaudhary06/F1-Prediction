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
                          "constructor", "circuit", "year", "round"]].copy()