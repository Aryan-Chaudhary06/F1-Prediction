import pandas as pd
import numpy as np
from app.data.ergast_client import get_historical_results

CIRCUIT_TYPES = {
    "street":        ["Monaco","Azerbaijan","Singapore","Miami","Las Vegas","Jeddah","Baku"],
    "power":         ["Monza","Spa","Silverstone","Austria","Canada"],
    "technical":     ["Hungary","Japan","Suzuka","COTA","United States","Mexico","São Paulo","Brazil"],
    "high_downforce":["Bahrain","Spain","Barcelona","Abu Dhabi","Qatar","Netherlands","Zandvoort","Lusail"],
}

POINTS_MAP = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}

def classify_circuit(circuit_name: str) -> str:
    for ctype, circuits in CIRCUIT_TYPES.items():
        if any(c.lower() in circuit_name.lower() for c in circuits):
            return ctype
    return "technical"

def build_driver_dna(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build driver DNA profiles — average performance score per circuit type.
    Returns DataFrame with columns: driver, street, power, technical, high_downforce,
    consistency, race_craft
    """
    df = historical_df.copy()
    df = df.dropna(subset=["position","grid"])
    df["position"]     = df["position"].astype(int)
    df["grid"]         = df["grid"].astype(int)
    df["points"]       = df["position"].map(POINTS_MAP).fillna(0)
    df["circuit_type"] = df["circuit"].apply(classify_circuit)
    df["dnf"]          = (~df["status"].str.contains("Finished|Lap", na=False)).astype(int)
    df["positions_gained"] = df["grid"] - df["position"]

    results = []
    for driver in df["driver"].unique():
        d = df[df["driver"] == driver]
        if len(d) < 5:
            continue

        row = {"driver": driver}

        # Circuit type avg points (normalized 0-100)
        for ctype in ["street","power","technical","high_downforce"]:
            sub = d[d["circuit_type"] == ctype]
            if len(sub) >= 2:
                avg = sub["points"].mean()
            else:
                avg = d["points"].mean() * 0.7
            row[ctype] = round(avg, 2)

        # Consistency = inverse of position variance (normalized)
        pos_std = d["position"].std()
        row["consistency"] = round(max(0, 20 - pos_std), 2)

        # Race craft = avg positions gained from grid
        row["race_craft"] = round(
            max(0, 10 + d["positions_gained"].mean()), 2
        )

        results.append(row)

    dna_df = pd.DataFrame(results)

    # Normalize all columns to 0-100
    for col in ["street","power","technical","high_downforce","consistency","race_craft"]:
        if col in dna_df.columns:
            max_val = dna_df[col].max()
            min_val = dna_df[col].min()
            if max_val > min_val:
                dna_df[col] = ((dna_df[col] - min_val) / (max_val - min_val) * 100).round(1)
            else:
                dna_df[col] = 50.0

    return dna_df.sort_values("street", ascending=False).reset_index(drop=True)