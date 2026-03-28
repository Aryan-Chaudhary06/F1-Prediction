import pandas as pd
import numpy as np
from typing import Dict, List

POINTS_SYSTEM = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

def simulate_race(driver_strengths: Dict[str, float],
                  noise_std: float = 0.15) -> List[str]:
    scores = {}
    for driver, strength in driver_strengths.items():
        noise = np.random.normal(0, noise_std)
        scores[driver] = max(0, strength + noise)
    return sorted(scores, key=scores.get, reverse=True)

def simulate_season(
    current_standings: pd.DataFrame,
    remaining_races: int,
    driver_strengths: Dict[str, float],
    n_simulations: int = 10000,
) -> pd.DataFrame:
    drivers = current_standings["driver"].tolist()
    base_points = dict(zip(
        current_standings["driver"],
        current_standings["points"]
    ))

    win_counts = {d: 0 for d in drivers}
    final_points_sum = {d: 0.0 for d in drivers}

    for _ in range(n_simulations):
        season_points = base_points.copy()
        for race_num in range(remaining_races):
            finish_order = simulate_race(driver_strengths)
            for pos, driver in enumerate(finish_order, 1):
                if driver in season_points:
                    pts = POINTS_SYSTEM.get(pos, 0)
                    season_points[driver] = season_points.get(driver, 0) + pts
                    if pos == 1:
                        season_points[driver] += 1

        champion = max(season_points, key=season_points.get)
        win_counts[champion] += 1
        for d in drivers:
            final_points_sum[d] += season_points.get(d, 0)

    results = []
    for driver in drivers:
        results.append({
            "driver": driver,
            "wdc_probability": round(win_counts[driver] / n_simulations * 100, 1),
            "avg_final_points": round(final_points_sum[driver] / n_simulations, 1),
            "current_points": base_points.get(driver, 0),
        })

    return pd.DataFrame(results).sort_values(
        "wdc_probability", ascending=False
    ).reset_index(drop=True)

def build_driver_strengths(standings: pd.DataFrame) -> Dict[str, float]:
    max_pts = standings["points"].max()
    if max_pts == 0:
        return {d: 0.5 for d in standings["driver"]}
    strengths = {}
    for _, row in standings.iterrows():
        base = row["points"] / max_pts
        strengths[row["driver"]] = round(0.3 + 0.7 * base, 4)
    return strengths