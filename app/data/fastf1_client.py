import fastf1
import pandas as pd
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../data/cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def get_session(year: int, gp: str, session_type: str = "R"):
    """
    Load a FastF1 session.
    session_type: 'R' = Race, 'Q' = Qualifying, 'FP1/FP2/FP3'
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session

def get_lap_times(year: int, gp: str, session_type: str = "R") -> pd.DataFrame:
    """Returns all lap times for a session as a clean DataFrame."""
    session = get_session(year, gp, session_type)
    laps = session.laps[["Driver", "LapNumber", "LapTime", "Compound",
                           "TyreLife", "Stint", "SpeedI1", "SpeedI2",
                           "SpeedFL", "SpeedST", "IsPersonalBest"]].copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTimeSeconds"])
    laps = laps[laps["LapTimeSeconds"] > 60]
    return laps.reset_index(drop=True)

def get_race_results(year: int, gp: str) -> pd.DataFrame:
    """Returns final race results with positions and points."""
    session = get_session(year, gp, "R")
    results = session.results[["DriverNumber", "Abbreviation", "FullName",
                                "TeamName", "Position", "Points",
                                "GridPosition", "Status"]].copy()
    results["Year"] = year
    results["GP"] = gp
    return results.reset_index(drop=True)

def get_driver_telemetry(year: int, gp: str, driver: str,
                          lap_number: int = None) -> pd.DataFrame:
    """Returns telemetry for a specific driver. Uses fastest lap if no lap number given."""
    session = get_session(year, gp, "R")
    driver_laps = session.laps.pick_driver(driver)
    lap = driver_laps.pick_fastest() if lap_number is None \
          else driver_laps[driver_laps["LapNumber"] == lap_number].iloc[0]
    telemetry = lap.get_telemetry()
    return telemetry[["Time", "Speed", "RPM", "Gear", "Throttle",
                       "Brake", "DRS", "Distance"]].copy()

def get_session_drivers(year: int, gp: str, session_type: str = "R") -> list:
    """Returns list of driver abbreviations in a session."""
    session = get_session(year, gp, session_type)
    return list(session.laps["Driver"].unique())
