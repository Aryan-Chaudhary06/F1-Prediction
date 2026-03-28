import requests
import pandas as pd
import time

OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

def _get_openf1(endpoint: str, params: dict = None) -> list:
    """GET from OpenF1 API."""
    url = f"{OPENF1_BASE}/{endpoint}"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(1)

def _get_jolpica(endpoint: str) -> dict:
    """GET from Jolpica (Ergast-compatible replacement API)."""
    url = f"{JOLPICA_BASE}/{endpoint}.json?limit=1000"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(1)

def get_season_schedule(year: int) -> pd.DataFrame:
    """Returns the full race schedule for a season."""
    data = _get_jolpica(f"{year}")
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in races:
        rows.append({
            "round": int(r["round"]),
            "gp_name": r["raceName"],
            "circuit": r["Circuit"]["circuitName"],
            "country": r["Circuit"]["Location"]["country"],
            "date": r["date"],
        })
    return pd.DataFrame(rows)

def get_driver_standings(year: int, round_num: int = None) -> pd.DataFrame:
    """Returns driver championship standings."""
    endpoint = f"{year}/driverStandings" if round_num is None \
               else f"{year}/{round_num}/driverStandings"
    data = _get_jolpica(endpoint)
    standings = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return pd.DataFrame()
    rows = []
    for s in standings[0]["DriverStandings"]:
        rows.append({
            "position": int(s["position"]),
            "driver": s["Driver"]["code"],
            "full_name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "constructor": s["Constructors"][0]["name"],
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        })
    return pd.DataFrame(rows)

def get_constructor_standings(year: int, round_num: int = None) -> pd.DataFrame:
    """Returns constructor championship standings."""
    endpoint = f"{year}/constructorStandings" if round_num is None \
               else f"{year}/{round_num}/constructorStandings"
    data = _get_jolpica(endpoint)
    standings = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return pd.DataFrame()
    rows = []
    for s in standings[0]["ConstructorStandings"]:
        rows.append({
            "position": int(s["position"]),
            "constructor": s["Constructor"]["name"],
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        })
    return pd.DataFrame(rows)

def get_historical_results(year_start: int, year_end: int) -> pd.DataFrame:
    """Fetches race results across multiple seasons for ML training."""
    all_rows = []
    for year in range(year_start, year_end + 1):
        try:
            data = _get_jolpica(f"{year}/results")
            races = data["MRData"]["RaceTable"]["Races"]
            for race in races:
                for result in race["Results"]:
                    pos = result["position"]
                    all_rows.append({
                        "year": year,
                        "round": int(race["round"]),
                        "gp_name": race["raceName"],
                        "circuit": race["Circuit"]["circuitName"],
                        "driver": result["Driver"]["code"],
                        "constructor": result["Constructor"]["name"],
                        "grid": int(result["grid"]),
                        "position": int(pos) if str(pos).isdigit() else None,
                        "points": float(result["points"]),
                        "status": result["status"],
                        "laps": int(result["laps"]),
                    })
            time.sleep(0.3)
        except Exception as e:
            print(f"Warning: could not fetch {year}: {e}")
    return pd.DataFrame(all_rows)

def get_qualifying_results(year: int, round_num: int) -> pd.DataFrame:
    """Returns qualifying results for a specific round."""
    data = _get_jolpica(f"{year}/{round_num}/qualifying")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    rows = []
    for r in races[0]["QualifyingResults"]:
        rows.append({
            "position": int(r["position"]),
            "driver": r["Driver"]["code"],
            "constructor": r["Constructor"]["name"],
            "q1": r.get("Q1", None),
            "q2": r.get("Q2", None),
            "q3": r.get("Q3", None),
        })
    return pd.DataFrame(rows)

def get_current_drivers(year: int = 2025) -> pd.DataFrame:
    """Returns all drivers on the current grid via OpenF1."""
    data = _get_openf1("drivers", {"session_key": "latest"})
    rows = []
    seen = set()
    for d in data:
        code = d.get("name_acronym")
        if code and code not in seen:
            seen.add(code)
            rows.append({
                "driver": code,
                "full_name": d.get("full_name", ""),
                "team": d.get("team_name", ""),
                "number": d.get("driver_number"),
            })
    return pd.DataFrame(rows)
