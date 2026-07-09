import requests
import pandas as pd
import time
import os
import json

OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# ── Local results cache ──────────────────────────────────────────────────────
# Mirrors the convention in fastf1_client.py (data/cache). Each completed
# season is cached once and never re-fetched. The *current* in-progress
# season is partially cached and only the rounds run since the last cache
# write are re-fetched — see get_cached_historical_results().
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../data/cache/results")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(year: int) -> str:
    return os.path.join(CACHE_DIR, f"results_{year}.csv")

def _meta_path(year: int) -> str:
    return os.path.join(CACHE_DIR, f"results_{year}.meta.json")

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
    """Fetches race results across multiple seasons for ML training. Always
    hits the API fresh — use get_cached_historical_results() instead if you
    want local caching with incremental updates for the in-progress season."""
    all_rows = []
    for year in range(year_start, year_end + 1):
        try:
            all_rows.extend(_fetch_year_results(year))
            time.sleep(0.3)
        except Exception as e:
            print(f"Warning: could not fetch {year}: {e}")
    return pd.DataFrame(all_rows)

def _fetch_year_results(year: int) -> list:
    """Fetches all race results for a single season. Raises on failure —
    callers decide how to handle it (get_historical_results swallows and
    warns; get_cached_historical_results treats it as 'no update available
    right now, keep using the cache')."""
    rows = []
    data = _get_jolpica(f"{year}/results")
    races = data["MRData"]["RaceTable"]["Races"]
    for race in races:
        for result in race["Results"]:
            pos = result["position"]
            rows.append({
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
    return rows

def get_cached_historical_results(year_start: int, year_end: int,
                                  force_refresh: bool = False) -> pd.DataFrame:
    """
    Like get_historical_results(), but caches each season's results to a
    local CSV file under data/cache/results/ so repeated training runs
    don't re-fetch years that are already complete.

    Completed seasons (anything before the current calendar year) are
    fetched once and cached forever — they cannot change.

    The CURRENT season is special-cased: it's still in progress, so on
    every call we check whether more rounds have completed since the last
    fetch (using the round count in the cached metadata) and only re-fetch
    that season if so. This avoids re-downloading the whole season's
    results every single time, while still picking up new races as they
    happen.

    Pass force_refresh=True to ignore the cache entirely (e.g. for the
    standalone retrain script, or a "Force refresh" button in the UI).
    """
    import datetime
    current_calendar_year = datetime.date.today().year

    all_rows = []
    for year in range(year_start, year_end + 1):
        cache_file = _cache_path(year)
        meta_file = _meta_path(year)
        is_current_season = (year == current_calendar_year)

        cached_df = None
        if os.path.exists(cache_file) and not force_refresh:
            try:
                cached_df = pd.read_csv(cache_file)
            except Exception as e:
                print(f"Warning: cache for {year} unreadable ({e}), refetching.")
                cached_df = None

        needs_fetch = force_refresh or cached_df is None
        if cached_df is not None and is_current_season:
            # Current season — check if new rounds exist before refetching.
            cached_rounds = int(cached_df["round"].max()) if len(cached_df) else 0
            try:
                latest_round = _get_latest_completed_round(year)
            except Exception as e:
                print(f"Warning: could not check latest round for {year}: {e}")
                latest_round = cached_rounds  # assume no change, use cache as-is
            needs_fetch = latest_round > cached_rounds

        if needs_fetch:
            try:
                fresh_rows = _fetch_year_results(year)
                fresh_df = pd.DataFrame(fresh_rows)
                if len(fresh_df) > 0:
                    fresh_df.to_csv(cache_file, index=False)
                    with open(meta_file, "w") as f:
                        json.dump({
                            "fetched_at": datetime.datetime.now().isoformat(),
                            "rounds_cached": int(fresh_df["round"].max()),
                            "rows_cached": len(fresh_df),
                        }, f)
                    cached_df = fresh_df
                    print(f"[cache] {year}: fetched fresh ({len(fresh_df)} rows, "
                          f"through round {int(fresh_df['round'].max()) if len(fresh_df) else 0})")
                elif cached_df is None:
                    cached_df = pd.DataFrame()
            except Exception as e:
                print(f"Warning: could not fetch {year} ({e}); "
                      f"using cached data if available.")
                if cached_df is None:
                    cached_df = pd.DataFrame()
        else:
            print(f"[cache] {year}: using cache "
                  f"({len(cached_df)} rows, no new rounds)")

        if cached_df is not None and len(cached_df) > 0:
            all_rows.append(cached_df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

def _get_latest_completed_round(year: int) -> int:
    """Returns the highest round number with at least one completed race
    result in Jolpica's data for the given season. Cheap-ish check used to
    decide whether the cached current-season data is stale."""
    data = _get_jolpica(f"{year}/results")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return 0
    return max(int(r["round"]) for r in races if r.get("Results"))

def get_cache_status(year_start: int, year_end: int) -> pd.DataFrame:
    """Returns a small status table (year, rows_cached, rounds_cached,
    fetched_at) for the UI to display — e.g. a 'Data freshness' panel."""
    rows = []
    for year in range(year_start, year_end + 1):
        meta_file = _meta_path(year)
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            rows.append({"year": year, **meta})
        else:
            rows.append({"year": year, "fetched_at": None,
                        "rounds_cached": 0, "rows_cached": 0})
    return pd.DataFrame(rows)

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


# ── Bulk historical qualifying results (for the qualifying ranker model) ────
# Same shape and same caching strategy as get_cached_historical_results()
# above (results), just pointed at Jolpica's qualifying endpoint instead of
# results. Kept as separate functions/cache files rather than generalizing
# both into one parameterized cache, since qualifying and race results have
# different row shapes (Q1/Q2/Q3 times vs laps/points/status) and merging
# the caching logic would make both harder to read for a small de-dup win.

def _qual_cache_path(year: int) -> str:
    return os.path.join(CACHE_DIR, f"qualifying_{year}.csv")

def _qual_meta_path(year: int) -> str:
    return os.path.join(CACHE_DIR, f"qualifying_{year}.meta.json")

def _fetch_year_qualifying(year: int) -> list:
    """Fetches all qualifying results for a single season. Raises on
    failure, same contract as _fetch_year_results()."""
    rows = []
    data = _get_jolpica(f"{year}/qualifying")
    races = data["MRData"]["RaceTable"]["Races"]
    for race in races:
        for r in race.get("QualifyingResults", []):
            rows.append({
                "year": year,
                "round": int(race["round"]),
                "gp_name": race["raceName"],
                "circuit": race["Circuit"]["circuitName"],
                "position": int(r["position"]),
                "driver": r["Driver"]["code"],
                "constructor": r["Constructor"]["name"],
                "q1": r.get("Q1", None),
                "q2": r.get("Q2", None),
                "q3": r.get("Q3", None),
            })
    return rows

def get_cached_historical_qualifying(year_start: int, year_end: int,
                                     force_refresh: bool = False) -> pd.DataFrame:
    """
    Qualifying-results equivalent of get_cached_historical_results(): caches
    each completed season's qualifying results to local CSV, and for the
    current in-progress season only re-fetches when a new round's
    qualifying session has happened since the last check.
    """
    import datetime
    current_calendar_year = datetime.date.today().year

    all_rows = []
    for year in range(year_start, year_end + 1):
        cache_file = _qual_cache_path(year)
        meta_file = _qual_meta_path(year)
        is_current_season = (year == current_calendar_year)

        cached_df = None
        if os.path.exists(cache_file) and not force_refresh:
            try:
                cached_df = pd.read_csv(cache_file)
            except Exception as e:
                print(f"Warning: qualifying cache for {year} unreadable ({e}), refetching.")
                cached_df = None

        needs_fetch = force_refresh or cached_df is None
        if cached_df is not None and is_current_season:
            cached_rounds = int(cached_df["round"].max()) if len(cached_df) else 0
            try:
                latest_round = _get_latest_completed_round(year)  # results-based check is fine —
                # qualifying always happens before/with the race for the same round, so "race round
                # N has results" implies "qualifying round N also has results".
            except Exception as e:
                print(f"Warning: could not check latest round for {year}: {e}")
                latest_round = cached_rounds
            needs_fetch = latest_round > cached_rounds

        if needs_fetch:
            try:
                fresh_rows = _fetch_year_qualifying(year)
                fresh_df = pd.DataFrame(fresh_rows)
                if len(fresh_df) > 0:
                    fresh_df.to_csv(cache_file, index=False)
                    with open(meta_file, "w") as f:
                        json.dump({
                            "fetched_at": datetime.datetime.now().isoformat(),
                            "rounds_cached": int(fresh_df["round"].max()),
                            "rows_cached": len(fresh_df),
                        }, f)
                    cached_df = fresh_df
                    print(f"[cache] qualifying {year}: fetched fresh ({len(fresh_df)} rows)")
                elif cached_df is None:
                    cached_df = pd.DataFrame()
            except Exception as e:
                print(f"Warning: could not fetch qualifying {year} ({e}); using cache if available.")
                if cached_df is None:
                    cached_df = pd.DataFrame()
        else:
            print(f"[cache] qualifying {year}: using cache ({len(cached_df)} rows)")

        if cached_df is not None and len(cached_df) > 0:
            all_rows.append(cached_df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

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
