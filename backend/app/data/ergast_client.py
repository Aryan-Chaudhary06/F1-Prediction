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

# Some Jolpica endpoints — specifically /results and /qualifying, which
# return nested race+result objects rather than one row per race — were
# observed silently capping each response at 100 result-rows REGARDLESS of
# the `?limit=1000` requested. Simpler endpoints (schedule, standings) never
# exposed this because a full season's worth of races/standings already
# fits comfortably under 100 rows in one response. This was the actual
# root cause behind a "100 rows every season" pattern that survived even
# after fixing the separate stale-cache bug (_cache_looks_incomplete) —
# see RaceMindAI conversation history: the FRESH fetch was capping at 100
# rows too, proving it wasn't a caching problem at all.
#
# Fixed via real offset-based pagination using MRData.total, rather than
# just requesting a larger limit (which the server ignores past 100 for
# these endpoints).
_PAGE_SIZE = 100
_MAX_PAGES = 20  # safety cap — a full season is ~5 pages at 100/page;
                  # 20 pages (2000 rows) is generously beyond anything a
                  # single season could need, guards against an infinite
                  # loop if `total` is ever missing/wrong in a response.

def _get_jolpica_paginated(endpoint: str) -> list:
    """
    Paginates a Jolpica endpoint that returns nested Race objects (each
    containing a Results or QualifyingResults list) via offset/limit,
    using MRData.total to know when every row has been fetched. Returns
    the concatenated list of Race entries across all pages.

    NOTE: a single race's Results can end up split across two pages if its
    rows happen to straddle a page boundary (e.g. round 5's last few
    results on page 1, its remaining results on page 2) — this is fine for
    every current caller, which flattens race->results into flat rows
    anyway rather than relying on each Race entry being complete in one
    piece; offset-based pagination guarantees no row is skipped or
    duplicated across pages either way.
    """
    all_races = []
    offset = 0
    for _ in range(_MAX_PAGES):
        url = f"{JOLPICA_BASE}/{endpoint}.json?limit={_PAGE_SIZE}&offset={offset}"
        data = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(1)

        races = data["MRData"]["RaceTable"]["Races"]
        total = int(data["MRData"].get("total", 0))
        all_races.extend(races)

        page_row_count = sum(
            len(race.get("Results") or race.get("QualifyingResults") or [])
            for race in races
        )
        offset += page_row_count
        if page_row_count == 0 or offset >= total:
            break
    else:
        print(f"Warning: _get_jolpica_paginated({endpoint!r}) hit the "
              f"{_MAX_PAGES}-page safety cap without reaching MRData.total "
              f"— data may be incomplete. Investigate if this happens.")

    return all_races

# A complete modern F1 season has ~20-24 rounds (even 2026, with 2 rounds
# cancelled, still has 22). Used below to detect a cache file that looks
# suspiciously incomplete — e.g. a stale CSV left over from an interrupted
# fetch, or one written before `?limit=1000` was added to _get_jolpica().
# See RaceMindAI conversation history: a "100 rows" cache was observed
# identically across FIVE different seasons (2022-2026), including the
# in-progress 2026 season where only round 1 had actually happened at the
# time — a dead giveaway that this was a stale file on disk, not
# organically fetched data, since 100 rows doesn't correspond to either "1
# round" or "a full season" for any of those years.
MIN_PLAUSIBLE_ROUNDS_FOR_COMPLETE_SEASON = 15

def _cache_looks_incomplete(cached_df: pd.DataFrame) -> bool:
    if cached_df is None or cached_df.empty or "round" not in cached_df.columns:
        return True
    return cached_df["round"].nunique() < MIN_PLAUSIBLE_ROUNDS_FOR_COMPLETE_SEASON

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
            # Race start time in UTC (e.g. "14:00:00Z"), when Jolpica
            # provides it — added for weather_features.py, which otherwise
            # has to guess a session time via
            # weather_client.DEFAULT_SESSION_HOUR_LOCAL. Not every event
            # has this populated far in advance, so callers must still
            # handle it being None. See RaceMindAI_Redesign_Phases6-7.md
            # §6.4 point 1.
            "time": r.get("time"),
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
    right now, keep using the cache').

    Uses _get_jolpica_paginated() rather than a single _get_jolpica() call
    — this endpoint silently caps each response at 100 rows regardless of
    the requested limit, so a single request only ever returned the first
    ~5 rounds of a season. See _get_jolpica_paginated()'s docstring."""
    rows = []
    races = _get_jolpica_paginated(f"{year}/results")
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
        elif cached_df is not None and not is_current_season and _cache_looks_incomplete(cached_df):
            # A "completed" season's round count is fixed and known-ish
            # (~20-24) — if the cache has far fewer distinct rounds than
            # that, it's very likely a stale/partial file from an earlier
            # interrupted fetch, not a real complete season, and would
            # otherwise be trusted forever (this branch previously never
            # re-checked past seasons at all). See the module-level
            # MIN_PLAUSIBLE_ROUNDS_FOR_COMPLETE_SEASON comment above.
            print(f"[cache] {year}: cached file has only "
                  f"{cached_df['round'].nunique()} distinct round(s) — looks "
                  f"incomplete for a finished season, refetching instead of "
                  f"trusting it.")
            needs_fetch = True

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
    decide whether the cached current-season data is stale.

    Uses _get_jolpica_paginated() rather than a single _get_jolpica() call
    — this had the SAME 100-row cap as _fetch_year_results(), which meant
    this function could only ever see the season's earliest ~5 rounds
    (Jolpica returns races in chronological order) and would silently
    under-report the true latest completed round for any season past
    round ~5. Since this function's whole purpose is deciding whether the
    current season's cache needs refreshing, an under-reported round
    number could make the app wrongly conclude there's nothing new to
    fetch even when several more races had actually happened."""
    races = _get_jolpica_paginated(f"{year}/results")
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
    failure, same contract as _fetch_year_results(). Same pagination fix
    applies here — this endpoint has the identical 100-row-per-response
    cap as /results."""
    rows = []
    races = _get_jolpica_paginated(f"{year}/qualifying")
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
        elif cached_df is not None and not is_current_season and _cache_looks_incomplete(cached_df):
            # Same fix as get_cached_historical_results() — see that
            # function's comment for the full rationale.
            print(f"[cache] qualifying {year}: cached file has only "
                  f"{cached_df['round'].nunique()} distinct round(s) — looks "
                  f"incomplete for a finished season, refetching instead of "
                  f"trusting it.")
            needs_fetch = True

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
