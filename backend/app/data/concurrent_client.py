"""
app/data/concurrent_client.py
─────────────────────────────
Concurrent data fetcher for RaceMindAI.

Replaces the three sequential blocking calls on the Live Standings page:

    BEFORE (sequential — ~2.1s on a typical connection)
    ─────────────────────────────────────────────────────
    drivers      = get_driver_standings(year)   # blocks ~0.8s
    constructors = get_constructor_standings(year)  # blocks ~0.7s
    schedule     = get_season_schedule(year)    # blocks ~0.6s

    AFTER (concurrent — ~0.8s, i.e. max(latencies) not sum)
    ─────────────────────────────────────────────────────────
    from app.data.concurrent_client import fetch_standings_page
    data         = asyncio.run(fetch_standings_page(year))
    drivers      = data["drivers"]
    constructors = data["constructors"]
    schedule     = data["schedule"]

Design notes
────────────
- Uses asyncio + aiohttp so all three HTTP calls are dispatched at once
  and the event loop waits only on the slowest one (max, not sum).
- return_exceptions=True in asyncio.gather() means one timeout doesn't
  kill the other two requests — each degrades to None independently.
- Thread-safe for Streamlit: asyncio.run() creates a fresh event loop
  per call (Streamlit reruns are each single-threaded).
- Falls back gracefully if aiohttp is missing (ImportError message is
  clear so the fix is obvious in deployment logs).
"""

import asyncio
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Shared base URLs (mirrors ergast_client.py) ──────────────────────────────
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE  = "https://api.openf1.org/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level async HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

async def _get_json(
    session,          # aiohttp.ClientSession
    url: str,
    params: dict | None = None,
    label: str = "",
) -> dict[str, Any]:
    """
    Single async GET → parse JSON → return {'source', 'latency_s', 'data'}.
    Retries once on transient errors (same policy as the sync _get_jolpica).
    """
    for attempt in range(2):
        try:
            t0 = time.perf_counter()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            elapsed = round(time.perf_counter() - t0, 3)
            logger.info("[concurrent] %s: %.3fs", label, elapsed)
            return {"source": label, "latency_s": elapsed, "data": data}
        except Exception as exc:
            if attempt == 1:
                raise
            await asyncio.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Per-endpoint coroutines  (mirror the sync functions in ergast_client.py)
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_driver_standings(session, year: int) -> dict:
    url = f"{JOLPICA_BASE}/{year}/driverStandings.json?limit=1000"
    return await _get_json(session, url, label="jolpica_driver_standings")


async def _fetch_constructor_standings(session, year: int) -> dict:
    url = f"{JOLPICA_BASE}/{year}/constructorStandings.json?limit=1000"
    return await _get_json(session, url, label="jolpica_constructor_standings")


async def _fetch_season_schedule(session, year: int) -> dict:
    url = f"{JOLPICA_BASE}/{year}.json?limit=1000"
    return await _get_json(session, url, label="jolpica_schedule")


async def _fetch_current_drivers(session) -> dict:
    """OpenF1 live driver list — used on the Live Standings page."""
    url    = f"{OPENF1_BASE}/drivers"
    params = {"session_key": "latest"}
    return await _get_json(session, url, params=params, label="openf1_drivers")


# ─────────────────────────────────────────────────────────────────────────────
# Public orchestrators
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

def _parse_driver_standings(raw: dict) -> pd.DataFrame:
    """Identical parsing logic to ergast_client.get_driver_standings()."""
    standings = raw["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return pd.DataFrame()
    rows = []
    for s in standings[0]["DriverStandings"]:
        rows.append({
            "position":    int(s["position"]),
            "driver":      s["Driver"]["code"],
            "full_name":   f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "constructor": s["Constructors"][0]["name"],
            "points":      float(s["points"]),
            "wins":        int(s["wins"]),
        })
    return pd.DataFrame(rows)


def _parse_constructor_standings(raw: dict) -> pd.DataFrame:
    """Identical parsing logic to ergast_client.get_constructor_standings()."""
    standings = raw["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return pd.DataFrame()
    rows = []
    for s in standings[0]["ConstructorStandings"]:
        rows.append({
            "position":    int(s["position"]),
            "constructor": s["Constructor"]["name"],
            "points":      float(s["points"]),
            "wins":        int(s["wins"]),
        })
    return pd.DataFrame(rows)


def _parse_schedule(raw: dict) -> pd.DataFrame:
    """Identical parsing logic to ergast_client.get_season_schedule()."""
    races = raw["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in races:
        rows.append({
            "round":   int(r["round"]),
            "gp_name": r["raceName"],
            "circuit": r["Circuit"]["circuitName"],
            "country": r["Circuit"]["Location"]["country"],
            "date":    r["date"],
        })
    return pd.DataFrame(rows)


async def _fetch_standings_page_async(year: int, timeout_s: float = 12.0):
    """
    Core coroutine: fires all three Jolpica calls concurrently with
    asyncio.gather() and returns parsed DataFrames + timing metadata.
    """
    import aiohttp  # imported here so ImportError surfaces at call time

    wall_start = time.perf_counter()

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            _fetch_driver_standings(session, year),
            _fetch_constructor_standings(session, year),
            _fetch_season_schedule(session, year),
            return_exceptions=True,   # ← one failure won't kill the others
        )

    wall_elapsed = round(time.perf_counter() - wall_start, 3)

    drv_result, con_result, sch_result = results

    def safe_parse(result, parser, fallback_name: str):
        if isinstance(result, Exception):
            logger.error("[concurrent] %s failed: %s", fallback_name, result)
            return pd.DataFrame()
        return parser(result["data"])

    drivers      = safe_parse(drv_result, _parse_driver_standings,      "driver_standings")
    constructors = safe_parse(con_result, _parse_constructor_standings,  "constructor_standings")
    schedule     = safe_parse(sch_result, _parse_schedule,               "schedule")

    per_call = {
        r["source"]: r["latency_s"]
        for r in results
        if not isinstance(r, Exception)
    }

    logger.info(
        "[concurrent] total wall=%.3fs  per-call=%s",
        wall_elapsed, per_call,
    )

    return {
        "drivers":      drivers,
        "constructors": constructors,
        "schedule":     schedule,
        "latency_s":    wall_elapsed,
        "per_call_latency": per_call,
    }


def fetch_standings_page(year: int, timeout_s: float = 12.0) -> dict:
    """
    Public entry point — synchronous wrapper around the async core.

    Safe to call from Streamlit (each rerun is single-threaded).
    Raises ImportError with a clear message if aiohttp is not installed.

    Returns
    -------
    dict with keys:
        drivers      : pd.DataFrame  — driver championship standings
        constructors : pd.DataFrame  — constructor championship standings
        schedule     : pd.DataFrame  — full season race calendar
        latency_s    : float         — total wall-clock time
        per_call_latency : dict[str, float] — per-endpoint breakdown
    """
    try:
        import aiohttp  # noqa: F401 — trigger ImportError early with clear message
    except ImportError:
        raise ImportError(
            "aiohttp is required for concurrent fetching. "
            "Run: pip install aiohttp"
        )
    return asyncio.run(_fetch_standings_page_async(year, timeout_s))
