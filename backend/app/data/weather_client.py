"""
app/data/weather_client.py
────────────────────────────
Weather data for the race-predictor and qualifying-predictor feature
pipelines. See RaceMindAI_Redesign_Phases6-7.md §6.4 for the full design
rationale — summarized here:

THE TRAP THIS AVOIDS: the obvious approach is "train on actual observed
weather (easy — FastF1's session.weather_data already has it), predict
using a live forecast for the upcoming race." That mismatch is a real
problem — actual observed weather and a forecast made days in advance have
different error distributions (forecasts smooth out extremes, miss
short-lived showers), so a model trained on ground-truth weather learns
relationships that don't hold when fed forecast-shaped data at inference
time.

THE FIX: Open-Meteo publishes a "Historical Forecast" archive (from 2021
onward) that stores what the forecast actually said at the time, in the
exact same schema as its live forecast endpoint — not what really
happened. Training and inference both consume forecast-shaped data, so
there's no train/inference distribution mismatch. This module wraps both
endpoints behind one interface so the caller (weather_features.py) doesn't
need to know which one it's talking to.

Open-Meteo requires no API key and has no request quota for this app's
volume (bounded, ~a few hundred calls for a one-time historical backfill,
then permanently cached — see §7.6 "Backfill cost" in the roadmap doc).

NETWORK NOTE: api.open-meteo.com is not reachable from the sandboxed
environment this module was written in, so the HTTP calls below are
implemented against Open-Meteo's documented API contract but have NOT been
exercised against the live network from here — run a real smoke test
(get_live_forecast() for a real circuit/near-future date) before trusting
this in production. If the response schema has drifted from what's
documented, the parsing in _extract_hourly_slice() is the most likely spot
to need adjustment.
"""

import datetime
import hashlib
import json
import os

import requests

LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Historical Forecast archive coverage — per Open-Meteo's documented range.
# Dates before this fall back to climatology (see get_climatology()).
HISTORICAL_FORECAST_START = datetime.date(2021, 1, 1)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../data/cache/weather")
os.makedirs(CACHE_DIR, exist_ok=True)

HOURLY_VARS = "temperature_2m,precipitation_probability,precipitation,windspeed_10m,cloudcover"

DEFAULT_SESSION_HOUR_LOCAL = 14  # 2pm local — a reasonable default for a
# typical race/qualifying start time when the actual scheduled time isn't
# available from the schedule data. Overridden by session_hour_local when
# the caller has a real value (see weather_features.py, which pulls this
# from ergast_client's schedule once that's extended to include it — see
# the note in weather_features.py).

# Rough global fallback if even climatology can't be computed (e.g. no
# cached historical data yet AND the live API is unreachable). Deliberately
# a bland, low-confidence "assume average" value — same philosophy as the
# rookie/new-constructor fallbacks elsewhere in this codebase, not a
# fabricated extreme in either direction.
GLOBAL_DEFAULT = {
    "forecast_air_temp_c": 22.0,
    "forecast_precip_prob": 0.15,
    "forecast_precip_mm": 0.5,
    "forecast_wind_speed_kmh": 15.0,
    "forecast_cloudcover_pct": 50.0,
}


def _cache_key(lat: float, lon: float, date_str: str, hour: int) -> str:
    raw = f"{round(lat, 3)}_{round(lon, 3)}_{date_str}_{hour}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _extract_hourly_slice(payload: dict, date_str: str, hour: int) -> dict | None:
    """Pulls the single hourly row nearest `hour` on `date_str` out of an
    Open-Meteo hourly response. Returns None if the requested date isn't
    covered by the response (e.g. outside forecast range)."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return None
    target_prefix = f"{date_str}T{hour:02d}:"
    times = hourly["time"]
    idx = None
    for i, t in enumerate(times):
        if t.startswith(target_prefix):
            idx = i
            break
    if idx is None:
        # Fall back to the closest available hour on that date, if any.
        same_day = [i for i, t in enumerate(times) if t.startswith(date_str)]
        if not same_day:
            return None
        idx = min(same_day, key=lambda i: abs(int(times[i][11:13]) - hour))

    def _get(field, default=None):
        vals = hourly.get(field)
        return vals[idx] if vals and idx < len(vals) else default

    return {
        "forecast_air_temp_c": _get("temperature_2m"),
        "forecast_precip_prob": (_get("precipitation_probability", 0) or 0) / 100.0,
        "forecast_precip_mm": _get("precipitation", 0.0),
        "forecast_wind_speed_kmh": _get("windspeed_10m"),
        "forecast_cloudcover_pct": _get("cloudcover"),
    }


def _track_temp_proxy(air_temp_c: float, cloudcover_pct: float) -> float:
    """Open-Meteo doesn't expose track temperature directly. Approximate it
    via a simple solar-heating offset from air temp — sunnier (lower cloud
    cover) means more radiative heating of the asphalt above ambient air
    temperature. This is a deliberately simple, hand-set heuristic, not a
    physical model — flagged the same way this codebase already flags its
    other hand-set approximations (e.g. season_simulator's safety-car
    probabilities). If more accuracy matters later, FastF1's
    session.weather_data has real TrackTemp for any session that's already
    happened, which could be used to fit a better offset model per circuit
    surface type — but that's only available after the fact, not for a
    live pre-session forecast, so this proxy is still needed at inference
    time regardless.
    """
    if air_temp_c is None:
        return None
    cloud = cloudcover_pct if cloudcover_pct is not None else 50.0
    if cloud < 30:
        offset = 20.0
    elif cloud < 70:
        offset = 12.0
    else:
        offset = 5.0
    return round(air_temp_c + offset, 1)


def _fetch(base_url: str, lat: float, lon: float, date_str: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }
    resp = requests.get(base_url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_live_forecast(lat: float, lon: float, target_date: datetime.date,
                      hour: int = DEFAULT_SESSION_HOUR_LOCAL) -> dict:
    """
    Live forecast for an upcoming session. NOT cached — a forecast for a
    future date changes as the date approaches, so caching it would go
    stale silently. Falls back to climatology on any failure (network
    error, date outside the forecast API's window, etc.) rather than
    raising, matching this codebase's existing try/except-soft-fail
    pattern for optional data (see main.py's optional-import pattern).
    """
    date_str = target_date.isoformat()
    try:
        payload = _fetch(LIVE_FORECAST_URL, lat, lon, date_str)
        slice_ = _extract_hourly_slice(payload, date_str, hour)
        if slice_ is None:
            raise ValueError("date outside live forecast window")
    except Exception as e:
        print(f"[weather_client] live forecast failed for ({lat},{lon}) "
              f"{date_str}: {e} — falling back to climatology")
        return {**get_climatology(lat, lon, target_date.month), "forecast_source": "climatology_fallback"}

    track_temp = _track_temp_proxy(slice_["forecast_air_temp_c"], slice_["forecast_cloudcover_pct"])
    return {**slice_, "forecast_track_temp_proxy_c": track_temp, "forecast_source": "live_forecast"}


def get_historical_forecast(lat: float, lon: float, target_date: datetime.date,
                            hour: int = DEFAULT_SESSION_HOUR_LOCAL) -> dict:
    """
    What the forecast actually said, for a PAST date — used to backfill
    training rows so training and inference both see forecast-shaped
    weather (see module docstring). Permanently cached to disk: weather for
    a past session never changes once fetched, exactly like the completed-
    season results cache in ergast_client.py.

    Falls back to climatology for dates before HISTORICAL_FORECAST_START
    (Open-Meteo's archive coverage begins 2021) or on any fetch failure.
    """
    date_str = target_date.isoformat()
    key = _cache_key(lat, lon, date_str, hour)
    cache_file = _cache_path(key)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    if target_date < HISTORICAL_FORECAST_START:
        result = {**get_climatology(lat, lon, target_date.month), "forecast_source": "climatology_pre_archive"}
        return result  # not cached — climatology may improve as more real data accumulates

    try:
        payload = _fetch(HISTORICAL_FORECAST_URL, lat, lon, date_str)
        slice_ = _extract_hourly_slice(payload, date_str, hour)
        if slice_ is None:
            raise ValueError("date not covered by historical forecast archive")
        track_temp = _track_temp_proxy(slice_["forecast_air_temp_c"], slice_["forecast_cloudcover_pct"])
        result = {**slice_, "forecast_track_temp_proxy_c": track_temp, "forecast_source": "historical_forecast_archive"}
    except Exception as e:
        print(f"[weather_client] historical forecast failed for ({lat},{lon}) "
              f"{date_str}: {e} — falling back to climatology")
        result = {**get_climatology(lat, lon, target_date.month), "forecast_source": "climatology_fallback"}
        return result  # don't cache a fallback — retry on next call in case it was transient

    with open(cache_file, "w") as f:
        json.dump({**result, "_lat": round(lat, 1), "_lon": round(lon, 1), "_month": target_date.month}, f)
    return result


# ── Climatology fallback ─────────────────────────────────────────────────
# "Assume average for this circuit/month, not a fabricated extreme" — same
# fallback philosophy already used for rookie drivers and new constructors
# elsewhere in this codebase (feature_engineering.apply_rookie_fallback,
# apply_new_constructor_fallback). Computed lazily from whatever historical
# forecast rows have already been cached for that circuit; if none exist
# yet (e.g. very first call before any backfill has run), falls all the way
# back to GLOBAL_DEFAULT.
def get_climatology(lat: float, lon: float, month: int) -> dict:
    prefix_lat, prefix_lon = round(lat, 1), round(lon, 1)
    matches = []
    if os.path.isdir(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, fname)
            try:
                with open(fpath) as f:
                    row = json.load(f)
            except Exception:
                continue
            # Filter to the same circuit (rounded lat/lon match) and same
            # calendar month across whatever years have been backfilled —
            # e.g. "what has the forecast typically said for THIS circuit
            # in THIS month, across all cached seasons." Falls back to
            # unfiltered / global default below if nothing matches yet
            # (e.g. before any backfill has run for this circuit).
            if row.get("_lat") == prefix_lat and row.get("_lon") == prefix_lon and row.get("_month") == month:
                matches.append(row)

    numeric_fields = ["forecast_air_temp_c", "forecast_precip_prob",
                       "forecast_precip_mm", "forecast_wind_speed_kmh",
                       "forecast_cloudcover_pct"]
    if not matches:
        return dict(GLOBAL_DEFAULT)

    result = {}
    for field in numeric_fields:
        vals = [m[field] for m in matches if m.get(field) is not None]
        result[field] = round(sum(vals) / len(vals), 2) if vals else GLOBAL_DEFAULT[field]
    result["forecast_track_temp_proxy_c"] = _track_temp_proxy(
        result["forecast_air_temp_c"], result["forecast_cloudcover_pct"]
    )
    return result
