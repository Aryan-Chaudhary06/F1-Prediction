"""
app/models/weather_features.py
────────────────────────────────
Ties weather_client.py into the race and qualifying feature pipelines.
Kept as its own module rather than folded into feature_engineering.py /
qualifying_feature_engineering.py because it needs network I/O (weather
API calls) while those two files are otherwise pure functions of an
already-fetched dataframe — mixing the two would make the existing
pipelines harder to unit-test in isolation.

Two entry points:
  - build_weather_lookup()   : batch-backfills weather for every historical
                                (year, round) in a schedule — the "bounded,
                                one-time job" described in
                                RaceMindAI_Redesign_Phases6-7.md §7.6.
  - get_session_weather()    : single live/upcoming-session lookup, for
                                inference-time predictions in main.py.

WEATHER_FEATURE_COLUMNS lists every column this module adds — add new
weather signals here AND in this list together so callers building
FEATURES lists (race_predictor.py, qualifying_predictor.py) don't drift
out of sync with what this module actually produces.
"""

import datetime
import pandas as pd

from app.data.circuit_coordinates import get_circuit_coordinates
from app.data.weather_client import (
    get_historical_forecast,
    get_live_forecast,
    DEFAULT_SESSION_HOUR_LOCAL,
)

WEATHER_FEATURE_COLUMNS = [
    "forecast_air_temp_c",
    "forecast_track_temp_proxy_c",
    "forecast_precip_prob",
    "forecast_precip_mm",
    "forecast_wind_speed_kmh",
    "weather_regime_code",
]

# Thresholds for collapsing forecast_precip_prob into a 3-way categorical,
# the direct replacement for qualifying_feature_engineering's old constant
# `is_wet` column. Kept as a derived feature (rather than only the raw
# probability) because it's genuinely useful in its own right for
# interpretability (SHAP/feature-importance output reads as "Wet" rather
# than "precip_prob=0.83"), same spirit as circuit_type_code being derived
# from CIRCUIT_TYPE. WEATHER_REGIMES order matters — it defines the integer
# coding (Dry=0, Mixed=1, Wet=2).
WEATHER_REGIMES = ["Dry", "Mixed", "Wet"]


def _weather_regime_code(precip_prob: float) -> int:
    if precip_prob is None:
        return WEATHER_REGIMES.index("Dry")  # unknown defaults to the
        # median-common case rather than an alarming "Wet", consistent
        # with this codebase's general "assume average, don't fabricate
        # an extreme" fallback philosophy.
    if precip_prob >= 0.6:
        return WEATHER_REGIMES.index("Wet")
    if precip_prob >= 0.25:
        return WEATHER_REGIMES.index("Mixed")
    return WEATHER_REGIMES.index("Dry")


def default_weather_row() -> dict:
    """Public helper: the same bland/no-signal default row used when
    weather data is unavailable, for callers (race_predictor.py,
    qualifying_predictor.py) that need to fill WEATHER_FEATURE_COLUMNS
    when no weather_lookup was supplied at all, without training/inference
    code reaching into this module's private _finalize_row()."""
    return _finalize_row({})


def _finalize_row(raw: dict) -> dict:
    """
    Builds the final WEATHER_FEATURE_COLUMNS dict from a (possibly empty
    or partial) raw weather dict.

    BUG FIX: this used to do a bare `raw.get(field)`, which returns None
    for every numeric field when `raw` is {} (the exact case
    default_weather_row() and every "no data available" fallback path
    relies on). Filling a training row's weather columns with None meant
    every row got dropped by train_model()'s dropna(subset=FEATURES) —
    silently producing an EMPTY training set whenever no weather_lookup
    was supplied, rather than the intended "bland Dry default, train
    anyway" behavior. Confirmed via tests/test_models.py:test_full_pipeline
    failing with "cannot call vectorize on size 0 inputs" — 1,320 real
    race rows going in, 0 rows surviving dropna().

    Now falls back to weather_client.GLOBAL_DEFAULT's real numeric values
    (the same "assume average, don't fabricate an extreme" bland defaults
    already used elsewhere in weather_client.py) for any field missing
    from `raw`, rather than leaving it None.
    """
    from app.data.weather_client import GLOBAL_DEFAULT, _track_temp_proxy

    def _val(field):
        v = raw.get(field)
        return v if v is not None else GLOBAL_DEFAULT.get(field)

    air_temp = _val("forecast_air_temp_c")
    precip_prob = _val("forecast_precip_prob")

    track_temp = raw.get("forecast_track_temp_proxy_c")
    if track_temp is None:
        # Derive from the (real-or-default) air temp rather than leaving
        # it None too — GLOBAL_DEFAULT doesn't carry cloudcover, so use a
        # neutral 50% for the proxy's solar-heating estimate here.
        track_temp = _track_temp_proxy(air_temp, 50.0)

    return {
        "forecast_air_temp_c": air_temp,
        "forecast_track_temp_proxy_c": track_temp,
        "forecast_precip_prob": precip_prob,
        "forecast_precip_mm": _val("forecast_precip_mm"),
        "forecast_wind_speed_kmh": _val("forecast_wind_speed_kmh"),
        "weather_regime_code": _weather_regime_code(precip_prob),
    }


def build_weather_lookup(schedule_df: pd.DataFrame,
                         session_hour_local: int = DEFAULT_SESSION_HOUR_LOCAL) -> pd.DataFrame:
    """
    Batch-backfills weather features for every row in `schedule_df`, which
    must have columns: year, round, circuit, date (the shape returned by
    ergast_client.get_season_schedule(), concatenated across seasons).

    Uses get_historical_forecast() for every row — even for the
    currently-in-progress season's PAST rounds, since those already
    happened and should use the same forecast-archive path as older
    seasons for consistency. Only genuinely upcoming rounds should go
    through get_session_weather() / get_live_forecast() instead (call
    that separately at prediction time, not through this function).

    Returns a DataFrame keyed by (year, round) with WEATHER_FEATURE_COLUMNS
    — merge this onto a training feature dataframe on ["year","round"].

    This is the "bounded, one-time-ish job" from §7.6 of the roadmap: a
    few hundred API calls total across ~4 seasons × ~24 races, each
    permanently cached afterward by weather_client, so re-running this
    after the first backfill is cheap (cache hits only) except for
    genuinely new rows.

    Skips schedule rows whose date is in the future — the Historical
    Forecast archive only covers past dates, so calling it for a race that
    hasn't happened yet is a GUARANTEED 400 error every time, not an
    occasional one (this was previously discovered the expensive way: a
    wasted API round-trip + a soft-fail-to-climatology for every future
    round in the current season, every single training run). Those rounds
    have no results yet anyway, so build_training_features()'s own
    dropna(subset=FEATURES) already excludes them from training — there
    was never any point backfilling weather for them.

    This is an intentional limitation, not a workaround: an upcoming
    round's weather should come from get_session_weather() /
    get_live_forecast() at actual PREDICTION time (see main.py's predict
    routes), when a live forecast is meaningful — not from this training
    backfill function.
    """
    rows = []
    today = datetime.date.today()
    skipped_future = 0
    for _, race in schedule_df.iterrows():
        coords = get_circuit_coordinates(race["circuit"])
        if coords is None:
            print(f"[weather_features] no coordinates for circuit "
                  f"'{race['circuit']}' — skipping (will fall back to "
                  f"climatology default at merge time via NaN handling).")
            continue
        lat, lon = coords
        try:
            target_date = datetime.date.fromisoformat(str(race["date"])[:10])
        except (ValueError, TypeError):
            print(f"[weather_features] unparseable date '{race['date']}' "
                  f"for {race['circuit']} {race['year']} round {race['round']} — skipping.")
            continue

        if target_date > today:
            skipped_future += 1
            continue  # hasn't happened yet — see docstring above

        raw = get_historical_forecast(lat, lon, target_date, hour=session_hour_local)
        row = _finalize_row(raw)
        row.update({"year": int(race["year"]), "round": int(race["round"])})
        rows.append(row)

    if skipped_future:
        print(f"[weather_features] skipped {skipped_future} future race(s) "
              f"(no results yet to train on — weather for these belongs at "
              f"prediction time via get_session_weather(), not this backfill).")

    return pd.DataFrame(rows)


def get_session_weather(circuit_name: str, session_date: datetime.date,
                        session_hour_local: int = DEFAULT_SESSION_HOUR_LOCAL,
                        manual_override: str | None = None) -> dict:
    """
    Single-session weather lookup for INFERENCE (an upcoming race or
    qualifying prediction) — the function main.py's predict routes should
    call directly, rather than build_weather_lookup() (which is for batch
    training backfill).

    `manual_override`: "Dry" | "Mixed" | "Wet" | None. If given, skips the
    live forecast entirely and returns that regime with its raw fields set
    to None — preserves the existing "what if it rains" scenario-exploration
    UX qualifying_predict already partially supported via its `weather`
    field, now applying to both models consistently instead of just
    qualifying's is_wet flag.
    """
    if manual_override is not None:
        if manual_override not in WEATHER_REGIMES:
            raise ValueError(f"manual_override must be one of {WEATHER_REGIMES}, got {manual_override!r}")
        return {
            "forecast_air_temp_c": None,
            "forecast_track_temp_proxy_c": None,
            "forecast_precip_prob": None,
            "forecast_precip_mm": None,
            "forecast_wind_speed_kmh": None,
            "weather_regime_code": WEATHER_REGIMES.index(manual_override),
            "forecast_source": "manual_override",
            "forecast_lead_time_hours": None,
        }

    coords = get_circuit_coordinates(circuit_name)
    if coords is None:
        # No known coordinates — fall back to the Dry/default regime rather
        # than block the prediction entirely.
        row = _finalize_row({})
        row["forecast_source"] = "no_coordinates_default"
        row["forecast_lead_time_hours"] = None
        return row

    lat, lon = coords
    lead_time_hours = (
        datetime.datetime.combine(session_date, datetime.time(hour=session_hour_local))
        - datetime.datetime.now()
    ).total_seconds() / 3600.0

    raw = get_live_forecast(lat, lon, session_date, hour=session_hour_local)
    row = _finalize_row(raw)
    row["forecast_source"] = raw.get("forecast_source", "live_forecast")
    row["forecast_lead_time_hours"] = round(lead_time_hours, 1)
    # Forecast skill degrades past ~5-7 days out (§6.4 point 4 in the
    # roadmap) — surface that explicitly rather than presenting a 10-day-
    # out forecast with the same implied certainty as a same-day one. The
    # API layer (main.py) should pass this through to the frontend as a
    # "forecast confidence" indicator; not collapsed into a boolean here
    # since the UI may want to show the actual lead time.
    row["forecast_reliable"] = lead_time_hours <= 168  # 7 days
    return row


def attach_weather_features(feat_df: pd.DataFrame, weather_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merges a weather_lookup (from build_weather_lookup) onto a training
    feature dataframe (from build_training_features /
    build_qualifying_training_features) on ["year","round"]. Any row with
    no matching weather (a circuit missing from circuit_coordinates.py, or
    a fetch that fell through even climatology) gets the GLOBAL_DEFAULT-
    equivalent Dry/no-signal row rather than NaN, so it doesn't get
    silently dropped by a downstream dropna(subset=FEATURES) the way a raw
    NaN would.
    """
    df = feat_df.merge(weather_lookup, on=["year", "round"], how="left")
    missing = df["weather_regime_code"].isna()
    if missing.any():
        defaults = _finalize_row({})
        for col, val in defaults.items():
            df.loc[missing, col] = val
        print(f"[weather_features] {missing.sum()} rows had no weather match "
              f"— filled with the default Dry/no-signal row.")
    return df