"""
app/models/practice_pace_features.py
──────────────────────────────────────
FP2 long-run pace feature for the RACE predictor (race_predictor.py only —
NOT the qualifying model, which is deliberately single-lap-pace-focused
per qualifying_feature_engineering.py's own docstring; long-run race pace
is a different signal and belongs with the race model, same reasoning
already applied elsewhere in this codebase to keep the two pipelines
separate).

See RaceMindAI_Audit_Phases1-5.md Phase 3.2 (FP2 is the session focused on
race-simulation stints) and RaceMindAI_Redesign_Phases6-7.md §6.5 (this
was designed but never built until now).

WHY THIS IS SAFE TO USE AS A SAME-RACE FEATURE (NOT LEAKAGE): FP2 happens
BEFORE the race it's predicting, on the same race weekend — exactly like
`grid` (qualifying result, also same-weekend, pre-race information) is
already used as a feature in feature_engineering.py. Using a race's own
FP2 pace to help predict that same race's result is legitimate pre-race
signal, not a leak of the race's own outcome.

NETWORK NOTE: like weather_client.py, this module's FastF1 calls have NOT
been exercised against a live session from the sandboxed environment this
was written in — fastf1_client.py's own network dependency (FastF1's
timing-data downloads) isn't reachable here. The lap-filtering / pace-
aggregation logic below has been verified against fabricated lap data
shaped exactly like fastf1_client.get_lap_times()'s documented output
schema. Run a real smoke test — build_practice_pace_lookup() for one
already-completed 2026 race — before trusting this in production.
"""

import numpy as np
import pandas as pd

from app.data.fastf1_client import get_lap_times

PRACTICE_PACE_FEATURE_COLUMNS = [
    "practice_pace_delta_pct",
    "practice_lap_count",
]

# A "long run" stint is one long enough to be race-representative (fuel
# load, tyre management) rather than a short qualifying-simulation stint
# (1-3 laps on fresh tyres, low fuel — fast but NOT representative of race
# pace). Below this many laps, a stint is excluded entirely.
MIN_LONG_RUN_LAPS = 5

# Within a long-run stint, drop the first lap (out-lap / still finding
# grip after the pit stop — not representative) and any lap more than this
# fraction slower than the stint's own median (traffic, yellow flag, a
# mistake — noise, not pace signal).
OUTLIER_THRESHOLD_PCT = 0.15

# Bland "at the field's median pace, low confidence" fallback — same
# "assume average, don't fabricate an extreme" philosophy used everywhere
# else in this codebase (rookie fallback, new-constructor fallback,
# weather climatology). A driver/session with no long-run data gets THIS,
# never a 0.0 (which would misleadingly read as "tied for fastest").
DEFAULT_PRACTICE_PACE_ROW = {
    "practice_pace_delta_pct": 3.0,  # "a bit off the pace" — a neutral,
    # unremarkable assumption, not "fastest" (0.0) and not "last" (some
    # large number) — see the note on why 0.0 would be actively misleading
    # in this module's docstring above and in the other fallback functions
    # this codebase already uses.
    "practice_lap_count": 0,  # 0 laps — the model can itself learn that a
    # low/zero lap count means "don't trust this row's pace value much",
    # since it's an explicit separate feature rather than baked silently
    # into the pace number itself.
}


def _extract_long_run_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a single session's raw lap data down to laps that represent
    genuine long-run (race-simulation) pace: stints with enough laps to be
    representative, with the out-lap and obvious outlier laps removed.
    """
    if laps_df.empty:
        return laps_df

    kept_rows = []
    for (driver, stint), stint_laps in laps_df.groupby(["Driver", "Stint"]):
        if len(stint_laps) < MIN_LONG_RUN_LAPS:
            continue  # short/quali-sim stint — not race-representative

        stint_laps = stint_laps.sort_values("LapNumber")
        # Drop the first lap of the stint (out-lap).
        stint_laps = stint_laps.iloc[1:]
        if stint_laps.empty:
            continue

        median_time = stint_laps["LapTimeSeconds"].median()
        threshold = median_time * (1 + OUTLIER_THRESHOLD_PCT)
        clean_laps = stint_laps[stint_laps["LapTimeSeconds"] <= threshold]
        if not clean_laps.empty:
            kept_rows.append(clean_laps)

    if not kept_rows:
        return pd.DataFrame(columns=laps_df.columns)
    return pd.concat(kept_rows, ignore_index=True)


def _session_pace_summary(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a single session's ALREADY-FILTERED long-run laps (see
    _extract_long_run_laps), returns one row per driver:
    practice_pace_delta_pct (% slower than the session's fastest driver's
    long-run median — 0.0 = fastest) and practice_lap_count (how many
    laps fed into that driver's median, a confidence signal).
    """
    if laps_df.empty:
        return pd.DataFrame(columns=["driver", "practice_pace_delta_pct", "practice_lap_count"])

    per_driver = laps_df.groupby("Driver")["LapTimeSeconds"].agg(["median", "count"])
    fastest_median = per_driver["median"].min()
    per_driver["practice_pace_delta_pct"] = (
        (per_driver["median"] - fastest_median) / fastest_median * 100
    )
    per_driver = per_driver.rename(columns={"count": "practice_lap_count"})
    per_driver = per_driver.reset_index().rename(columns={"Driver": "driver"})
    return per_driver[["driver", "practice_pace_delta_pct", "practice_lap_count"]]


def build_practice_pace_lookup(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch-builds the FP2 long-run pace lookup for every ALREADY-COMPLETED
    race in `schedule_df` (columns: year, round, date — same shape used by
    weather_features.build_weather_lookup(), and for the identical reason:
    a future round has no FP2 data yet, so it's skipped here rather than
    guaranteed to fail/return nothing useful).

    Returns a DataFrame keyed by (year, round, driver) with
    PRACTICE_PACE_FEATURE_COLUMNS — merge onto a training feature
    dataframe on ["year", "round", "driver"].

    Relies on FastF1's OWN built-in caching (fastf1_client.py already
    calls fastf1.Cache.enable_cache()) rather than a separate cache layer
    like weather_client.py's — FastF1 handles that internally, so repeated
    calls across training runs are already cheap after the first fetch.
    """
    rows = []
    today = pd.Timestamp.now().date()
    skipped_future = 0
    skipped_errors = 0
    rate_limited = False

    for _, race in schedule_df.iterrows():
        if rate_limited:
            # Once FastF1's own client-side rate limiter trips (500 calls/h
            # — see fastf1.req.RateLimitExceededError), it stays tripped
            # for the rest of that hour. Continuing to call get_lap_times()
            # for every remaining race would just fail identically dozens
            # more times — pure log noise and wasted time, and (worse) it
            # meant almost the ENTIRE practice_lookup fell back to
            # DEFAULT_PRACTICE_PACE_ROW in practice, silently wasting the
            # whole feature for that training run. Stop immediately
            # instead — whatever WAS successfully fetched before the limit
            # hit is kept (and permanently cached by FastF1 locally, so a
            # later re-run picks up from there rather than re-fetching
            # from scratch). See RaceMindAI conversation history for the
            # log this was diagnosed from.
            skipped_errors += 1
            continue

        try:
            race_date = pd.Timestamp(str(race["date"])[:10]).date()
        except (ValueError, TypeError):
            skipped_errors += 1
            continue

        if race_date > today:
            skipped_future += 1
            continue  # no FP2 data yet — same reasoning as weather_features.py's future-date skip

        try:
            # Pass the round NUMBER rather than a gp-name string — FastF1
            # accepts either, but round numbers are exact and unambiguous,
            # unlike matching against gp_name strings which can drift out
            # of sync with FastF1's own event-name conventions the same
            # way circuit names drifted out of sync with Jolpica's (see
            # f1_constants.py's canonical_circuit_name() docstring).
            laps = get_lap_times(int(race["year"]), int(race["round"]), "FP2")
        except Exception as e:
            if type(e).__name__ == "RateLimitExceededError":
                print(f"[practice_pace_features] FastF1 rate limit hit "
                      f"after {race['year']} round {race['round']} — "
                      f"stopping this backfill early rather than "
                      f"continuing to fail on every remaining race. "
                      f"Sessions already fetched are cached locally by "
                      f"FastF1; re-run training later (the limit resets "
                      f"hourly) to backfill more.")
                rate_limited = True
                skipped_errors += 1
                continue
            print(f"[practice_pace_features] FP2 fetch failed for "
                  f"{race['year']} round {race['round']}: {e} — skipping "
                  f"(will fall back to DEFAULT_PRACTICE_PACE_ROW at merge time).")
            skipped_errors += 1
            continue

        long_run_laps = _extract_long_run_laps(laps)
        summary = _session_pace_summary(long_run_laps)
        summary["year"] = int(race["year"])
        summary["round"] = int(race["round"])
        rows.append(summary)

    if skipped_future:
        print(f"[practice_pace_features] skipped {skipped_future} future "
              f"race(s) — no FP2 data yet.")
    if skipped_errors:
        print(f"[practice_pace_features] {skipped_errors} race(s) had no "
              f"usable FP2 data (fetch error, no sprint-weekend FP2, etc.) "
              f"— those rows will fall back to the default at merge time.")

    if not rows:
        return pd.DataFrame(columns=["year", "round", "driver"] + PRACTICE_PACE_FEATURE_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def attach_practice_pace_features(feat_df: pd.DataFrame, practice_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merges a practice_lookup (from build_practice_pace_lookup) onto a
    training feature dataframe on ["year", "round", "driver"]. Any row
    with no match (FP2 fetch failed, sprint weekend with no FP2, a driver
    who didn't set a representative long run that session) gets
    DEFAULT_PRACTICE_PACE_ROW rather than NaN — same reasoning as
    weather_features.attach_weather_features().
    """
    df = feat_df.merge(practice_lookup, on=["year", "round", "driver"], how="left")
    missing = df["practice_pace_delta_pct"].isna()
    if missing.any():
        for col, val in DEFAULT_PRACTICE_PACE_ROW.items():
            df.loc[missing, col] = val
        print(f"[practice_pace_features] {missing.sum()} rows had no FP2 "
              f"long-run match — filled with the default pace row.")
    return df


def get_session_practice_pace(year: int, round_: int) -> dict:
    """
    Single-session FP2 long-run pace lookup for INFERENCE — the function
    main.py's /api/predictor/predict route calls, mirroring
    weather_features.get_session_weather()'s role for weather.

    Returns {driver_code: {"practice_pace_delta_pct": ..., "practice_lap_count": ...}}
    for every driver with a usable long run in that session, or an empty
    dict if FP2 hasn't happened yet / the fetch fails — callers should
    treat a missing driver in the returned dict as "use
    DEFAULT_PRACTICE_PACE_ROW for this driver", not as an error.
    """
    try:
        laps = get_lap_times(year, round_, "FP2")
    except Exception as e:
        print(f"[practice_pace_features] FP2 lookup failed for {year} "
              f"round {round_}: {e} — predictions will use the default "
              f"practice pace row for every driver.")
        return {}

    long_run_laps = _extract_long_run_laps(laps)
    summary = _session_pace_summary(long_run_laps)
    return {
        row["driver"]: {
            "practice_pace_delta_pct": row["practice_pace_delta_pct"],
            "practice_lap_count": row["practice_lap_count"],
        }
        for _, row in summary.iterrows()
    }


def default_practice_pace_row() -> dict:
    """Public helper mirroring weather_features.default_weather_row() —
    for callers that need to fill PRACTICE_PACE_FEATURE_COLUMNS when no
    practice_lookup was supplied at all (e.g. train_model() called without
    one), without reaching into this module's internal constant directly."""
    return dict(DEFAULT_PRACTICE_PACE_ROW)
