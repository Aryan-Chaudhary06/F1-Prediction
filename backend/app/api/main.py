"""
RaceMind AI — FastAPI layer
Run locally:
    uvicorn app.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs 
"""

import os
import sys
import datetime
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.data.ergast_client import (
    get_driver_standings,
    get_constructor_standings,
    get_season_schedule,
    get_cached_historical_results,
    get_cache_status,
    get_qualifying_results,
)
from app.data.fastf1_client import (
    get_lap_times,
    get_race_results,
    get_driver_telemetry,
    get_session_drivers,
)
from app.models.feature_engineering import (
    build_training_features,
    apply_new_constructor_fallback,
    # Aliased: app.models.qualifying_feature_engineering also exports a
    # function named apply_rookie_fallback with a DIFFERENT signature
    # (row, driver_name, rookie_names, historical_features_df) and is
    # imported below under the same bare name — without aliasing, that
    # later import silently shadows this one and calling this race-model
    # version would raise a TypeError (or worse, silently misbehave if
    # positional args happened to line up). See
    # RaceMindAI_Audit_Phases1-5.md, Phase 2.3.1.
    apply_rookie_fallback as apply_race_rookie_fallback,
    CIRCUIT_TYPE,
)
from app.models.f1_constants import classify_circuit
from app.models.race_predictor import (
    load_or_train_model,
    load_model_metadata,
    model_is_stale,
    model_exists,
    train_model,
    predict_race,
    get_feature_importance,
    FEATURES,
)
from app.models.season_simulator import (
    simulate_season,
    build_driver_strengths,
    build_driver_dnf_rates,
)
from app.models.weather_features import build_weather_lookup, get_session_weather, default_weather_row
from app.models.practice_pace_features import build_practice_pace_lookup, default_practice_pace_row, get_session_practice_pace
from app.data.weather_client import DEFAULT_SESSION_HOUR_LOCAL

try:
    from app.data.ergast_client import get_cached_historical_qualifying
    from app.models.qualifying_predictor import (
        train_qualifying_model, load_or_train_qualifying_model, qualifying_model_exists,
        load_qualifying_model_metadata, qualifying_model_is_stale,
        predict_qualifying_order, FEATURES as QUALI_FEATURES,
    )
    from app.models.qualifying_feature_engineering import (
        build_qualifying_training_features, apply_qualifying_new_constructor_fallback,
        apply_rookie_fallback,
    )
except Exception as e:
    get_cached_historical_qualifying = None
    train_qualifying_model = load_or_train_qualifying_model = None
    qualifying_model_exists = load_qualifying_model_metadata = None
    qualifying_model_is_stale = predict_qualifying_order = None
    QUALI_FEATURES = None
    build_qualifying_training_features = apply_qualifying_new_constructor_fallback = None
    apply_rookie_fallback = None
    print(f"[startup warning] could not import qualifying predictor modules: {e}")


try:
    from app.data.drivers_2026 import DRIVERS_2026
except Exception as e:
    DRIVERS_2026 = None
    print(f"[startup warning] could not import DRIVERS_2026: {e}")

try:
    from app.models.explainability import get_shap_explanation, get_top_factors
except Exception as e:
    get_shap_explanation = None
    get_top_factors = None
    print(f"[startup warning] could not import explainability module: {e}")

try:
    from app.models.driver_dna import build_driver_dna
except Exception as e:
    build_driver_dna = None
    print(f"[startup warning] could not import build_driver_dna: {e}")



app = FastAPI(
    title="RaceMind AI API",
    description="F1 ML predictions, standings, and championship simulation.",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

# Was registered twice (once with allow_origin_regex, once without) —
# FastAPI applied both, which happened to be harmless since the regex
# already covered the one prod origin listed explicitly in each, but it
# was redundant and easy to accidentally drift out of sync. Merged into
# one registration. See RaceMindAI_Audit_Phases1-5.md, Phase 2.3.1.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://race-mind-ai-f1-intelligence-platfo.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRAIN_YEAR_START, TRAIN_YEAR_END = 2022, 2026


_model_cache = {"model": None}


def _get_model():
    """Load the cached model, or load-from-disk / auto-train if missing —
    same fallback chain as the Race Predictor page in app.py (lines
    1027-1051)."""
    if _model_cache["model"] is not None:
        return _model_cache["model"]

    if model_exists():
        model = load_or_train_model()
    else:
        hist_df = get_cached_historical_results(TRAIN_YEAR_START, TRAIN_YEAR_END)
        if hist_df.empty:
            raise HTTPException(
                status_code=503,
                detail="No trained model exists and training data could not be "
                       "fetched. Check network connectivity to the Jolpica API.",
            )
        model = load_or_train_model(historical_df=hist_df)

    _model_cache["model"] = model
    return model


QUALI_TRAIN_YEAR_START, QUALI_TRAIN_YEAR_END = 2022, 2026
_quali_model_cache = {"model": None}


def _get_quali_model():
    """Same load-from-disk / auto-train fallback chain as _get_model(), for
    the separate qualifying ranker."""
    if load_or_train_qualifying_model is None:
        raise HTTPException(501, "Qualifying predictor module not available on server.")

    if _quali_model_cache["model"] is not None:
        return _quali_model_cache["model"]

    if qualifying_model_exists():
        model = load_or_train_qualifying_model()
    else:
        hist_df = get_cached_historical_qualifying(QUALI_TRAIN_YEAR_START, QUALI_TRAIN_YEAR_END)
        if hist_df.empty:
            raise HTTPException(
                status_code=503,
                detail="No trained qualifying model exists and training data could not "
                       "be fetched. Check network connectivity to the Jolpica API.",
            )
        model = load_or_train_qualifying_model(historical_quali_df=hist_df)

    _quali_model_cache["model"] = model
    return model


class TrainResponse(BaseModel):
    trained: bool
    rows_trained_on: int
    accuracy: float
    years_trained_on: List[int]


class GridEntry(BaseModel):
    driver: str       # 3-letter driver code, e.g. "NOR"
    grid_position: int


class PredictRequest(BaseModel):
    year: int
    round: int
    grid: List[GridEntry]
    weather: Optional[str] = None  # "Dry" | "Mixed" | "Wet" | None (None = use live forecast)


class SimulateRequest(BaseModel):
    year: int
    current_round: int          # round number standings are taken AT
    n_simulations: int = 10000
    noise_std: float = 0.20
    safety_car_multiplier: float = 1.0
    use_dnf_modeling: bool = True


# ══════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.datetime.now().isoformat()}


# ══════════════════════════════════════════════════════════════════════════
# STANDINGS  (maps to app.py "Live Standings" / "Dashboard" pages)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/standings/drivers/{year}")
def driver_standings(year: int, round: Optional[int] = None):
    df = get_driver_standings(year, round_num=round)
    if df.empty:
        raise HTTPException(404, f"No driver standings found for {year}")
    return df.to_dict(orient="records")


@app.get("/api/standings/constructors/{year}")
def constructor_standings(year: int, round: Optional[int] = None):
    df = get_constructor_standings(year, round_num=round)
    if df.empty:
        raise HTTPException(404, f"No constructor standings found for {year}")
    return df.to_dict(orient="records")


@app.get("/api/schedule/{year}")
def schedule(year: int):
    df = get_season_schedule(year)
    if df.empty:
        raise HTTPException(404, f"No schedule found for {year}")
    return df.sort_values("round").to_dict(orient="records")


@app.get("/api/qualifying/{year}/{round_num}")
def qualifying(year: int, round_num: int):
    df = get_qualifying_results(year, round_num)
    if df.empty:
        raise HTTPException(404, f"No qualifying results for {year} round {round_num}")
    return df.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════
# RACE ANALYSIS  (maps to app.py "Race Analysis" page — FastF1)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/race-analysis/laps")
def race_laps(year: int, gp: str, session_type: str = "R"):
    """gp = official GP name as used by FastF1, e.g. 'British Grand Prix'."""
    try:
        df = get_lap_times(year, gp, session_type)
    except Exception as e:
        raise HTTPException(502, f"FastF1 could not load this session: {e}")
    return df.to_dict(orient="records")


@app.get("/api/race-analysis/results")
def race_results(year: int, gp: str):
    try:
        df = get_race_results(year, gp)
    except Exception as e:
        raise HTTPException(502, f"FastF1 could not load results: {e}")
    return df.to_dict(orient="records")


@app.get("/api/race-analysis/telemetry")
def telemetry(year: int, gp: str, driver: str, lap_number: Optional[int] = None):
    try:
        df = get_driver_telemetry(year, gp, driver, lap_number)
    except Exception as e:
        raise HTTPException(502, f"FastF1 could not load telemetry: {e}")
    # Time column is a Timedelta — convert to seconds for clean JSON
    if "Time" in df.columns:
        df["Time"] = df["Time"].dt.total_seconds()
    return df.to_dict(orient="records")


@app.get("/api/race-analysis/drivers")
def session_drivers(year: int, gp: str, session_type: str = "R"):
    try:
        return get_session_drivers(year, gp, session_type)
    except Exception as e:
        raise HTTPException(502, f"FastF1 could not load session: {e}")


# ══════════════════════════════════════════════════════════════════════════
# RACE PREDICTOR  (maps to app.py "Race Predictor" page)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/predictor/status")
def predictor_status():
    """Mirrors the model-freshness banner at the top of the Predictor page."""
    meta = load_model_metadata()
    return {
        "model_exists": model_exists(),
        "is_stale": model_is_stale(max_age_days=7),
        "metadata": meta,
    }


@app.post("/api/predictor/train", response_model=TrainResponse)
def predictor_train(force_refresh: bool = False):
    """Equivalent to clicking 'Train / Refresh Model' in the UI."""
    df = get_cached_historical_results(TRAIN_YEAR_START, TRAIN_YEAR_END, force_refresh=force_refresh)
    if df.empty:
        raise HTTPException(503, "No training data available — check network connectivity.")

    # Weather backfill — see RaceMindAI_Redesign_Phases6-7.md §6.4/§7.5
    # step 3. Uses one schedule fetch per season rather than per-row, then
    # weather_client's permanent on-disk cache means re-running this after
    # the first backfill only costs API calls for genuinely NEW races
    # since the last training run. Soft-fails to "no weather signal" (via
    # train_model's own weather_lookup=None fallback) rather than blocking
    # training entirely if the weather API is unreachable — same
    # philosophy as this codebase's other optional-dependency handling.
    try:
        schedules = pd.concat(
            [get_season_schedule(y).assign(year=y) for y in range(TRAIN_YEAR_START, TRAIN_YEAR_END + 1)],
            ignore_index=True,
        )
        weather_lookup = build_weather_lookup(schedules)
    except Exception as e:
        print(f"[predictor_train] weather backfill failed ({e}) — training without weather signal.")
        weather_lookup = None
        schedules = None

    # FP2 long-run pace backfill — see RaceMindAI_Redesign_Phases6-7.md
    # §6.5 (designed but not built until now). Reuses the same `schedules`
    # fetched above rather than re-fetching; each individual FP2 fetch is
    # cached by FastF1 itself (fastf1_client.py already calls
    # fastf1.Cache.enable_cache()), so this is cheap after the first run.
    # Soft-fails to "no practice signal" the same way weather does above —
    # a slow/unreachable FastF1 backend shouldn't block training entirely.
    try:
        if schedules is None:
            schedules = pd.concat(
                [get_season_schedule(y).assign(year=y) for y in range(TRAIN_YEAR_START, TRAIN_YEAR_END + 1)],
                ignore_index=True,
            )
        practice_lookup = build_practice_pace_lookup(schedules)
    except Exception as e:
        print(f"[predictor_train] practice pace backfill failed ({e}) — training without practice signal.")
        practice_lookup = None

    model = train_model(df, use_era_weighting=True, weather_lookup=weather_lookup, practice_lookup=practice_lookup)
    _model_cache["model"] = model  # refresh in-memory cache too

    meta = load_model_metadata()
    return TrainResponse(
        trained=True,
        rows_trained_on=meta["rows_trained_on"],
        accuracy=meta["accuracy"],
        years_trained_on=meta["years_trained_on"],
    )


@app.get("/api/predictor/cache-status")
def predictor_cache_status():
    df = get_cache_status(TRAIN_YEAR_START, TRAIN_YEAR_END)
    return df.to_dict(orient="records")


@app.get("/api/predictor/grid-defaults/{year}")
def predictor_grid_defaults(year: int):
    """
    Returns the 2026 driver list (code, name, team) for the frontend to
    render the editable grid-position table, equivalent to DRIVERS_2026
    used directly in app.py.

    NOTE: this assumes DRIVERS_2026 is a list of dicts shaped like
    {"code": "NOR", "name": "Lando Norris", "team": "McLaren"}, based on
    how app.py uses it (name_to_code_map = {d["name"]: d["code"] ...} and
    driver_team_by_code = {... : d["team"] ...}). If your actual
    drivers_2026.py uses different key names, adjust the dict access below
    to match — the rest of this route doesn't need to change.
    """
    if DRIVERS_2026 is None:
        raise HTTPException(
            501,
            "drivers_2026.py could not be imported on the server — "
            "check the deploy logs for the real import error.",
        )
    return DRIVERS_2026


@app.post("/api/predictor/predict")
def predictor_predict(req: PredictRequest):
    """
    Equivalent to clicking 'Predict Podium' in the UI, with the grid table
    already filled in. Returns podium probabilities + predicted finishing
    order, same shape as predictions[["driver","podium_probability",
    "predicted_position"]] in app.py.
    """
    if DRIVERS_2026 is None:
        raise HTTPException(501, "drivers_2026.py not available on server.")

    model = _get_model()

    circuit_type_categories = sorted(set(CIRCUIT_TYPE.values()) | {"unknown"})

    sched = get_season_schedule(req.year)
    race_row = sched[sched["round"] == req.round]
    if race_row.empty:
        raise HTTPException(404, f"Round {req.round} not found in {req.year} schedule.")
    circuit_name = race_row.iloc[0]["circuit"]
    circuit_type = classify_circuit(circuit_name)  # was a raw CIRCUIT_TYPE.get() that bypassed canonical_circuit_name() — see f1_constants.py
    circuit_code = (
        circuit_type_categories.index(circuit_type)
        if circuit_type in circuit_type_categories
        else len(circuit_type_categories)
    )

    # Weather — see RaceMindAI_Redesign_Phases6-7.md §6.4. One lookup for
    # the whole race (same session, same weather for every driver), not
    # per-driver. Uses req.weather as a manual override if given (mirrors
    # qualifying_predict's existing scenario-exploration UX), otherwise a
    # live forecast for the scheduled race date/time.
    try:
        race_date = datetime.datetime.fromisoformat(str(race_row.iloc[0]["date"])).date()
    except (ValueError, TypeError):
        race_date = datetime.date.today()
    race_time_str = race_row.iloc[0].get("time")
    session_hour = int(race_time_str[:2]) if race_time_str else DEFAULT_SESSION_HOUR_LOCAL
    weather_row = get_session_weather(
        circuit_name, race_date, session_hour_local=session_hour, manual_override=req.weather,
    )

    hist = get_cached_historical_results(2022, 2026)
    feat_df = build_training_features(hist)
    # feat_df doesn't have weather columns — build_training_features() is a
    # pure function of results data and never touches weather (see
    # weather_features.py's module docstring: it's a deliberately separate
    # step). Without this, `driver_hist.iloc[-1][FEATURES]` below KeyErrors
    # on every weather column, since they're in FEATURES but not in
    # feat_df's columns at all. The placeholder values here don't matter —
    # every row gets overwritten with the real session weather via
    # weather_row a few lines down; this just needs the columns to EXIST.
    for _col, _val in default_weather_row().items():
        feat_df[_col] = _val
    for _col, _val in default_practice_pace_row().items():
        feat_df[_col] = _val

    # FP2 long-run pace — see RaceMindAI_Redesign_Phases6-7.md §6.5.
    # UNLIKE weather (one shared value for the whole race), this is
    # PER-DRIVER, so it's a dict keyed by driver code, applied inside the
    # per-driver loop below rather than a single row.update() beforehand.
    # Only returns real data if this race's FP2 has already happened
    # (get_session_practice_pace soft-fails to {} otherwise, per its own
    # docstring) — a driver missing from this dict falls back to
    # DEFAULT_PRACTICE_PACE_ROW via the feat_df placeholder fill above.
    practice_pace_by_driver = get_session_practice_pace(req.year, req.round)

    code_to_team = {d["code"]: d["team"] for d in DRIVERS_2026}

    rows = []
    for entry in req.grid:
        driver_hist = feat_df[feat_df["driver"] == entry.driver]
        has_history = len(driver_hist) > 0
        row = (
            driver_hist.iloc[-1][FEATURES].to_dict()
            if has_history
            else {f: 0.0 for f in FEATURES}
        )
        row.update({
            "driver": entry.driver,
            "grid": entry.grid_position,
            "grid_squared": entry.grid_position ** 2,
            "circuit_type_code": circuit_code,
            "round": req.round,
            "year": req.year,
        })
        # Overwrite whatever stale weather value came from driver_hist's
        # last cached race with THIS race's actual forecast — every driver
        # in the same race shares the same weather, so this must come
        # after the driver_hist base row, not be left to that row's own
        # (irrelevant, different-race) historical weather.
        row.update({k: v for k, v in weather_row.items() if k in FEATURES})
        # Same idea for practice pace, but per-driver rather than shared —
        # a driver with no entry in practice_pace_by_driver (FP2 hasn't
        # happened yet, or they had no representative long run that
        # session) keeps whatever DEFAULT_PRACTICE_PACE_ROW value the
        # feat_df placeholder fill above already gave them.
        if entry.driver in practice_pace_by_driver:
            row.update(practice_pace_by_driver[entry.driver])
        constructor = code_to_team.get(entry.driver)
        if constructor:
            row = apply_new_constructor_fallback(row, constructor, feat_df)
        # Was missing entirely — a driver with zero rows in feat_df (any
        # true rookie, i.e. Lindblad now that ROOKIE_2026 is corrected)
        # fell straight to the fabricated {f: 0.0 ...} above instead of
        # the field-average fallback. apply_rookie_fallback() unconditionally
        # overwrites the rookie_fields with column means, so it must only be
        # called when there's genuinely no history to overwrite — matches its
        # documented contract in feature_engineering.py.
        # See RaceMindAI_Audit_Phases1-5.md, Phase 2.3.1.
        if not has_history:
            row = apply_race_rookie_fallback(row, entry.driver, feat_df)
        rows.append(row)

    rows_df = pd.DataFrame(rows)
    predictions = predict_race(model, rows_df)

    return {
        "circuit": circuit_name,
        "circuit_type": circuit_type,
        "weather": {k: v for k, v in weather_row.items()},
        # "calibrated" tells the frontend whether podium_probability is a
        # real (isotonic-calibrated) probability or the raw XGBoost score
        # — matters because a raw score can look like a probability
        # without actually being one. See
        # RaceMindAI_Redesign_Phases6-7.md §6.5 / Audit Phase 5.
        "predictions": predictions[
            ["driver", "podium_probability", "raw_podium_probability", "calibrated", "predicted_position"]
        ].to_dict(orient="records"),
        "_rows_used": rows_df.to_dict(orient="records"),
    }


class ExplainRequest(BaseModel):
    driver: str
    rows_used: List[dict]  


@app.post("/api/predictor/explain")
def predictor_explain(req: ExplainRequest, top_n: int = 6):
    """SHAP explainability for one driver's prediction — equivalent to
    selecting a driver in the 'Why did the model predict this?' section."""
    if get_shap_explanation is None or get_top_factors is None:
        raise HTTPException(
            501,
            "app/models/explainability.py could not be imported on the "
            "server — SHAP explanations unavailable until that's fixed.",
        )

    model = _get_model()
    rows_df = pd.DataFrame(req.rows_used)

    try:
        shap_df = get_shap_explanation(model, rows_df, FEATURES)
        factors = get_top_factors(shap_df, req.driver, top_n=top_n)
    except Exception as e:
        raise HTTPException(500, f"SHAP explanation failed: {e}")

    # Kept as a bare list (NOT wrapped in {explanation_method, factors})
    # since the frontend (Predictor.jsx / api.js) already consumes this
    # endpoint expecting an array — wrapping it would silently break that
    # working integration. explanation_method ("shap" | "approximate") is
    # already present as a per-row field via get_shap_explanation()'s fix
    # (see RaceMindAI_Audit_Phases1-5.md Phase 1 finding #9 /
    # explainability.py) — the frontend can read `factors[0].explanation_method`
    # today without any backend change, and a future frontend update could
    # surface a "these are approximate" notice from that per-row field
    # whenever it's convenient to add, without needing another API shape
    # change then either.
    return factors.to_dict(orient="records")


@app.get("/api/predictor/feature-importance")
def predictor_feature_importance():
    model = _get_model()
    df = get_feature_importance(model)
    return df.to_dict(orient="records")


# ── Qualifying predictor ────────────────────────────────────────────────
# Separate model/pipeline from everything above — predicts single-lap grid
# order (XGBRanker) rather than race podium probability (XGBClassifier).

class QualiTrainResponse(BaseModel):
    trained: bool
    sessions_trained_on: int
    rows_trained_on: int
    pole_accuracy: float
    top3_accuracy: float
    years_trained_on: List[int]


@app.get("/api/qualifying/status")
def qualifying_status():
    if qualifying_model_exists is None:
        raise HTTPException(501, "Qualifying predictor module not available on server.")
    meta = load_qualifying_model_metadata()
    return {
        "model_exists": qualifying_model_exists(),
        "is_stale": qualifying_model_is_stale(max_age_days=7),
        "metadata": meta,
    }


@app.post("/api/qualifying/train", response_model=QualiTrainResponse)
def qualifying_train(force_refresh: bool = False):
    if train_qualifying_model is None:
        raise HTTPException(501, "Qualifying predictor module not available on server.")

    df = get_cached_historical_qualifying(
        QUALI_TRAIN_YEAR_START, QUALI_TRAIN_YEAR_END, force_refresh=force_refresh
    )
    if df.empty:
        raise HTTPException(503, "No qualifying training data available — check network connectivity.")

    # Same weather backfill as predictor_train — see
    # RaceMindAI_Redesign_Phases6-7.md §6.4. Qualifying happens the day
    # before (or same day as) the race, but this app doesn't track
    # qualifying's own session time separately from the race's, so this
    # reuses the race schedule/date as an approximation of qualifying day —
    # close enough for weather purposes (same weekend, same region) but
    # worth a real qualifying-specific date/time if Jolpica ever exposes
    # one distinctly.
    try:
        schedules = pd.concat(
            [get_season_schedule(y).assign(year=y)
             for y in range(QUALI_TRAIN_YEAR_START, QUALI_TRAIN_YEAR_END + 1)],
            ignore_index=True,
        )
        weather_lookup = build_weather_lookup(schedules)
    except Exception as e:
        print(f"[qualifying_train] weather backfill failed ({e}) — training without weather signal.")
        weather_lookup = None

    try:
        model = train_qualifying_model(df, use_era_weighting=True, weather_lookup=weather_lookup)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _quali_model_cache["model"] = model

    meta = load_qualifying_model_metadata()
    return QualiTrainResponse(
        trained=True,
        sessions_trained_on=meta["sessions_trained_on"],
        rows_trained_on=meta["rows_trained_on"],
        pole_accuracy=meta["pole_accuracy"],
        top3_accuracy=meta["top3_accuracy"],
        years_trained_on=meta["years_trained_on"],
    )


class QualiPredictRequest(BaseModel):
    year: int
    round: int
    weather: Optional[str] = None  # "Dry" | "Wet" | "Mixed" | None (None = use live forecast)


@app.post("/api/qualifying/predict")
def qualifying_predict(req: QualiPredictRequest):
    """
    Predicts qualifying order for every 2026 driver in one session —
    equivalent to clicking 'Predict Qualifying Order' in the UI. No grid
    input needed (unlike /api/predictor/predict) since qualifying position
    IS the thing being predicted.
    """
    if DRIVERS_2026 is None:
        raise HTTPException(501, "drivers_2026.py not available on server.")
    if predict_qualifying_order is None:
        raise HTTPException(501, "Qualifying predictor module not available on server.")

    model = _get_quali_model()

    circuit_type_categories = sorted(set(CIRCUIT_TYPE.values()) | {"unknown"})

    sched = get_season_schedule(req.year)
    race_row = sched[sched["round"] == req.round]
    if race_row.empty:
        raise HTTPException(404, f"Round {req.round} not found in {req.year} schedule.")
    circuit_name = race_row.iloc[0]["circuit"]
    circuit_type = classify_circuit(circuit_name)  # was a raw CIRCUIT_TYPE.get() that bypassed canonical_circuit_name() — see f1_constants.py
    circuit_code = (
        circuit_type_categories.index(circuit_type)
        if circuit_type in circuit_type_categories
        else len(circuit_type_categories)
    )

    # `is_wet_flag` (a hand-set boolean derived from req.weather) REPLACED
    # by a real weather lookup — live forecast when req.weather is None,
    # or the same manual-override path as before when the user picks a
    # condition explicitly. See RaceMindAI_Redesign_Phases6-7.md §6.4.
    # Qualifying is typically the day before the race; this app doesn't
    # currently track a separate qualifying date from the race schedule,
    # so this uses the race date as an approximation (same weekend, same
    # region — close enough for weather purposes; see the matching note in
    # qualifying_train() above).
    try:
        session_date = datetime.datetime.fromisoformat(str(race_row.iloc[0]["date"])).date()
    except (ValueError, TypeError):
        session_date = datetime.date.today()
    race_time_str = race_row.iloc[0].get("time")
    session_hour = int(race_time_str[:2]) if race_time_str else DEFAULT_SESSION_HOUR_LOCAL
    weather_row = get_session_weather(
        circuit_name, session_date, session_hour_local=session_hour, manual_override=req.weather,
    )

    quali_hist = get_cached_historical_qualifying(QUALI_TRAIN_YEAR_START, QUALI_TRAIN_YEAR_END)
    feat_df = build_qualifying_training_features(quali_hist)
    # Same reason as predictor_predict above — feat_df has no weather
    # columns until attach_weather_features() runs (that only happens
    # inside train_qualifying_model(), not here), so QUALI_FEATURES
    # indexing below would KeyError without this. Overwritten with real
    # session weather via weather_row right after.
    for _col, _val in default_weather_row().items():
        feat_df[_col] = _val

    driver_team_by_code = {d["code"]: d["team"] for d in DRIVERS_2026}
    from app.data.drivers_2026 import ROOKIE_2026

    rows = []
    for d in DRIVERS_2026:
        code = d["code"]
        dh = feat_df[feat_df["driver"] == code]
        row = dh.iloc[-1][QUALI_FEATURES].to_dict() if len(dh) > 0 else {f: 0.0 for f in QUALI_FEATURES}
        row.update({
            "driver": code,
            "circuit_type_code": circuit_code,
            "round": req.round,
            "year": req.year,
        })
        # Overwrite with THIS session's actual weather — every driver in
        # the same qualifying session shares it, same reasoning as the
        # race predictor above.
        row.update({k: v for k, v in weather_row.items() if k in QUALI_FEATURES})
        constructor = driver_team_by_code.get(code)
        if constructor:
            row = apply_qualifying_new_constructor_fallback(row, constructor, feat_df)
        row = apply_rookie_fallback(row, d["name"], ROOKIE_2026, feat_df)
        rows.append(row)

    predictions = predict_qualifying_order(model, pd.DataFrame(rows))

    return {
        "circuit": circuit_name,
        "circuit_type": circuit_type,
        "weather": {k: v for k, v in weather_row.items()},
        "predictions": predictions[
            ["driver", "predicted_quali_position", "confidence"]
        ].to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════════════════
# SEASON CHAMPIONSHIP SIMULATOR  (maps to app.py "Season Championship" page)
# ══════════════════════════════════════════════════════════════════════════

def _compute_model_driver_strengths(standings: pd.DataFrame) -> dict:
    """
    Reconnects the season simulator to the trained race podium classifier,
    instead of letting the simulator run on the points-only heuristic
    alone. See RaceMindAI_Redesign_Phases6-7.md §6.5 / Audit
    Phase 2.3 finding.

    For each driver in the current standings, takes their most recent
    cached race feature row (rolling form, constructor pace, etc. as of
    their last race) and asks the trained model for a podium probability.

    IMPORTANT CAVEAT: this is necessarily an approximation, not a
    per-circuit prediction — the season simulator runs many different
    FUTURE races, each at a different circuit with a different grid
    result, neither of which is known yet. This reuses each driver's last
    known grid/circuit/weather context as a stand-in, so it reflects
    roughly "how strong does the model currently think this driver's form
    is" rather than anything race-specific. Still a meaningfully better
    prior than championship points alone, which lag current form (e.g. a
    driver on a hot streak after a slow start won't show it in points
    yet) — but it should NOT be read as "the model predicts a podium
    probability of X for every remaining race."

    Returns {driver: podium_probability}. Returns {} (a safe no-op — the
    caller falls back to pure points-based strength) on any failure: no
    trained model yet, no feature history available, etc. — same
    soft-fail philosophy as this codebase's other optional-signal
    handling (e.g. weather_client's climatology fallback).
    """
    try:
        model = _get_model()
    except Exception as e:
        print(f"[_compute_model_driver_strengths] no usable race model ({e}) — "
              f"season sim will use points-only strength.")
        return {}

    try:
        hist = get_cached_historical_results(2022, 2026)
        feat_df = build_training_features(hist)
        for col, val in default_weather_row().items():
            feat_df[col] = val  # placeholder — no specific upcoming
            # session to fetch real weather for here, see caveat above;
            # these columns just need to exist for the FEATURES indexing
            # below, same reasoning as the predict routes.

        rows = []
        for driver in standings["driver"]:
            dh = feat_df[feat_df["driver"] == driver]
            if dh.empty:
                continue  # e.g. a rookie with no feature history yet —
                # build_driver_strengths() falls back to points-only for
                # this specific driver, not the whole grid.
            row = dh.iloc[-1][FEATURES].to_dict()
            row["driver"] = driver
            rows.append(row)

        if not rows:
            return {}

        preds = predict_race(model, pd.DataFrame(rows))
        return dict(zip(preds["driver"], preds["podium_probability"]))
    except Exception as e:
        print(f"[_compute_model_driver_strengths] failed ({e}) — "
              f"season sim will use points-only strength.")
        return {}


@app.post("/api/simulate")
def run_simulation(req: SimulateRequest):
    standings = get_driver_standings(req.year, round_num=req.current_round)
    if standings.empty:
        raise HTTPException(404, f"No standings for {req.year} round {req.current_round}")

    sched = get_season_schedule(req.year)
    remaining = sched[sched["round"] > req.current_round].sort_values("round")
    remaining_races = len(remaining)
    remaining_circuits = remaining["circuit"].tolist()

    # Reconnected to the trained race model — see
    # RaceMindAI_Redesign_Phases6-7.md §6.5 / Audit Phase 2.3 finding
    # ("season sim disconnected from trained models"). Previously
    # driver_strengths was ONLY build_driver_strengths(standings), a pure
    # points-based heuristic with no relationship to the podium classifier
    # this whole app is built around. model_blend_weight=0.5 is a starting
    # point, not a tuned constant — see build_driver_strengths()'s
    # docstring for what changing it does.
    model_podium_probs = _compute_model_driver_strengths(standings)
    driver_strengths = build_driver_strengths(
        standings, model_podium_probs=model_podium_probs, model_blend_weight=0.5
    )
    driver_constructors = dict(zip(standings["driver"], standings["constructor"]))

    dnf_rates = None
    if req.use_dnf_modeling:
        hist = get_cached_historical_results(2022, 2026)
        dnf_rates = build_driver_dnf_rates(hist)

    results = simulate_season(
        current_standings=standings,
        remaining_races=remaining_races,
        driver_strengths=driver_strengths,
        n_simulations=req.n_simulations,
        remaining_circuits=remaining_circuits,
        safety_car_multiplier=req.safety_car_multiplier,
        dnf_rates=dnf_rates,
        driver_constructors=driver_constructors,
        noise_std=req.noise_std,
    )

    constructors = get_constructor_standings(req.year, round_num=req.current_round)

    return {
        "remaining_races": remaining_races,
        "results": results.to_dict(orient="records"),
        "constructor_standings": constructors.to_dict(orient="records"),
        # True if the trained race model's podium-probability signal was
        # actually blended into driver_strengths for this run; False means
        # it silently fell back to the old points-only heuristic (no
        # trained model available, or no feature history for any driver
        # in the standings) — see _compute_model_driver_strengths()'s
        # docstring above for the fallback conditions.
        "model_signal_used": bool(model_podium_probs),
    }


# ══════════════════════════════════════════════════════════════════════════
# DRIVER DYNAMICS  (maps to app.py "Driver Dynamics" page)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/driver-dynamics")
def driver_dynamics():
    """
    Returns the 6-dimension DNA profile (street, power, technical,
    high_downforce, consistency, race_craft) for every driver.

    NOTE: app/models/driver_dna.py wasn't available when this file was
    written. This route assumes build_driver_dna(hist_df) returns a
    DataFrame with a "driver" column plus those 6 numeric columns, based
    on exactly how app.py consumes it (lines ~1507-1576: `dna[d].iloc[0]`
    for d in DIMENSIONS). If the real signature differs, only this route
    needs updating — everything else in this file is unaffected.
    """
    if build_driver_dna is None:
        raise HTTPException(
            501,
            "app/models/driver_dna.py could not be imported on the server.",
        )
    hist = get_cached_historical_results(2022, 2026)
    dna = build_driver_dna(hist)
    return dna.to_dict(orient="records")
