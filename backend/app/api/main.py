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
from fastapi.middleware.cors import CORSMiddleware

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
    CIRCUIT_TYPE,
)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://race-mind-ai-f1-intelligence-platfo.vercel.app",   # your production Vercel URL
        "https://YOUR-CUSTOM-DOMAIN.com",           # if/when you add a custom domain
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # covers PR preview deployments too
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://race-mind-ai-f1-intelligence-platfo.vercel.app",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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


# ── Response models (just for clean OpenAPI docs — not strictly required) ──
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

    model = train_model(df, use_era_weighting=True)
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

    # Resolve circuit for this round from the real schedule, exactly like
    # app.py does via selected_race_row — not trusting the client to send
    # a correct circuit name.
    sched = get_season_schedule(req.year)
    race_row = sched[sched["round"] == req.round]
    if race_row.empty:
        raise HTTPException(404, f"Round {req.round} not found in {req.year} schedule.")
    circuit_name = race_row.iloc[0]["circuit"]
    circuit_type = CIRCUIT_TYPE.get(circuit_name, "unknown")
    circuit_code = (
        circuit_type_categories.index(circuit_type)
        if circuit_type in circuit_type_categories
        else len(circuit_type_categories)
    )

    hist = get_cached_historical_results(2022, 2026)
    feat_df = build_training_features(hist)

    code_to_team = {d["code"]: d["team"] for d in DRIVERS_2026}

    rows = []
    for entry in req.grid:
        driver_hist = feat_df[feat_df["driver"] == entry.driver]
        row = (
            driver_hist.iloc[-1][FEATURES].to_dict()
            if len(driver_hist) > 0
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
        constructor = code_to_team.get(entry.driver)
        if constructor:
            row = apply_new_constructor_fallback(row, constructor, feat_df)
        rows.append(row)

    rows_df = pd.DataFrame(rows)
    predictions = predict_race(model, rows_df)

    return {
        "circuit": circuit_name,
        "circuit_type": circuit_type,
        "predictions": predictions[
            ["driver", "podium_probability", "predicted_position"]
        ].to_dict(orient="records"),
        # rows_used is returned so /api/predictor/explain can be called
        # afterwards without rebuilding features from scratch client-side.
        "_rows_used": rows_df.to_dict(orient="records"),
    }


class ExplainRequest(BaseModel):
    driver: str
    rows_used: List[dict]  # pass through the "_rows_used" field from /predict


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

    try:
        model = train_qualifying_model(df, use_era_weighting=True)
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
    weather: str = "Dry"  # "Dry" | "Wet" | "Mixed"


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
    circuit_type = CIRCUIT_TYPE.get(circuit_name, "unknown")
    circuit_code = (
        circuit_type_categories.index(circuit_type)
        if circuit_type in circuit_type_categories
        else len(circuit_type_categories)
    )
    is_wet_flag = 1 if req.weather in ("Wet", "Mixed") else 0

    quali_hist = get_cached_historical_qualifying(QUALI_TRAIN_YEAR_START, QUALI_TRAIN_YEAR_END)
    feat_df = build_qualifying_training_features(quali_hist)

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
            "is_wet": is_wet_flag,
            "round": req.round,
            "year": req.year,
        })
        constructor = driver_team_by_code.get(code)
        if constructor:
            row = apply_qualifying_new_constructor_fallback(row, constructor, feat_df)
        row = apply_rookie_fallback(row, d["name"], ROOKIE_2026, feat_df)
        rows.append(row)

    predictions = predict_qualifying_order(model, pd.DataFrame(rows))

    return {
        "circuit": circuit_name,
        "circuit_type": circuit_type,
        "predictions": predictions[
            ["driver", "predicted_quali_position", "confidence"]
        ].to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════════════════
# SEASON CHAMPIONSHIP SIMULATOR  (maps to app.py "Season Championship" page)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/simulate")
def run_simulation(req: SimulateRequest):
    standings = get_driver_standings(req.year, round_num=req.current_round)
    if standings.empty:
        raise HTTPException(404, f"No standings for {req.year} round {req.current_round}")

    sched = get_season_schedule(req.year)
    remaining = sched[sched["round"] > req.current_round].sort_values("round")
    remaining_races = len(remaining)
    remaining_circuits = remaining["circuit"].tolist()

    driver_strengths = build_driver_strengths(standings)
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
