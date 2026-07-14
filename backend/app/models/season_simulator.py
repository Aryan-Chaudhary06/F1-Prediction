import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Was a local copy identical to feature_engineering.POINTS_MAP and
# driver_dna.POINTS_MAP — now the shared definition. Kept as POINTS_SYSTEM
# (this file's original name) via alias so every other reference in this
# file stays unchanged. See RaceMindAI_Redesign_Phases6-7.md §6.5.
from app.models.f1_constants import POINTS_MAP as POINTS_SYSTEM

# ── Circuit-level safety car probability ────────────────────────────────────
# Keyed by the SAME official circuit names used everywhere else in this
# codebase (feature_engineering.CIRCUIT_TYPE, ergast_client schedule rows),
# not a separate slug system — one fewer place for circuit names to drift
# out of sync with each other.
#
# These are hand-set approximations based on well-known SC/VSC tendencies
# at each track (tight street circuits and high-attrition venues run hot;
# wide, low-walled circuits run cooler), not derived from a dataset of
# actual historical safety car counts. Treat them as a reasonable starting
# point — worth replacing with real historical SC-rate-per-circuit stats
# from FastF1/Ergast race-control data if more precision matters later.
SAFETY_CAR_PROBABILITY_BY_CIRCUIT = {
    "Circuit de Monaco": 0.72,
    "Baku City Circuit": 0.68,
    "Marina Bay Street Circuit": 0.65,
    "Jeddah Corniche Circuit": 0.55,
    "Las Vegas Strip Street Circuit": 0.50,
    "Miami International Autodrome": 0.48,
    "Albert Park Grand Prix Circuit": 0.45,
    "Circuit Gilles Villeneuve": 0.45,
    "Suzuka Circuit": 0.35,
    "Circuit de Spa-Francorchamps": 0.42,
    "Silverstone Circuit": 0.38,
    "Autodromo Nazionale di Monza": 0.28,
    "Bahrain International Circuit": 0.30,
    "Shanghai International Circuit": 0.32,
    "Autodromo Enzo e Dino Ferrari": 0.40,
    "Circuit de Barcelona-Catalunya": 0.25,
    "Red Bull Ring": 0.30,
    "Hungaroring": 0.35,
    "Circuit Zandvoort": 0.38,
    "Circuit of the Americas": 0.30,
    "Autodromo Hermanos Rodriguez": 0.40,
    "Autodromo Jose Carlos Pace": 0.45,
    "Lusail International Circuit": 0.30,
    "Yas Marina Circuit": 0.25,
    # NEW for 2026 — a genuinely new street circuit was previously falling
    # to DEFAULT_SAFETY_CAR_PROBABILITY (0.40) here since it had no entry
    # at all, which is very likely an underestimate: new/tight street
    # circuits in this table run 0.48-0.72 (Miami 0.48, Jeddah 0.55, Baku
    # 0.68, Monaco 0.72), not the generic default. Set in line with those
    # rather than left as a silent gap. See RaceMindAI_Audit_Phases1-5.md
    # Phase 2.5.1 finding #5 — replace with a real observed rate once
    # actual 2026 data exists.
    "Madring": 0.55,
    # Was missing entirely from every taxonomy dict in the repo (not a
    # naming-variant issue like the fixes above) — hosted the 2022 French
    # GP, within this app's training window. Wide, low-walled circuit with
    # generous runoff — similar SC profile to Barcelona/COTA, not a tight
    # street circuit.
    "Circuit Paul Ricard": 0.25,
}
DEFAULT_SAFETY_CAR_PROBABILITY = 0.40

# Approximate per-driver DNF rate (mechanical failure, crash, etc.) per
# race, independent of the safety-car mechanic above. Like the SC table,
# these are reasonable hand-set approximations, not pulled from a live
# rolling stat — season_simulator has no access to per-driver historical
# DNF rates the way feature_engineering's constructor_dnf_rate does, since
# this module only ever receives driver_strengths (a single float per
# driver), not full race history. If more accuracy matters, the natural
# upgrade is to pass per-driver DNF rates in alongside driver_strengths,
# computed the same way feature_engineering.py already computes
# constructor_dnf_rate.
DEFAULT_DNF_RATE = 0.06

# New constructors (Audi, Cadillac as of 2026) have no in-season reliability
# track record yet — bump their DNF chance until that changes.
NEW_CONSTRUCTORS_2026 = {"Audi", "Cadillac"}
DNF_RATE_BOOST_NEW_CONSTRUCTOR = 0.08  # +8 percentage points


def get_safety_car_probability(circuit_name: str, multiplier: float = 1.0) -> float:
    """
    Returns the (multiplier-adjusted) probability that a safety car or VSC
    occurs at the given circuit. multiplier comes from the UI's "Safety car
    frequency" slider (0.5x-2x) — applied here rather than baked into the
    table so the table stays a clean baseline.
    """
    base = SAFETY_CAR_PROBABILITY_BY_CIRCUIT.get(circuit_name, DEFAULT_SAFETY_CAR_PROBABILITY)
    return float(np.clip(base * multiplier, 0.0, 1.0))


def simulate_race(driver_strengths: Dict[str, float],
                  noise_std: float = 0.15,
                  circuit_name: Optional[str] = None,
                  safety_car_multiplier: float = 1.0,
                  dnf_rates: Optional[Dict[str, float]] = None,
                  driver_constructors: Optional[Dict[str, str]] = None,
                  rng: Optional[np.random.Generator] = None) -> List[str]:
    """
    Simulates one race's finishing order.

    New behavior vs. the original version of this function:
    - If `circuit_name` is given, draws a Bernoulli safety-car event using
      that circuit's probability (scaled by safety_car_multiplier). When a
      safety car occurs, every driver's finishing position gets ±3 random
      noise added on top of the base pace ordering — modeling the chaos of
      undercut/overcut timing and SC restarts scrambling the order.
    - If `dnf_rates` is given, each driver is independently rolled for a
      DNF. A DNF'd driver is moved to the back of the order (still included
      in the returned list — the caller decides how/whether to score them;
      simulate_season below scores them with 0 points, matching how a real
      retirement usually nets zero points).

    Both new behaviors are OPT-IN via their respective arguments staying
    None/default — calling this with just driver_strengths behaves exactly
    like the original implementation, so any other caller relying on the
    old signature/behavior keeps working unchanged.
    """
    rng = rng or np.random.default_rng()

    scores = {}
    for driver, strength in driver_strengths.items():
        noise = rng.normal(0, noise_std)
        scores[driver] = max(0, strength + noise)

    finish_order = sorted(scores, key=scores.get, reverse=True)

    safety_car_occurred = False
    if circuit_name is not None:
        sc_prob = get_safety_car_probability(circuit_name, safety_car_multiplier)
        safety_car_occurred = rng.random() < sc_prob

    if safety_car_occurred:
        # Reshuffle: add position-noise to each driver's rank, then re-sort.
        # ±3 positions captures undercut/overcut and restart chaos without
        # producing fully random results — a midfield driver under a randomly
        # timed safety car might gain or lose a few spots, not jump from
        # P18 to P1.
        ranks = {d: i for i, d in enumerate(finish_order)}
        jittered = {d: r + rng.integers(-3, 4) for d, r in ranks.items()}
        finish_order = sorted(jittered, key=jittered.get)

    if dnf_rates:
        finishers, dnfs = [], []
        for driver in finish_order:
            rate = dnf_rates.get(driver, DEFAULT_DNF_RATE)
            if driver_constructors and driver_constructors.get(driver) in NEW_CONSTRUCTORS_2026:
                rate += DNF_RATE_BOOST_NEW_CONSTRUCTOR
            if rng.random() < rate:
                dnfs.append(driver)
            else:
                finishers.append(driver)
        finish_order = finishers + dnfs  # DNFs always finish last (P21/P22 etc.)

    return finish_order


def simulate_season(
    current_standings: pd.DataFrame,
    remaining_races: int,
    driver_strengths: Dict[str, float],
    n_simulations: int = 10000,
    remaining_circuits: Optional[List[str]] = None,
    safety_car_multiplier: float = 1.0,
    dnf_rates: Optional[Dict[str, float]] = None,
    driver_constructors: Optional[Dict[str, str]] = None,
    noise_std: float = 0.15,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Monte Carlo championship simulation.

    `remaining_circuits`: the actual circuit name for each of the
    `remaining_races` races left this season (e.g. from the real schedule),
    in calendar order. Each simulated race draws its safety-car probability
    from the circuit it's actually run at, instead of every remaining race
    using one generic probability. If not given (or shorter than
    remaining_races), any race without a known circuit falls back to
    DEFAULT_SAFETY_CAR_PROBABILITY.

    `dnf_rates` / `driver_constructors`: optional per-driver DNF modeling —
    see simulate_race() above. If omitted, no DNFs are simulated (matches
    the original behavior before this feature existed).
    """
    rng = np.random.default_rng(seed)

    drivers = current_standings["driver"].tolist()
    base_points = dict(zip(
        current_standings["driver"],
        current_standings["points"]
    ))

    circuits_for_races = list(remaining_circuits) if remaining_circuits else []
    # Pad with None (-> default probability) if we weren't given enough circuits
    while len(circuits_for_races) < remaining_races:
        circuits_for_races.append(None)

    win_counts = {d: 0 for d in drivers}
    final_points_sum = {d: 0.0 for d in drivers}
    safety_car_count = 0

    for _ in range(n_simulations):
        season_points = base_points.copy()
        for race_num in range(remaining_races):
            circuit = circuits_for_races[race_num]
            finish_order = simulate_race(
                driver_strengths,
                noise_std=noise_std,
                circuit_name=circuit,
                safety_car_multiplier=safety_car_multiplier,
                dnf_rates=dnf_rates,
                driver_constructors=driver_constructors,
                rng=rng,
            )
            for pos, driver in enumerate(finish_order, 1):
                if driver in season_points:
                    pts = POINTS_SYSTEM.get(pos, 0)
                    season_points[driver] = season_points.get(driver, 0) + pts
                    if pos == 1:
                        season_points[driver] += 1

        champion = max(season_points, key=season_points.get)
        win_counts[champion] += 1
        for d in drivers:
            final_points_sum[d] += season_points.get(d, 0)

    results = []
    for driver in drivers:
        results.append({
            "driver": driver,
            "wdc_probability": round(win_counts[driver] / n_simulations * 100, 1),
            "avg_final_points": round(final_points_sum[driver] / n_simulations, 1),
            "current_points": base_points.get(driver, 0),
        })

    return pd.DataFrame(results).sort_values(
        "wdc_probability", ascending=False
    ).reset_index(drop=True)

def build_driver_strengths(standings: pd.DataFrame,
                           model_podium_probs: Optional[Dict[str, float]] = None,
                           model_blend_weight: float = 0.5) -> Dict[str, float]:
    """
    Driver "strength" input for simulate_season()/simulate_race().

    Previously this was ONLY a linear rescale of current championship
    points (0.3 + 0.7 * points/max_points) — meaning the season simulator
    had no connection at all to the trained race podium classifier, even
    though that model exists and captures signal points don't (e.g. a
    driver on a hot streak after a slow start won't show it in points yet,
    since points lag current form). See RaceMindAI_Audit_Phases1-5.md
    Phase 2.3 finding, RaceMindAI_Redesign_Phases6-7.md §6.5.

    `model_podium_probs`: optional {driver: podium_probability} from the
    trained race model (see main.py's `_compute_model_driver_strengths()`,
    which builds this from each driver's most recent feature snapshot —
    itself an approximation, since the simulator runs many different
    future races rather than one specific circuit/grid; see that
    function's docstring for the caveat). When given, each driver's final
    strength is a blend of the points-based value and the model's value;
    when omitted (or a driver has no model prediction available — e.g. a
    brand-new rookie with no feature history), behavior is UNCHANGED from
    before — pure points-based strength, so this is backward-compatible
    with any caller not yet passing model_podium_probs.

    `model_blend_weight`: 0.0 = ignore the model entirely (old behavior),
    1.0 = use only the model's podium probability, 0.5 = equal blend
    (default). Deliberately a simple linear blend rather than something
    more elaborate — both signals are themselves approximations (points
    lag form; the model's per-driver number here isn't circuit-specific),
    so a transparent blend is more honest than a falsely precise
    combination formula.
    """
    max_pts = standings["points"].max()
    if max_pts == 0:
        points_strength = {d: 0.5 for d in standings["driver"]}
    else:
        points_strength = {}
        for _, row in standings.iterrows():
            base = row["points"] / max_pts
            points_strength[row["driver"]] = round(0.3 + 0.7 * base, 4)

    if not model_podium_probs:
        return points_strength

    blended = {}
    for driver, pts_strength in points_strength.items():
        model_component = model_podium_probs.get(driver)
        if model_component is None:
            blended[driver] = pts_strength  # no model signal for this driver — fall back unchanged
        else:
            # model_component is a 0-1 podium probability; points_strength
            # lives on the 0.3-1.0 range set above. Rescale the model
            # component onto the same range before blending — otherwise
            # this would be averaging two differently-scaled numbers and
            # calling it a strength, which isn't meaningful.
            model_component_scaled = 0.3 + 0.7 * model_component
            blended[driver] = round(
                model_blend_weight * model_component_scaled + (1 - model_blend_weight) * pts_strength, 4
            )
    return blended


def build_driver_dnf_rates(historical_results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes each driver's overall DNF rate from historical race results
    (the same get_cached_historical_results() output used to train the race
    predictor), as a simple lifetime average — status NOT containing
    "Finished" or "Lap" counts as a DNF, matching the convention already
    used in feature_engineering.py's constructor_dnf_rate.

    This is intentionally simpler than a rolling/recency-weighted rate
    (which is what feature_engineering.py computes per-constructor for the
    podium model) — the season simulator only needs a single representative
    number per driver, not a time series, since every simulated remaining
    race uses the same rate.
    """
    df = historical_results_df.copy()
    df["dnf"] = (~df["status"].astype(str).str.contains("Finished|Lap", na=False)).astype(int)
    rates = df.groupby("driver")["dnf"].mean()
    return rates.to_dict()
