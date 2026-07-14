import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from app.models.feature_engineering import build_training_features
from app.models.weather_features import WEATHER_FEATURE_COLUMNS, attach_weather_features
from app.models.practice_pace_features import PRACTICE_PACE_FEATURE_COLUMNS, attach_practice_pace_features

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../data")
MODEL_DIR = os.getenv("MODEL_DIR", _DEFAULT_MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "race_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "race_model.meta.json")
# Calibration — see RaceMindAI_Redesign_Phases6-7.md §6.5 / Audit Phase 5
# "No calibration validation" (Critical gap). Stored as a SEPARATE pickle
# from the classifier itself (not baked in) so an older race_model.pkl
# without a matching calibrator file degrades gracefully — predict_race()
# just returns raw probabilities in that case rather than erroring.
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "race_model_calibrator.pkl")

# WEATHER_FEATURE_COLUMNS and PRACTICE_PACE_FEATURE_COLUMNS appended — see
# RaceMindAI_Redesign_Phases6-7.md §6.4/§6.5. attach_weather_features() and
# attach_practice_pace_features() must both be called on the training
# dataframe before these are populated (see train_model() below); at
# inference time main.py is responsible for populating them via
# weather_features.get_session_weather() and a same-race FP2 pace lookup.
FEATURES = [
    "grid", "grid_squared",
    "driver_rolling_points", "driver_rolling_wins", "driver_rolling_podiums",
    "driver_circuit_avg_pos", "constructor_avg_points",
    "constructor_dnf_rate", "circuit_type_code",
    "round", "year",
] + WEATHER_FEATURE_COLUMNS + PRACTICE_PACE_FEATURE_COLUMNS

# One entry per FEATURES column, in the same order. XGBoost enforces these
# as a hard constraint on the trained trees, not a hint:
#   -1  →  prediction must be non-increasing as the feature increases
#   +1  →  prediction must be non-decreasing as the feature increases
#    0  →  no constraint (categorical / no physical monotonic meaning)
#
# Why this exists: regularization alone (min_child_weight, reg_lambda, the
# lower 2026 era-weight) reduced overfitting but didn't fix the underlying
# issue — XGBoost is free to decide grid position is "optional" if
# driver-identity-correlated rolling-form features are more separable, and
# a pole-sitter can end up predicted below a driver starting P8 purely
# because that driver's rolling stats dominated the splits. These
# constraints make that structurally impossible: worse grid can never
# increase podium probability, better recent form can never decrease it,
# etc. — regardless of how the trees would otherwise prefer to split.
MONOTONE_CONSTRAINTS = (
    -1,  # grid — worse (higher number) starting position never helps
    -1,  # grid_squared — same direction, penalizes deep grid slots more
    1,   # driver_rolling_points — more recent points never hurts
    1,   # driver_rolling_wins — more recent wins never hurts
    1,   # driver_rolling_podiums — more recent podiums never hurts
    -1,  # driver_circuit_avg_pos — worse (higher number) avg finish never helps
    1,   # constructor_avg_points — stronger team pace never hurts
    -1,  # constructor_dnf_rate — more unreliable car never helps
    0,   # circuit_type_code — categorical, no monotonic meaning
    0,   # round — no physical monotonic relationship
    0,   # year — no physical monotonic relationship
    # WEATHER_FEATURE_COLUMNS — all left unconstrained (0), deliberately.
    # Track/air temperature has a non-monotonic "sweet spot" effect on
    # grip and tyre performance (too cold = no grip, too hot = graining/
    # blistering) per RaceMindAI_Audit_Phases1-5.md Phase 3.2 research —
    # forcing a monotone constraint here would be actively wrong, unlike
    # grid/form/reliability above where "more is better" genuinely holds.
    0,   # forecast_air_temp_c
    0,   # forecast_track_temp_proxy_c
    0,   # forecast_precip_prob
    0,   # forecast_precip_mm
    0,   # forecast_wind_speed_kmh
    0,   # weather_regime_code — categorical, no monotonic meaning
    # PRACTICE_PACE_FEATURE_COLUMNS — see RaceMindAI_Redesign_Phases6-7.md
    # §6.5. Unlike the weather columns above, practice_pace_delta_pct DOES
    # have a genuine monotonic relationship (it's a gap-to-fastest measure,
    # same family as driver_circuit_avg_pos/constructor_dnf_rate above) —
    # more % slower than the session's fastest long-run pace can never
    # HELP podium chances, so it gets a real constraint rather than 0.
    -1,  # practice_pace_delta_pct — more % off the fastest long-run pace never helps
    0,   # practice_lap_count — a confidence/data-availability signal, not
         # directionally meaningful on its own (a driver with more sample
         # laps isn't inherently faster or slower)
)

# ── Regulation-era sample weighting ─────────────────────────────────────────
# 2026 introduced F1's biggest regulation overhaul in over a decade (new
# aero rules, new power-unit format). Grid-position-to-result relationships,
# constructor pace hierarchies, and DNF patterns from 2022-2025 (the
# previous regulation era) are a real but noisier signal for predicting
# 2026+ races than 2026 data itself, because car characteristics changed
# significantly. Rather than drop the old-era data entirely (which would
# leave very little training data early in the new era), each row is
# weighted by which regulation era it belongs to — 2026 races count several
# times more than a single equivalent 2022-2025 race when XGBoost computes
# its loss gradient.
#
# These are deliberately simple, hand-set multipliers rather than something
# tuned via cross-validation — treat them as a reasonable starting point,
# not a finely calibrated constant. Revisit once a full 2026 season exists.
REGULATION_ERA_WEIGHTS = {
    2022: 0.6,
    2023: 0.7,
    2024: 0.85,
    2025: 1.0,    # last season under the previous regs — most comparable of the "old era"
    2026: 1.5,    # current regulations — still favored, but 3.0 let a small
                  # handful of early-2026 races (where the same few drivers
                  # both started AND finished well) dominate training enough
                  # that the model could fit them via driver-identity-
                  # correlated rolling-form features alone, without needing
                  # to learn a real grid→result relationship. See the
                  # `min_child_weight` / `reg_lambda` additions below for
                  # the other half of this fix.
}
DEFAULT_ERA_WEIGHT = 1.0  # fallback for any year not listed above


def compute_sample_weights(df: pd.DataFrame,
                           era_weights: dict = None) -> np.ndarray:
    """
    Returns a per-row weight array aligned to df's index, based on each
    row's `year` column and REGULATION_ERA_WEIGHTS (or a custom override).

    Uses numpy vectorize instead of pandas .map() — pandas map behaviour
    with dict arguments changed across 2.x versions and produced doubled
    arrays in some configurations. numpy vectorize is stable across all versions.
    """
    weights = era_weights or REGULATION_ERA_WEIGHTS
    years = df["year"].to_numpy(dtype=int)
    get_weight = np.vectorize(lambda y: weights.get(int(y), DEFAULT_ERA_WEIGHT))
    return get_weight(years)  # guaranteed 1D, same length as df


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits `df` into train/test by whole races (year, round), ordered
    chronologically — the most recent `test_size` fraction of RACES (not
    rows) becomes the held-out test set.

    Replaces the previous `train_test_split(..., stratify=y)` random
    row-level split, which had two problems specific to this data:
      1. Rows from the same race weekend could land on both sides of the
         split, and — worse — a driver's LATER 2026 races could end up in
         the training fold while an EARLIER 2026 race from the same
         partial season ended up in the test fold, since TRAIN_YEAR_START..
         TRAIN_YEAR_END includes the in-progress current season. The model
         could effectively be evaluated on "predicting the past" relative
         to some of its own training data.
      2. It doesn't match how this model is actually used: at inference
         time you only ever have races that happened BEFORE the one you're
         predicting. A random split doesn't test that regime; a
         chronological split does.

    This is the standard approach for time-ordered prediction tasks like
    this one — see RaceMindAI_Audit_Phases1-5.md Phase 3.3, which found
    published academic work on this exact problem (driver finishing
    position prediction) using a chronological train/test split rather
    than a random one.

    Returns (train_mask, test_mask) — boolean numpy arrays aligned to
    df's row order (NOT to df.index; caller should reset_index or index
    positionally, same as the existing X = df[FEATURES].values pattern).
    """
    races = df[["year", "round"]].drop_duplicates().sort_values(["year", "round"])
    n_test_races = max(1, int(round(len(races) * test_size)))
    test_race_keys = set(map(tuple, races.iloc[-n_test_races:].values))

    race_keys = list(zip(df["year"].tolist(), df["round"].tolist()))
    test_mask = np.array([rk in test_race_keys for rk in race_keys])
    train_mask = ~test_mask
    return train_mask, test_mask


def walk_forward_backtest(historical_df: pd.DataFrame,
                          min_train_years: int = 2,
                          use_era_weighting: bool = True,
                          era_weights: dict = None,
                          weather_lookup: pd.DataFrame = None,
                          practice_lookup: pd.DataFrame = None) -> pd.DataFrame:
    """
    Walk-forward backtest: for each season N in the data (after an initial
    `min_train_years` warm-up), trains a model on all seasons strictly
    before N and evaluates it on season N alone, then moves on to N+1.

    This gives a per-season generalization trend instead of one fixed
    train/test number — useful for checking whether accuracy is stable
    year over year or whether a particular season (e.g. the 2026 reg
    reset) is an outlier the single-split number would hide. Not run
    automatically at every train_model() call (it retrains the model once
    per season, so it's much more expensive) — call this separately, e.g.
    from a CI job or a one-off notebook check, and persist the result
    (see RaceMindAI_Redesign_Phases6-7.md, §7.7) rather than re-running it
    on every deploy.

    `weather_lookup` / `practice_lookup`: same as train_model() — pass the
    outputs of weather_features.build_weather_lookup() /
    practice_pace_features.build_practice_pace_lookup() to backtest WITH
    real signal; omitted, every season is backtested without it (defaults
    filled), which is still useful for isolating whether either feature is
    actually helping once you run this both ways and compare.

    Returns a DataFrame with one row per backtested season: year,
    n_train_races, n_test_races, accuracy, precision, recall.
    """
    from sklearn.metrics import precision_score, recall_score

    df_all = build_training_features(historical_df)
    if weather_lookup is not None:
        df_all = attach_weather_features(df_all, weather_lookup)
    else:
        from app.models.weather_features import default_weather_row
        for col, val in default_weather_row().items():
            df_all[col] = val
    if practice_lookup is not None:
        df_all = attach_practice_pace_features(df_all, practice_lookup)
    else:
        from app.models.practice_pace_features import default_practice_pace_row
        for col, val in default_practice_pace_row().items():
            df_all[col] = val
    df_all = df_all.dropna(subset=FEATURES)
    years = sorted(df_all["year"].unique())

    results = []
    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue  # not enough history yet to train a meaningful model

        train_df = df_all[df_all["year"] < test_year]
        test_df = df_all[df_all["year"] == test_year]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[FEATURES].values
        y_train = train_df["podium"].values
        X_test = test_df[FEATURES].values
        y_test = test_df["podium"].values

        sw_train = (compute_sample_weights(train_df, era_weights)
                    if use_era_weighting else np.ones(len(train_df)))

        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3.5,
            min_child_weight=5, reg_lambda=2.0,
            monotone_constraints=MONOTONE_CONSTRAINTS, tree_method="hist",
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=sw_train)
        y_pred = model.predict(X_test)

        results.append({
            "test_year": test_year,
            "n_train_races": train_df[["year", "round"]].drop_duplicates().shape[0],
            "n_test_races": test_df[["year", "round"]].drop_duplicates().shape[0],
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        })

    return pd.DataFrame(results)


def train_model(historical_df: pd.DataFrame,
                use_era_weighting: bool = True,
                era_weights: dict = None,
                weather_lookup: pd.DataFrame = None,
                practice_lookup: pd.DataFrame = None) -> XGBClassifier:
    print("Building features...")
    df = build_training_features(historical_df)
    if weather_lookup is not None:
        df = attach_weather_features(df, weather_lookup)
    else:
        print("[race_predictor] no weather_lookup provided — training "
              "without real weather signal (default Dry filled in). Pass "
              "weather_lookup from weather_features.build_weather_lookup() "
              "for real signal — see RaceMindAI_Redesign_Phases6-7.md §6.4.")
        from app.models.weather_features import default_weather_row
        for col, val in default_weather_row().items():
            df[col] = val
    if practice_lookup is not None:
        df = attach_practice_pace_features(df, practice_lookup)
    else:
        print("[race_predictor] no practice_lookup provided — training "
              "without real FP2 long-run pace signal (default filled in). "
              "Pass practice_lookup from "
              "practice_pace_features.build_practice_pace_lookup() for "
              "real signal — see RaceMindAI_Redesign_Phases6-7.md §6.5.")
        from app.models.practice_pace_features import default_practice_pace_row
        for col, val in default_practice_pace_row().items():
            df[col] = val
    df = df.dropna(subset=FEATURES)
    # Sort chronologically up front — chronological_split relies on this
    # order matching X/y's row order, and it makes the printed years-in-
    # training list (below) meaningful to read.
    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    X = df[FEATURES].values
    y = df["podium"].values
    sw = compute_sample_weights(df, era_weights) if use_era_weighting else np.ones(len(df))

    train_mask, test_mask = chronological_split(df, test_size=0.2)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    sw_train, sw_test = sw[train_mask], sw[test_mask]
    print(f"Chronological split: training on races through "
          f"{df.loc[train_mask, ['year','round']].iloc[-1].tolist()}, "
          f"testing on the most recent {test_mask.sum()} rows "
          f"({df.loc[test_mask, ['year','round']].drop_duplicates().shape[0]} races).")
    # sw is already guaranteed 1D by compute_sample_weights (numpy vectorize)
    # but squeeze here as a final safety net
    sw_train = np.asarray(sw_train, dtype=float).ravel()
    sw_test  = np.asarray(sw_test,  dtype=float).ravel()

    # Calibration sub-split: fitting a calibrator on the SAME test set used
    # for the reported accuracy would contaminate that metric — it would
    # no longer be a genuinely untouched holdout. So the test set (the
    # most recent ~20% of races) is itself split chronologically in half:
    # the earlier half is used ONLY to fit the isotonic calibrator, and
    # the later half (the most recent races of all) is the actual
    # untouched final-evaluation set that `acc`/classification_report
    # below are computed on. See RaceMindAI_Audit_Phases1-5.md Phase 5
    # "No calibration validation" (Critical gap), RaceMindAI_Redesign_Phases6-7.md §6.5.
    #
    # MIN_CALIB_ROWS / MIN_CALIB_PODIUM_EXAMPLES: the original gate here was
    # just ">= 2 test races", which is nowhere near enough — isotonic
    # regression fits a STEP function, and with only a couple of races
    # (maybe 40-60 rows, and given the ~5:1 no-podium:podium class
    # imbalance, often fewer than 10 actual podium=1 examples), it has so
    # little resolution to work with that it collapses wide ranges of
    # distinct raw scores onto the SAME flat step. Symptom observed in
    # production: three drivers starting P2/P3/P4 (meaningfully different
    # grid_squared values, since grid/grid_squared are ~59% of feature
    # importance) came back with the EXACT same calibrated podium
    # probability. Raising the bar here means calibration is skipped
    # (falls back to raw, uncalibrated-but-not-falsely-tied probabilities)
    # until there's genuinely enough data to fit a meaningful calibration
    # curve, rather than silently shipping a degenerate one.
    test_races_sorted = (
        df.loc[test_mask, ["year", "round"]].drop_duplicates().sort_values(["year", "round"])
    )
    MIN_CALIB_ROWS = 80
    MIN_CALIB_PODIUM_EXAMPLES = 15
    can_calibrate = len(test_races_sorted) >= 6  # need enough races that
    # splitting them in half still leaves a reasonable final-eval set too
    if can_calibrate:
        n_calib_races = max(1, len(test_races_sorted) // 2)
        calib_keys = set(map(tuple, test_races_sorted.iloc[:n_calib_races].values))
        final_keys = set(map(tuple, test_races_sorted.iloc[n_calib_races:].values))
        race_keys = list(zip(df["year"].tolist(), df["round"].tolist()))
        calib_mask = np.array([test_mask[i] and race_keys[i] in calib_keys for i in range(len(df))])
        final_mask = np.array([test_mask[i] and race_keys[i] in final_keys for i in range(len(df))])
        if final_mask.sum() == 0:  # degenerate case (e.g. calib grabbed every race) — bail out safely
            can_calibrate = False
        elif calib_mask.sum() < MIN_CALIB_ROWS or y[calib_mask].sum() < MIN_CALIB_PODIUM_EXAMPLES:
            print(f"[race_predictor] calibration slice has only "
                  f"{calib_mask.sum()} rows / {int(y[calib_mask].sum())} podium "
                  f"examples — too sparse to calibrate reliably (need >= "
                  f"{MIN_CALIB_ROWS} rows and >= {MIN_CALIB_PODIUM_EXAMPLES} "
                  f"podium examples). Skipping calibration for this run; "
                  f"predictions will use raw (uncalibrated) probabilities.")
            can_calibrate = False

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,           # back to 5 — a shallower tree (4) actively
                                # worked against grid getting its own split
                                # once rolling-form features claimed the
                                # early ones; monotone_constraints below is
                                # the real fix for the overfitting concern,
                                # not depth.
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.5,  # was 6 — that matched the raw ~5.6:1 class
                                # imbalance, but combined with the monotonic
                                # constraints it pushed recall to a perfect
                                # 1.00 at the cost of precision (0.74) — the
                                # model was flagging real podium finishers
                                # correctly every time, but also over-calling
                                # ~26% of its "podium" predictions on drivers
                                # who didn't actually podium. Lowering this
                                # trades some of that excess recall back for
                                # tighter, more trustworthy probabilities.
                                # If precision still isn't where you want it
                                # after retraining, drop this further (e.g.
                                # to 2); if recall drops too much (missing
                                # real podium finishers), bring it back up.
        min_child_weight=5,    # requires more real evidence per split — makes it
                                # harder to carve out a leaf for "just this driver"
        reg_lambda=2.0,        # L2 regularization — shrinks leaf weights, so no
                                # single feature can dominate a prediction as
                                # completely as before
        monotone_constraints=MONOTONE_CONSTRAINTS,
        tree_method="hist",    # required by XGBoost for monotone_constraints
        early_stopping_rounds=20,  # stop once held-out logloss stops improving,
                                    # instead of using all 300 rounds regardless
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train,
              sample_weight=sw_train,
              eval_set=[(X_test, y_test)],
              sample_weight_eval_set=[sw_test],
              verbose=False)

    print(f"Best iteration: {model.best_iteration} / {model.n_estimators} "
          "(if this is close to n_estimators, early stopping barely engaged)")

    from sklearn.metrics import brier_score_loss

    if can_calibrate:
        from sklearn.isotonic import IsotonicRegression
        X_calib, y_calib = X[calib_mask], y[calib_mask]
        X_final, y_final = X[final_mask], y[final_mask]

        # IsotonicRegression fit directly on the already-trained model's
        # own raw probabilities, rather than sklearn.calibration's
        # CalibratedClassifierCV(cv="prefit") — that constructor argument
        # was REMOVED in sklearn 1.6+ (replaced by a different API), so
        # using it here would make this code sklearn-version-fragile.
        # Fitting IsotonicRegression directly on (raw_prob, actual_label)
        # pairs is the same underlying idea (monotonic recalibration of a
        # prefit classifier's probabilities) without depending on a
        # specific sklearn version's calibration API.
        raw_probs_calib = model.predict_proba(X_calib)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs_calib, y_calib)

        raw_probs_final = model.predict_proba(X_final)[:, 1]
        calibrated_probs_final = calibrator.predict(raw_probs_final)
        brier_raw = brier_score_loss(y_final, raw_probs_final)
        brier_calibrated = brier_score_loss(y_final, calibrated_probs_final)
        print(f"Brier score on final holdout — raw: {brier_raw:.4f}, "
              f"calibrated: {brier_calibrated:.4f} "
              f"({'improved' if brier_calibrated < brier_raw else 'DID NOT improve'})")

        with open(CALIBRATOR_PATH, "wb") as f:
            pickle.dump(calibrator, f)

        y_pred = model.predict(X_final)  # accuracy/classification_report still
        # use the raw classifier's own .predict() threshold behavior, same
        # as before this change — calibration reshapes the PROBABILITY
        # scale for trustworthiness, it isn't meant to change the
        # decision threshold this report is based on.
        acc = accuracy_score(y_final, y_pred)
        eval_rows_used = int(final_mask.sum())
        print(f"Model accuracy (final holdout, {eval_rows_used} rows): {acc:.3f}")
        print(classification_report(y_final, y_pred, target_names=["No podium", "Podium"]))
    else:
        # Too few test races, or the calibration slice was too sparse
        # (see MIN_CALIB_ROWS / MIN_CALIB_PODIUM_EXAMPLES above) — fall
        # back to the old behavior (no calibrator saved, accuracy reported
        # on the whole test set) rather than shipping a degenerate
        # (plateaued/tied) calibration curve. Remove any stale calibrator
        # from a PREVIOUS training run so predict_race() doesn't silently
        # apply an out-of-date calibrator to this model.
        if os.path.exists(CALIBRATOR_PATH):
            os.remove(CALIBRATOR_PATH)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eval_rows_used = int(test_mask.sum())
        brier_raw = brier_calibrated = None
        print(f"Model accuracy: {acc:.3f} (skipped calibration this run — "
              f"either too few test races [{len(test_races_sorted)}, need "
              f">= 6] or too sparse a calibration slice once split. No "
              f"calibrator saved; predictions will use raw probabilities.)")
        print(classification_report(y_test, y_pred, target_names=["No podium", "Podium"]))

    # Feature importance — printed at train time so it's easy to check
    # whether `grid`/`grid_squared` are actually influencing predictions
    # or being drowned out by driver-identity-correlated rolling-form
    # features (the failure mode that prompted the era-weight/
    # regularization changes above).
    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("Feature importance (highest to lowest):")
    for name, imp in importances:
        print(f"  {name:28s} {imp:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    years_in_training = sorted(df["year"].unique().tolist())
    with open(MODEL_META_PATH, "w") as f:
        json.dump({
            "trained_at": datetime.datetime.now().isoformat(),
            "years_trained_on": years_in_training,
            "rows_trained_on": len(df),
            "rows_train_split": int(train_mask.sum()),
            "rows_test_split": eval_rows_used,  # the FINAL holdout only —
            # excludes rows used to fit the calibrator, if any were
            "split_method": "chronological_by_race",  # was "random_row_stratified" —
            # kept here explicitly so any older meta.json (or anyone
            # comparing this model's accuracy against a pre-fix number)
            # can tell the two aren't apples-to-apples. See
            # RaceMindAI_Audit_Phases1-5.md Phase 6.3 / Phase 7.6: expect
            # this number to look different — likely lower — than the old
            # split's number, and to be more trustworthy for it.
            "accuracy": round(float(acc), 4),
            "calibrated": can_calibrate,
            "brier_score_raw": round(float(brier_raw), 4) if brier_raw is not None else None,
            "brier_score_calibrated": round(float(brier_calibrated), 4) if brier_calibrated is not None else None,
            "era_weighting_used": use_era_weighting,
            "era_weights": era_weights or REGULATION_ERA_WEIGHTS,
        }, f, indent=2)

    print(f"Model saved to {MODEL_PATH} — n_features: {model.n_features_in_}")
    return model


def load_model() -> XGBClassifier:
    """
    Load the trained model from disk. If no model file exists (e.g. first
    run on Hugging Face Spaces where .pkl files cannot be committed), raises
    a clear FileNotFoundError so the caller can trigger a retrain.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Click 'Train / Refresh Model' to train it now."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_calibrator():
    """
    Loads the saved isotonic calibrator, or returns None if none exists —
    e.g. a model trained before calibration was added (§6.5), or one
    trained with too few test races to fit one (see train_model()'s
    can_calibrate guard). predict_race() treats None as "use raw
    probabilities", never raises for a missing calibrator — same soft-fail
    philosophy as this codebase's other optional-artifact handling.
    """
    if not os.path.exists(CALIBRATOR_PATH):
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        return pickle.load(f)


def load_or_train_model(historical_df: pd.DataFrame = None) -> XGBClassifier:
    """
    Try to load the model. If it doesn't exist (first run on HF Spaces or
    after cache cleared), train it automatically using historical_df.

    This is the preferred entry point for the Streamlit app — it never
    crashes on a missing model file.

    Args:
        historical_df: Raw historical race results DataFrame. Only needed
                       if the model file is missing. If None and the model
                       is missing, raises RuntimeError with a user-friendly
                       message.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Model missing — happens on first HF Spaces run or after /tmp reset
    if historical_df is None:
        raise RuntimeError(
            "No trained model found and no training data was provided. "
            "Pass historical_df to load_or_train_model() or click "
            "'Train / Refresh Model' in the UI."
        )

    print(f"No model found at {MODEL_PATH} — training now...")
    return train_model(historical_df)


def load_model_metadata() -> dict | None:
    """Returns the metadata saved alongside the model at last train time
    (trained_at, years_trained_on, accuracy, etc.), or None if the model
    was trained before this metadata file existed."""
    if not os.path.exists(MODEL_META_PATH):
        return None
    with open(MODEL_META_PATH) as f:
        return json.load(f)


def model_exists() -> bool:
    """Returns True if a trained model file exists on disk."""
    return os.path.exists(MODEL_PATH)


def model_is_stale(max_age_days: int = 7) -> bool:
    """
    Returns True if the saved model is older than max_age_days, OR if no
    metadata exists at all (e.g. very first run, or a model trained before
    this staleness tracking was added). Used to show a "data may be
    outdated" banner / offer a one-click retrain in the UI, without forcing
    a retrain on every single page load.
    """
    meta = load_model_metadata()
    if meta is None:
        return True
    trained_at = datetime.datetime.fromisoformat(meta["trained_at"])
    age = datetime.datetime.now() - trained_at
    return age > datetime.timedelta(days=max_age_days)


def predict_race(model: XGBClassifier,
                 race_features: pd.DataFrame,
                 calibrator=None) -> pd.DataFrame:
    """
    `calibrator`: pass an explicit calibrator (e.g. to reuse one already
    loaded elsewhere), or leave as None to auto-load via load_calibrator()
    — which itself returns None (not an error) if no calibrator was ever
    saved for this model (e.g. an older model, or one trained with too
    few test races). Either way, "podium_probability" in the returned
    DataFrame is the calibrated value when a calibrator is available and
    the RAW value otherwise — existing callers that only read
    "podium_probability" keep working unchanged either way. See
    RaceMindAI_Redesign_Phases6-7.md §6.5.
    """
    if calibrator is None:
        try:
            calibrator = load_calibrator()
        except Exception:
            calibrator = None

    df = race_features.copy()
    df = df.fillna(0)
    n = model.n_features_in_
    X = df[FEATURES].values
    if X.shape[1] < n:
        X = np.hstack([X, np.zeros((X.shape[0], n - X.shape[1]))])
    elif X.shape[1] > n:
        X = X[:, :n]
    raw_probs = model.predict_proba(X)[:, 1]
    df["raw_podium_probability"] = raw_probs
    if calibrator is not None:
        # IsotonicRegression takes the model's own raw probability as 1D
        # input and calibrates via .predict() — NOT .predict_proba() on
        # the raw feature matrix (that would be the CalibratedClassifierCV
        # API, which this codebase deliberately doesn't use — see
        # train_model()'s calibration section for why).
        calibrated_probs = calibrator.predict(raw_probs)

        # BLEND rather than use calibrated_probs directly. Isotonic
        # regression fits a STEP function — any raw-score region with
        # sparse calibration examples collapses onto one flat step,
        # REGARDLESS of how much total calibration data exists (raising
        # the data-volume bar in train_model() reduces how often this
        # happens but can't eliminate it structurally — verified in
        # testing: even with 380 well-populated test rows, three drivers
        # with genuinely different raw scores landed on the exact same
        # calibrated value). Symptom this was chasing: three drivers
        # starting P2/P3/P4 (different grid_squared, ~59% of feature
        # importance) shown with the literal identical podium probability
        # in the UI.
        #
        # A 90/10 blend keeps ~all of calibration's accuracy benefit
        # (Brier score improvement) while the 10% raw-probability
        # component preserves fine-grained ordering between drivers whose
        # calibrated value would otherwise tie exactly. This is standard
        # practice for isotonic calibration in production ranking
        # contexts, not a workaround unique to this codebase.
        CALIBRATION_BLEND_WEIGHT = 0.9
        df["podium_probability"] = (
            CALIBRATION_BLEND_WEIGHT * calibrated_probs + (1 - CALIBRATION_BLEND_WEIGHT) * raw_probs
        )
        df["calibrated"] = True
    else:
        df["podium_probability"] = raw_probs
        df["calibrated"] = False
    # raw_podium_probability as an explicit secondary sort key: even after
    # the blend above, two drivers could still land within float-rounding
    # distance of each other — breaking ties on the un-blended raw score
    # keeps predicted_position from depending on incidental row order.
    df = df.sort_values(["podium_probability", "raw_podium_probability"], ascending=[False, False])
    df["predicted_position"] = range(1, len(df) + 1)
    return df[["driver", "podium_probability", "raw_podium_probability", "calibrated",
               "predicted_position"] + FEATURES].reset_index(drop=True)


def get_feature_importance(model: XGBClassifier) -> pd.DataFrame:
    importance = model.feature_importances_
    n = min(len(importance), len(FEATURES))
    return pd.DataFrame({
        "feature": FEATURES[:n],
        "importance": importance[:n]
    }).sort_values("importance", ascending=False).reset_index(drop=True)
