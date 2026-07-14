#!/usr/bin/env python3

import argparse
import sys
import os
import pandas as pd

# Make `app.*` imports work when this script is run from the repo root
# (python scripts/retrain_model.py) or from anywhere else.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from app.data.ergast_client import get_cached_historical_results, get_season_schedule
from app.models.race_predictor import train_model, load_model_metadata
from app.models.weather_features import build_weather_lookup


def main():
    parser = argparse.ArgumentParser(description="Retrain the RaceMindAI race predictor model.")
    parser.add_argument("--year-start", type=int, default=2022,
                        help="First season to train on (default: 2022)")
    parser.add_argument("--year-end", type=int, default=2026,
                        help="Last season to train on (default: 2026)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Ignore the local cache and re-fetch every season from the API")
    parser.add_argument("--no-era-weighting", action="store_true",
                        help="Disable regulation-era sample weighting (train with equal weights)")
    args = parser.parse_args()

    print(f"=== RaceMindAI model retrain — {args.year_start}-{args.year_end} ===")

    prev_meta = load_model_metadata()
    if prev_meta:
        print(f"Previous model: trained {prev_meta['trained_at']}, "
              f"accuracy {prev_meta['accuracy']:.1%}, "
              f"years {prev_meta['years_trained_on']}")
    else:
        print("No previous model metadata found (first run, or pre-metadata model).")

    print(f"\nFetching training data (force_refresh={args.force_refresh})...")
    df = get_cached_historical_results(args.year_start, args.year_end,
                                       force_refresh=args.force_refresh)

    if df.empty:
        print("ERROR: no training data fetched — aborting without touching the saved model.")
        sys.exit(1)

    print(f"Fetched {len(df)} result rows across years {sorted(df['year'].unique().tolist())}")

    # Weather backfill — see RaceMindAI_Redesign_Phases6-7.md §6.4/§7.5
    # step 3. Without this, train_model() silently falls back to filling
    # every weather column with a constant default (no real signal at
    # all) — see train_model()'s own "no weather_lookup provided" print.
    # A model trained via THIS script (as opposed to hitting
    # /api/predictor/train, which already does this) would otherwise ship
    # with dead weather features despite the columns existing, which
    # defeats the point of the whole feature for this code path. Soft-
    # fails to weather_lookup=None (matching train_model()'s own fallback)
    # rather than aborting the retrain if the weather API is unreachable —
    # a retrain with stale/no weather signal is better than no retrain.
    print(f"\nBuilding weather lookup for {args.year_start}-{args.year_end}...")
    try:
        schedules = pd.concat(
            [get_season_schedule(y).assign(year=y) for y in range(args.year_start, args.year_end + 1)],
            ignore_index=True,
        )
        weather_lookup = build_weather_lookup(schedules)
        print(f"Weather lookup built: {len(weather_lookup)} race-days covered.")
    except Exception as e:
        print(f"WARNING: weather backfill failed ({e}) — training without "
              f"real weather signal (defaults will be used instead).")
        weather_lookup = None

    print(f"\nTraining model (era_weighting={not args.no_era_weighting})...")
    train_model(df, use_era_weighting=not args.no_era_weighting, weather_lookup=weather_lookup)

    new_meta = load_model_metadata()
    print(f"\n=== Done ===")
    print(f"New model: trained {new_meta['trained_at']}, "
          f"accuracy {new_meta['accuracy']:.1%}, "
          f"{new_meta['rows_trained_on']} rows, "
          f"years {new_meta['years_trained_on']}")
    if new_meta.get("calibrated"):
        print(f"Calibrated: Brier score {new_meta['brier_score_raw']:.4f} (raw) "
              f"-> {new_meta['brier_score_calibrated']:.4f} (calibrated)")

    prev_split_method = prev_meta.get("split_method") if prev_meta else None
    new_split_method = new_meta.get("split_method")
    if prev_meta and prev_split_method != new_split_method:
        # The validation methodology itself changed (e.g. this retrain is
        # the first one after switching from a random row-level split to
        # a chronological one — see RaceMindAI_Audit_Phases1-5.md Phase
        # 6.3). Comparing accuracy across that change is comparing two
        # different measurements, not tracking real model drift — a lower
        # number here is very likely the leaky old split being replaced by
        # a stricter, more honest one, not a regression. Skip the
        # numeric-drop check entirely rather than firing a misleading
        # WARNING (and a misleading non-zero exit code) on a change that
        # was expected and desired.
        print(f"\nNOTE: previous model's split_method was "
              f"{prev_split_method!r}, this one's is {new_split_method!r} — "
              f"skipping the accuracy-regression check below, since "
              f"comparing accuracy across a validation-methodology change "
              f"isn't a like-for-like comparison. See "
              f"RaceMindAI_Audit_Phases1-5.md Phase 6.3 / Phase 7.6.")
    elif prev_meta and new_meta["accuracy"] < prev_meta["accuracy"] - 0.03:
        print(f"\nWARNING: accuracy dropped more than 3 points "
              f"({prev_meta['accuracy']:.1%} -> {new_meta['accuracy']:.1%}). "
              f"Consider reviewing before this model goes live.")
        sys.exit(2)  # non-zero exit so a CI workflow can flag this without failing the commit


if __name__ == "__main__":
    main()
