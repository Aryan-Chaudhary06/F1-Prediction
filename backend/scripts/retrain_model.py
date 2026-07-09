#!/usr/bin/env python3

import argparse
import sys
import os

# Make `app.*` imports work when this script is run from the repo root
# (python scripts/retrain_model.py) or from anywhere else.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from app.data.ergast_client import get_cached_historical_results
from app.models.race_predictor import train_model, load_model_metadata


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

    print(f"\nTraining model (era_weighting={not args.no_era_weighting})...")
    train_model(df, use_era_weighting=not args.no_era_weighting)

    new_meta = load_model_metadata()
    print(f"\n=== Done ===")
    print(f"New model: trained {new_meta['trained_at']}, "
          f"accuracy {new_meta['accuracy']:.1%}, "
          f"{new_meta['rows_trained_on']} rows, "
          f"years {new_meta['years_trained_on']}")

    if prev_meta and new_meta["accuracy"] < prev_meta["accuracy"] - 0.03:
        print(f"\nWARNING: accuracy dropped more than 3 points "
              f"({prev_meta['accuracy']:.1%} -> {new_meta['accuracy']:.1%}). "
              f"Consider reviewing before this model goes live.")
        sys.exit(2)  # non-zero exit so a CI workflow can flag this without failing the commit


if __name__ == "__main__":
    main()
