import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.ergast_client import get_historical_results, get_driver_standings
from app.models.race_predictor import train_model, predict_race, get_feature_importance
from app.models.season_simulator import simulate_season, build_driver_strengths

def test_full_pipeline():
    print("Fetching 2021-2023 historical results for training...")
    df = get_historical_results(2021, 2023)
    print(f"  Loaded {len(df)} race entries across {df['year'].nunique()} seasons")

    print("\nTraining race predictor model...")
    model = train_model(df)

    print("\nFeature importances:")
    fi = get_feature_importance(model)
    print(fi.to_string())

    print("\nRunning season simulation (2024)...")
    standings = get_driver_standings(2024, round_num=10)
    strengths = build_driver_strengths(standings)
    print(f"  Driver strengths: {strengths}")

    sim = simulate_season(
        current_standings=standings,
        remaining_races=14,
        driver_strengths=strengths,
        n_simulations=5000,
    )
    print("\nWDC Probability after Round 10, 2024:")
    print(sim[["driver","wdc_probability","avg_final_points","current_points"]]
          .head(8).to_string())

if __name__ == "__main__":
    test_full_pipeline()