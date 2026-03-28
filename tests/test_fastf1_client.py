import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.fastf1_client import get_lap_times, get_race_results

def test_lap_times():
    print("Fetching 2024 Bahrain lap times (this may take a minute first time)...")
    df = get_lap_times(2024, "Bahrain", "R")
    assert len(df) > 0
    assert "LapTimeSeconds" in df.columns
    print(f"  Laps fetched: {len(df)}")
    print(f"  Drivers: {sorted(df['Driver'].unique())}")
    print(df[["Driver","LapNumber","LapTimeSeconds","Compound","TyreLife"]].head(5).to_string())

def test_race_results():
    print("\nFetching 2024 Bahrain race results...")
    df = get_race_results(2024, "Bahrain")
    assert len(df) > 0
    print(f"  Results: {len(df)} drivers")
    print(df[["Abbreviation","TeamName","Position","GridPosition","Points"]].to_string())

if __name__ == "__main__":
    print("Testing FastF1 client...\n")
    test_lap_times()
    test_race_results()
    print("\nAll FastF1 tests passed!")
