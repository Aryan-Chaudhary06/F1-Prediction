import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.ergast_client import (
    get_season_schedule,
    get_driver_standings,
    get_constructor_standings,
)

def test_season_schedule():
    df = get_season_schedule(2024)
    assert len(df) > 0
    assert "gp_name" in df.columns
    print(f"  Schedule: {len(df)} races found")
    print(df.head(3).to_string())

def test_driver_standings():
    df = get_driver_standings(2024)
    assert len(df) > 0
    assert "driver" in df.columns
    print(f"\n  Standings: {len(df)} drivers")
    print(df.head(5).to_string())

def test_constructor_standings():
    df = get_constructor_standings(2024)
    assert len(df) > 0
    print(f"\n  Constructors: {len(df)} teams")
    print(df.to_string())

if __name__ == "__main__":
    print("Testing Ergast client...\n")
    test_season_schedule()
    test_driver_standings()
    test_constructor_standings()
    print("\nAll Ergast tests passed!")
