import pandas as pd
from app.data.ergast_client import (
    get_driver_standings, get_constructor_standings,
    get_season_schedule, get_historical_results,
    get_qualifying_results
)
from app.data.fastf1_client import get_lap_times, get_race_results

def tool_get_driver_standings(year: int = 2026) -> str:
    """Get current F1 driver championship standings."""
    try:
        df = get_driver_standings(year)
        lines = [f"P{r['position']} {r['full_name']} ({r['constructor']}): {int(r['points'])} pts, {int(r['wins'])} wins"
                 for _, r in df.iterrows()]
        return f"{year} Driver Standings:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def tool_get_constructor_standings(year: int = 2026) -> str:
    """Get current F1 constructor championship standings."""
    try:
        df = get_constructor_standings(year)
        lines = [f"P{r['position']} {r['constructor']}: {int(r['points'])} pts"
                 for _, r in df.iterrows()]
        return f"{year} Constructor Standings:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def tool_get_race_results(year: int, gp: str) -> str:
    """Get race results for a specific Grand Prix."""
    try:
        df = get_race_results(year, gp)
        lines = [f"P{int(r['Position'])} {r['Abbreviation']} ({r['TeamName']}): {int(r['Points'])} pts — {r['Status']}"
                 for _, r in df.iterrows()]
        return f"{year} {gp} GP Results:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error fetching {gp} {year}: {e}"

def tool_get_lap_times(year: int, gp: str) -> str:
    """Get lap time summary for a specific Grand Prix."""
    try:
        df = get_lap_times(year, gp, "R")
        fastest = df.loc[df["LapTimeSeconds"].idxmin()]
        avg_per_driver = df.groupby("Driver")["LapTimeSeconds"].mean().sort_values()
        lines = [f"{d}: avg {t:.3f}s" for d, t in avg_per_driver.head(10).items()]
        return (f"{year} {gp} GP Lap Times:\n"
                f"Fastest: {fastest['Driver']} — {fastest['LapTimeSeconds']:.3f}s (Lap {int(fastest['LapNumber'])})\n"
                f"Average pace:\n" + "\n".join(lines))
    except Exception as e:
        return f"Error fetching lap times for {gp} {year}: {e}"

def tool_get_season_schedule(year: int = 2026) -> str:
    """Get the full season schedule."""
    try:
        df = get_season_schedule(year)
        lines = [f"R{int(r['round'])} {r['gp_name']} — {r['date']} ({r['country']})"
                 for _, r in df.iterrows()]
        return f"{year} Season Schedule:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def tool_compare_drivers(driver1: str, driver2: str, year: int = 2026) -> str:
    """Compare two drivers' performance in a given season."""
    try:
        df = get_driver_standings(year)
        d1 = df[df["driver"] == driver1]
        d2 = df[df["driver"] == driver2]
        if d1.empty or d2.empty:
            return f"Could not find both drivers in {year} standings."
        r1, r2 = d1.iloc[0], d2.iloc[0]
        return (f"Driver comparison {year}:\n"
                f"{driver1}: P{int(r1['position'])} — {int(r1['points'])} pts, {int(r1['wins'])} wins ({r1['constructor']})\n"
                f"{driver2}: P{int(r2['position'])} — {int(r2['points'])} pts, {int(r2['wins'])} wins ({r2['constructor']})\n"
                f"Points gap: {abs(int(r1['points']) - int(r2['points']))} pts")
    except Exception as e:
        return f"Error: {e}"

# Tool definitions for Gemini function calling
TOOL_DEFINITIONS = [
    {
        "name": "get_driver_standings",
        "description": "Get the current F1 driver championship standings for a given year",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year (e.g. 2026)"}
            },
            "required": []
        }
    },
    {
        "name": "get_constructor_standings",
        "description": "Get the current F1 constructor/team championship standings",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year (e.g. 2026)"}
            },
            "required": []
        }
    },
    {
        "name": "get_race_results",
        "description": "Get the race results for a specific Grand Prix",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "gp":   {"type": "string",  "description": "GP name e.g. 'Australia', 'Bahrain', 'Monaco'"}
            },
            "required": ["year", "gp"]
        }
    },
    {
        "name": "get_lap_times",
        "description": "Get lap time data and pace analysis for a specific Grand Prix",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "gp":   {"type": "string",  "description": "GP name e.g. 'Australia', 'Bahrain'"}
            },
            "required": ["year", "gp"]
        }
    },
    {
        "name": "get_season_schedule",
        "description": "Get the full race calendar/schedule for a season",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"}
            },
            "required": []
        }
    },
    {
        "name": "compare_drivers",
        "description": "Compare two drivers' championship standings and performance",
        "parameters": {
            "type": "object",
            "properties": {
                "driver1": {"type": "string", "description": "First driver code e.g. 'VER', 'NOR'"},
                "driver2": {"type": "string", "description": "Second driver code e.g. 'LEC', 'HAM'"},
                "year":    {"type": "integer", "description": "Season year"}
            },
            "required": ["driver1", "driver2"]
        }
    },
]

TOOL_FUNCTIONS = {
    "get_driver_standings":    lambda args: tool_get_driver_standings(args.get("year", 2026)),
    "get_constructor_standings": lambda args: tool_get_constructor_standings(args.get("year", 2026)),
    "get_race_results":        lambda args: tool_get_race_results(args["year"], args["gp"]),
    "get_lap_times":           lambda args: tool_get_lap_times(args["year"], args["gp"]),
    "get_season_schedule":     lambda args: tool_get_season_schedule(args.get("year", 2026)),
    "compare_drivers":         lambda args: tool_compare_drivers(args["driver1"], args["driver2"], args.get("year", 2026)),
}