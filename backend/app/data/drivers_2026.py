"""
A NOTE ON ACCURACY
───────────────────
Codes for long-tenured drivers (VER, HAM, LEC, NOR, ALO, etc.) are
the standard, stable Ergast codes and are safe to rely on.

Codes for 2025/2026 rookies and mid-career drivers who joined after
this module was written (Antonelli, Hadjar, Lindblad, Bortoleto,
Colapinto) are this module's BEST GUESS at the convention Ergast/
Jolpica uses (first 3 letters of surname, upper-cased), not a
confirmed value pulled from the live API. Ergast occasionally
deviates from that pattern (e.g. shared surnames get disambiguated).

Treat `ERGAST_CODE_CONFIRMED` below as the source of truth for which
codes are verified vs. guessed. Before relying on this in production,
hit `https://api.jolpi.ca/ergast/f1/2026/drivers.json?limit=100` once
and confirm/correct the guessed codes — `verify_codes_against_api()`
at the bottom of this file does exactly that and will print any
mismatches.
"""

import pandas as pd

DRIVERS_2026 = [
    {"name": "George Russell",    "code": "RUS", "team": "Mercedes",      "team_color": "#00D2BE", "number": 63},
    {"name": "Kimi Antonelli",    "code": "ANT", "team": "Mercedes",      "team_color": "#00D2BE", "number": 12},
    {"name": "Charles Leclerc",   "code": "LEC", "team": "Ferrari",       "team_color": "#DC143C", "number": 16},
    {"name": "Lewis Hamilton",    "code": "HAM", "team": "Ferrari",       "team_color": "#DC143C", "number": 44},
    {"name": "Lando Norris",      "code": "NOR", "team": "McLaren",       "team_color": "#FF8000", "number": 4},
    {"name": "Oscar Piastri",     "code": "PIA", "team": "McLaren",       "team_color": "#FF8000", "number": 81},
    {"name": "Max Verstappen",    "code": "VER", "team": "Red Bull",      "team_color": "#3671C6", "number": 1},
    {"name": "Isack Hadjar",      "code": "HAD", "team": "Red Bull",      "team_color": "#3671C6", "number": 6},
    {"name": "Pierre Gasly",      "code": "GAS", "team": "Alpine",        "team_color": "#0090FF", "number": 10},
    {"name": "Franco Colapinto",  "code": "COL", "team": "Alpine",        "team_color": "#0090FF", "number": 43},
    {"name": "Liam Lawson",       "code": "LAW", "team": "Racing Bulls",  "team_color": "#6692FF", "number": 30},
    {"name": "Arvid Lindblad",    "code": "LIN", "team": "Racing Bulls",  "team_color": "#6692FF", "number": 41},
    {"name": "Esteban Ocon",      "code": "OCO", "team": "Haas",          "team_color": "#B6BABD", "number": 31},
    {"name": "Oliver Bearman",    "code": "BEA", "team": "Haas",          "team_color": "#B6BABD", "number": 87},
    {"name": "Carlos Sainz",      "code": "SAI", "team": "Williams",      "team_color": "#005AFF", "number": 55},
    {"name": "Alexander Albon",   "code": "ALB", "team": "Williams",      "team_color": "#005AFF", "number": 23},
    {"name": "Nico Hulkenberg",   "code": "HUL", "team": "Audi",          "team_color": "#C0C0C0", "number": 27},
    {"name": "Gabriel Bortoleto", "code": "BOR", "team": "Audi",          "team_color": "#C0C0C0", "number": 5},
    {"name": "Fernando Alonso",   "code": "ALO", "team": "Aston Martin",  "team_color": "#006F62", "number": 14},
    {"name": "Lance Stroll",      "code": "STR", "team": "Aston Martin",  "team_color": "#006F62", "number": 18},
    {"name": "Sergio Perez",      "code": "PER", "team": "Cadillac",      "team_color": "#9C8E55", "number": 11},
    {"name": "Valtteri Bottas",   "code": "BOT", "team": "Cadillac",      "team_color": "#9C8E55", "number": 77},
]

# Codes verified as stable, long-standing Ergast convention (safe to trust).
# Anything for a driver NOT in this set is a guess — see module docstring.
ERGAST_CODE_CONFIRMED = {
    "RUS", "LEC", "HAM", "NOR", "PIA", "VER", "GAS",
    "LAW", "OCO", "SAI", "ALB", "HUL", "ALO", "STR", "PER", "BOT",
}

# Drivers with no, or very little, prior F1 race history as of the
# 2026 season. Used to show a rookie badge and to trigger fallback
# logic in feature engineering (new-constructor / no-history handling).
ROOKIE_2026 = {"Kimi Antonelli", "Isack Hadjar", "Gabriel Bortoleto", "Arvid Lindblad"}

# Brand-new constructors with no historical pace/DNF data before 2026.
NEW_CONSTRUCTORS_2026 = {"Audi", "Cadillac"}

_BY_NAME = {d["name"]: d for d in DRIVERS_2026}
_BY_CODE = {d["code"]: d for d in DRIVERS_2026}


def driver_by_name(name: str) -> dict | None:
    return _BY_NAME.get(name)


def driver_by_code(code: str) -> dict | None:
    return _BY_CODE.get(code)


def name_to_code(name: str) -> str | None:
    d = _BY_NAME.get(name)
    return d["code"] if d else None


def code_to_name(code: str) -> str | None:
    d = _BY_CODE.get(code)
    return d["name"] if d else None


def as_dataframe() -> pd.DataFrame:
    """DRIVERS_2026 as a DataFrame, in default grid order (P1..P22)."""
    return pd.DataFrame(DRIVERS_2026)


def verify_codes_against_api(jolpica_drivers_json: dict) -> list[dict]:
    """
    Pass in the parsed JSON from
    https://api.jolpi.ca/ergast/f1/2026/drivers.json?limit=100
    (MRData.DriverTable.Drivers) to check our guessed codes against
    what the live API actually uses.

    Returns a list of mismatches: [{"name": ..., "guessed": ..., "actual": ...}]
    Empty list means everything matched.
    """
    actual_by_name = {}
    for d in jolpica_drivers_json["MRData"]["DriverTable"]["Drivers"]:
        full_name = f"{d['givenName']} {d['familyName']}"
        actual_by_name[full_name] = d.get("code")

    mismatches = []
    for d in DRIVERS_2026:
        actual = actual_by_name.get(d["name"])
        if actual and actual != d["code"]:
            mismatches.append({"name": d["name"], "guessed": d["code"], "actual": actual})
    return mismatches
