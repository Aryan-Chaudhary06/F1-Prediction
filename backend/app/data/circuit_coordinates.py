"""
app/data/circuit_coordinates.py
────────────────────────────────
Latitude/longitude per circuit, keyed with the SAME circuit-name strings
used in app.models.feature_engineering.CIRCUIT_TYPE — one fewer place for
circuit names to drift out of sync (same rationale already used for
season_simulator.SAFETY_CAR_PROBABILITY_BY_CIRCUIT).

Needed because neither the Jolpica schedule (ergast_client.get_season_schedule)
nor FastF1 session objects reliably expose usable lat/lon for a weather-API
call — Jolpica's Circuit.Location DOES include lat/long, but it isn't
currently extracted by _parse_schedule()/get_season_schedule() (see
weather_client.py docstring for why this file exists as a static fallback
rather than relying on that).

Coordinates are approximate circuit-centroid values (adequate for a
city/region-level weather forecast — Open-Meteo's grid resolution doesn't
require pinpoint precision within a circuit). All are real, published
circuit locations, EXCEPT Madring (Madrid), which is new for 2026 and
marked explicitly below as approximate pending an authoritative source —
verify before relying on it for anything precision-sensitive.
"""

CIRCUIT_COORDINATES = {
    "Bahrain International Circuit":       (26.0325, 50.5106),
    "Jeddah Corniche Circuit":              (21.6319, 39.1044),
    "Albert Park Grand Prix Circuit":       (-37.8497, 144.9680),
    "Suzuka Circuit":                       (34.8431, 136.5410),
    "Shanghai International Circuit":       (31.3389, 121.2200),
    "Miami International Autodrome":        (25.9581, -80.2389),
    "Autodromo Enzo e Dino Ferrari":        (44.3439, 11.7167),   # Imola — off the 2026 calendar; kept for historical-row backfill
    "Circuit de Monaco":                    (43.7347, 7.4206),
    "Circuit de Barcelona-Catalunya":       (41.5700, 2.2611),
    "Circuit Gilles Villeneuve":            (45.5000, -73.5228),
    "Red Bull Ring":                        (47.2197, 14.7647),
    "Silverstone Circuit":                  (52.0786, -1.0169),
    "Hungaroring":                          (47.5789, 19.2486),
    "Circuit de Spa-Francorchamps":         (50.4372, 5.9714),
    "Circuit Zandvoort":                    (52.3888, 4.5409),
    "Autodromo Nazionale di Monza":         (45.6156, 9.2811),
    "Baku City Circuit":                    (40.3725, 49.8533),
    "Marina Bay Street Circuit":            (1.2914, 103.8640),
    "Circuit of the Americas":              (30.1328, -97.6411),
    "Autodromo Hermanos Rodriguez":         (19.4042, -99.0907),
    "Autodromo Jose Carlos Pace":           (-23.7036, -46.6997),
    "Las Vegas Strip Street Circuit":       (36.1147, -115.1728),
    "Lusail International Circuit":         (25.4900, 51.4542),
    "Yas Marina Circuit":                   (24.4672, 54.6031),
    # NEW for 2026 — approximate, unverified against an authoritative
    # source (flagged per RaceMindAI_Redesign_Phases6-7.md §6.4 step 1).
    # Replace with confirmed coordinates once available.
    "Madring":                              (40.4590, -3.6170),
    # Was missing entirely — see the matching CIRCUIT_TYPE entry in
    # f1_constants.py for why (hosted the 2022 French GP, within this
    # app's training window).
    "Circuit Paul Ricard":                  (43.2506, 5.7917),
}


def get_circuit_coordinates(circuit_name: str) -> tuple[float, float] | None:
    """
    Returns (lat, lon) for a circuit name, or None if truly not in the
    table (caller should fall back to climatology — see weather_client.py
    — rather than skip the weather features entirely).

    Routes through f1_constants.canonical_circuit_name() first, since raw
    Jolpica circuit-name strings have been observed NOT matching this
    dict's keys exactly — accented characters, extra words, or alternate
    spellings for the same physical circuit (e.g. "Autódromo Hermanos
    Rodríguez" vs "Autodromo Hermanos Rodriguez", "Losail International
    Circuit" vs "Lusail International Circuit"). Without this, every one
    of those variants silently missed and fell all the way through to the
    generic climatology default instead of using this circuit's actual
    coordinates.
    """
    from app.models.f1_constants import canonical_circuit_name
    resolved = canonical_circuit_name(circuit_name)
    return CIRCUIT_COORDINATES.get(resolved) or CIRCUIT_COORDINATES.get(circuit_name)
