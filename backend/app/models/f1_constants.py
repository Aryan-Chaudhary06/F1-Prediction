"""
app/models/f1_constants.py
────────────────────────────
Single source of truth for two pieces of data that were previously
duplicated across this codebase in ways that could silently disagree:

  1. POINTS_MAP — was defined identically in driver_dna.py,
     feature_engineering.py, and season_simulator.py. Fine as long as all
     three stayed in sync by hand; a real maintenance risk otherwise.

  2. Circuit archetype classification — TWO DIFFERENT implementations
     existed: feature_engineering.CIRCUIT_TYPE (exact-name dict lookup,
     fallback "unknown") and driver_dna.CIRCUIT_TYPES (substring match
     against a shorter keyword list, fallback "technical"). These could
     and did disagree on some circuits, and used different fallback
     philosophies (silently-wrong "technical" catch-all vs. an honest
     "unknown"). See RaceMindAI_Audit_Phases1-5.md Phase 1.8 finding, and
     Phase 2.4.

Both feature_engineering.py and driver_dna.py now import from here instead
of defining their own copies. feature_engineering.py re-exports CIRCUIT_TYPE
and POINTS_MAP under their original names so main.py's existing
`from app.models.feature_engineering import ... CIRCUIT_TYPE` import
doesn't need to change.

NAME NORMALIZATION: the exact circuit-name strings used as dict keys
below are hand-set to match what this codebase's authors expected Jolpica
to return, but Jolpica has been observed returning DIFFERENT exact
strings for the same physical circuit — accented vs. unaccented
characters ("Autódromo Hermanos Rodríguez" vs "Autodromo Hermanos
Rodriguez"), an extra word ("Circuit Park Zandvoort" vs "Circuit
Zandvoort"), or a genuinely different common name for the same track
("Losail International Circuit" vs "Lusail International Circuit" — both
real names for the Qatar circuit). See canonical_circuit_name() below,
which every dict lookup in this codebase should route through rather than
indexing CIRCUIT_TYPE (or circuit_coordinates.CIRCUIT_COORDINATES)
directly with a raw API string.
"""

import unicodedata

POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Exact circuit-name → archetype. This is now the ONLY primary source —
# previously feature_engineering.py had this exact-match version while
# driver_dna.py had a separate substring-match version with different
# coverage and a different fallback default.
CIRCUIT_TYPE = {
    "Bahrain International Circuit": "high_downforce",
    "Jeddah Corniche Circuit": "street",
    "Albert Park Grand Prix Circuit": "street",
    "Suzuka Circuit": "technical",
    "Shanghai International Circuit": "technical",
    "Miami International Autodrome": "street",
    "Autodromo Enzo e Dino Ferrari": "technical",   # Imola — off the 2026 calendar; kept for historical-row lookups
    "Circuit de Monaco": "street",
    "Circuit de Barcelona-Catalunya": "high_downforce",
    "Circuit Gilles Villeneuve": "street",
    "Red Bull Ring": "power",
    "Silverstone Circuit": "power",
    "Hungaroring": "high_downforce",
    "Circuit de Spa-Francorchamps": "power",
    "Circuit Zandvoort": "high_downforce",
    "Autodromo Nazionale di Monza": "power",
    "Baku City Circuit": "street",
    "Marina Bay Street Circuit": "street",
    "Circuit of the Americas": "technical",
    "Autodromo Hermanos Rodriguez": "high_downforce",
    "Autodromo Jose Carlos Pace": "technical",
    "Las Vegas Strip Street Circuit": "street",
    "Lusail International Circuit": "high_downforce",
    "Yas Marina Circuit": "high_downforce",
    # NEW for 2026 — added per RaceMindAI_Audit_Phases1-5.md Phase 2.5.1
    # finding #5 (this circuit was previously missing from every taxonomy
    # dict in the repo, silently falling back to "unknown"/default).
    "Madring": "street",
    # Was missing entirely (not a naming-variant issue) — hosted the 2022
    # French GP, within this app's TRAIN_YEAR_START..TRAIN_YEAR_END
    # window, so historical rows referencing it were silently falling
    # back to "unknown" every time. Long Mistral straight into a tight
    # chicane — power-circuit archetype fits better than technical here.
    "Circuit Paul Ricard": "power",
}

# Fallback keyword table, used ONLY when a circuit isn't in CIRCUIT_TYPE
# above (e.g. a circuit-name variant from a different data source, or a
# brand-new circuit not yet added to the exact table). This preserves
# driver_dna.py's original resilience to unlisted circuits without
# reintroducing a second disagreeing primary classification — it's a
# fallback path now, not an alternate source of truth.
_SUBSTRING_FALLBACK = {
    "street":         ["street", "monaco", "azerbaijan", "singapore", "miami", "vegas", "jeddah", "baku"],
    "power":          ["monza", "spa", "silverstone", "austria", "canada", "villeneuve"],
    "technical":      ["hungary", "japan", "suzuka", "cota", "americas", "mexico", "brazil", "pace"],
    "high_downforce": ["bahrain", "spain", "barcelona", "abu dhabi", "qatar", "zandvoort", "lusail", "yas marina"],
}


# Known name variants that differ by MORE than accents — i.e. can't be
# fixed by normalize_circuit_name()'s accent-stripping alone. Add an entry
# here the next time Jolpica is observed returning yet another name for a
# circuit already in CIRCUIT_TYPE, rather than duplicating that circuit's
# data under a second key.
CIRCUIT_NAME_ALIASES = {
    "losail international circuit": "Lusail International Circuit",
    "circuit park zandvoort": "Circuit Zandvoort",
}


def _strip_accents(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")


def canonical_circuit_name(name: str) -> str:
    """
    Resolves a raw circuit-name string (as returned by Jolpica, FastF1, or
    typed anywhere else) to whichever exact string is used as the dict key
    in CIRCUIT_TYPE and circuit_coordinates.CIRCUIT_COORDINATES. Every
    exact-match dict lookup on a circuit name in this codebase should
    route through this first, rather than indexing those dicts directly
    with a raw API string — see this module's docstring for why (accent
    variants, extra words, alternate common names all cause silent exact-
    match misses otherwise).

    Resolution order: (1) exact match as-is, (2) known alias table after
    accent-stripping/lowercasing, (3) accent-insensitive match against
    CIRCUIT_TYPE's own keys. Falls back to returning `name` unchanged if
    none of those hit — callers' own fallback paths (e.g.
    classify_circuit()'s substring match, or circuit_coordinates.py's
    climatology default) take it from there rather than this function
    raising or guessing.
    """
    if name in CIRCUIT_TYPE:
        return name

    normalized_key = " ".join(_strip_accents(name).lower().split())
    aliased = CIRCUIT_NAME_ALIASES.get(normalized_key)
    if aliased:
        return aliased

    target_ascii = _strip_accents(name).lower().strip()
    for key in CIRCUIT_TYPE:
        if _strip_accents(key).lower().strip() == target_ascii:
            return key

    return name


def classify_circuit(circuit_name: str) -> str:
    """
    Unified circuit-archetype classifier — the single function both
    driver_dna.py and (indirectly, via the CIRCUIT_TYPE dict) the feature
    pipelines now rely on. Resolves name variants via
    canonical_circuit_name() first, then exact match; substring fallback
    second; "unknown" (not a silent "technical" guess) if neither matches.
    """
    exact = CIRCUIT_TYPE.get(canonical_circuit_name(circuit_name))
    if exact:
        return exact
    lname = circuit_name.lower()
    for ctype, keywords in _SUBSTRING_FALLBACK.items():
        if any(k in lname for k in keywords):
            return ctype
    return "unknown"
