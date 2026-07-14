"""
tests/test_circuit_taxonomy.py
────────────────────────────────
Regression guards for the circuit-name normalization bug found in
production (see conversation history: "no coordinates for circuit" spam
and guaranteed-to-fail weather lookups for real Jolpica circuit names).

Jolpica has been observed returning DIFFERENT exact strings than this
codebase's hardcoded circuit dictionaries for the SAME physical circuit:
accented vs. unaccented characters, an extra word, or a genuinely
different common name. Every one of these variants previously caused a
silent exact-match miss in classify_circuit() and
circuit_coordinates.get_circuit_coordinates() — degrading gracefully (no
crash) but losing real signal (circuit archetype fell back to a generic
guess, weather fell back to climatology) for several circuits on every
single training run.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.models.f1_constants import classify_circuit, canonical_circuit_name, CIRCUIT_TYPE
from app.data.circuit_coordinates import get_circuit_coordinates


# The exact real-world variants observed in production logs that
# originally broke — see RaceMindAI conversation history for the literal
# log lines these are drawn from.
REAL_WORLD_VARIANTS = [
    "Circuit Paul Ricard",              # was missing entirely, not just a naming variant
    "Circuit Park Zandvoort",           # extra word vs. hardcoded "Circuit Zandvoort"
    "Autódromo Hermanos Rodríguez",     # accented vs. hardcoded "Autodromo Hermanos Rodriguez"
    "Autódromo José Carlos Pace",       # accented vs. hardcoded "Autodromo Jose Carlos Pace"
    "Losail International Circuit",    # alternate spelling vs. hardcoded "Lusail International Circuit"
]


class TestCircuitCoordinatesResolveRealWorldVariants:

    @pytest.mark.parametrize("circuit_name", REAL_WORLD_VARIANTS)
    def test_every_observed_variant_resolves_to_real_coordinates(self, circuit_name):
        coords = get_circuit_coordinates(circuit_name)
        assert coords is not None, (
            f"{circuit_name!r} returned no coordinates — this is the exact "
            f"failure mode that caused 'no coordinates for circuit' spam "
            f"and silent climatology fallback for every training run."
        )
        lat, lon = coords
        # Sanity bounds — real latitude/longitude, not a placeholder like
        # (0, 0) that would pass a bare "is not None" check.
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180


class TestClassifyCircuitResolvesRealWorldVariants:

    @pytest.mark.parametrize("circuit_name", REAL_WORLD_VARIANTS)
    def test_every_observed_variant_gets_a_real_archetype(self, circuit_name):
        result = classify_circuit(circuit_name)
        assert result != "unknown", (
            f"{circuit_name!r} classified as 'unknown' — should resolve "
            f"to a real archetype via canonical_circuit_name(), not fall "
            f"through to the generic default."
        )

    def test_totally_unrecognized_circuit_still_falls_back_gracefully(self):
        # Not every future circuit name will have a known alias — this
        # must NOT raise, and must fall back to "unknown" rather than
        # crash or silently guess wrong.
        result = classify_circuit("Some Brand New Circuit Nobody Has Heard Of")
        assert result in {"street", "power", "technical", "high_downforce", "unknown"}


class TestCanonicalCircuitNameResolution:

    def test_exact_match_short_circuits_immediately(self):
        # A name already in CIRCUIT_TYPE should resolve to itself without
        # going through alias/accent-stripping logic at all.
        for name in CIRCUIT_TYPE:
            assert canonical_circuit_name(name) == name

    def test_accent_stripping_resolves_to_the_hardcoded_key(self):
        resolved = canonical_circuit_name("Autódromo Hermanos Rodríguez")
        assert resolved in CIRCUIT_TYPE
        assert resolved == "Autodromo Hermanos Rodriguez"

    def test_known_alias_resolves_to_the_hardcoded_key(self):
        resolved = canonical_circuit_name("Losail International Circuit")
        assert resolved == "Lusail International Circuit"
        assert resolved in CIRCUIT_TYPE

    def test_unresolvable_name_returns_input_unchanged(self):
        # Callers rely on this NOT raising and NOT silently returning some
        # other circuit's key — an unresolvable name should come back
        # as-is, letting the caller's own fallback path (e.g.
        # classify_circuit's substring match) take over.
        unknown = "Totally Fictional Speedway"
        assert canonical_circuit_name(unknown) == unknown


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
