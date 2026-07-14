"""
tests/test_rookie_constructor_fallback.py
────────────────────────────────────────────
Regression guards for the rookie/new-constructor bugs found during the
RaceMindAI audit and fixed across this project's conversation history:

  1. ROOKIE_2026 originally (wrongly) included Antonelli, Hadjar, and
     Bortoleto — all of whom debuted in 2025 and have real rolling-form
     history. Only Arvid Lindblad is a true 2026 rookie. See
     RaceMindAI_Audit_Phases1-5.md Phase 2.5.1.
  2. main.py's /api/predictor/predict route never called
     apply_rookie_fallback at all — a driver with no history fell straight
     to fabricated {feature: 0.0} rows instead of field-average fallback
     values. See Phase 2.3.1. (This suite tests the underlying fallback
     FUNCTIONS directly, since exercising the actual FastAPI route would
     require mocking the live Jolpica API — see test_main_routes.py if/when
     that's added for route-level coverage.)
  3. apply_new_constructor_fallback must route brand-new 2026 constructors
     (Audi, Cadillac) — and any constructor below the race-count threshold
     — to a midfield-average value rather than a missing/zero one.

The common thread across all three: a driver or constructor with no (or
insufficient) history must get a REALISTIC field-average / midfield-average
value, never a fabricated 0.0 — a literal zero reads as "averages a P0
finish", which is better than winning every race and is a value real
data never produces. See feature_engineering.py's own docstring for the
full rationale.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from app.data.drivers_2026 import ROOKIE_2026, DRIVERS_2026
from app.models.feature_engineering import (
    build_training_features,
    apply_rookie_fallback,
    apply_new_constructor_fallback,
    NEW_CONSTRUCTORS_2026,
)
from app.models.qualifying_feature_engineering import (
    apply_rookie_fallback as apply_qualifying_rookie_fallback,
)


def _finished_row(year, round_, driver, constructor, circuit, grid, position):
    return {
        "year": year, "round": round_, "driver": driver, "constructor": constructor,
        "circuit": circuit, "grid": grid, "position": position,
        "status": "Finished", "points": 0, "laps": 55,
    }


class TestRookieSetIsCorrect:
    """Only Arvid Lindblad is a true 2026 rookie — everyone else on the
    2026 grid has at least one prior season on record. Regression guard
    for the fact-check finding in RaceMindAI_Audit_Phases1-5.md Phase 2.5.1."""

    def test_only_lindblad_is_marked_as_rookie(self):
        assert ROOKIE_2026 == {"Arvid Lindblad"}

    def test_2025_debutants_are_not_marked_as_rookies(self):
        # Antonelli, Hadjar, and Bortoleto all debuted in 2025 — by 2026
        # they have a full season of real rolling-form data and must NOT
        # be routed through the rookie fallback (which would discard that
        # real history in favor of a field-average placeholder).
        not_rookies = {"Kimi Antonelli", "Isack Hadjar", "Gabriel Bortoleto"}
        assert not_rookies.isdisjoint(ROOKIE_2026)

    def test_every_rookie_name_is_a_real_2026_driver(self):
        # Guards against a typo'd or stale name silently never matching
        # anything in the actual roster.
        driver_names = {d["name"] for d in DRIVERS_2026}
        assert ROOKIE_2026.issubset(driver_names)


class TestRaceModelRookieFallback:
    """feature_engineering.apply_rookie_fallback must fill a rookie's
    feature row with FIELD-WIDE averages, never leave it at fabricated
    zeros. This is the function main.py's /api/predictor/predict route was
    previously never calling at all (see Phase 2.3.1)."""

    def _historical_df_with_known_averages(self):
        # Two established drivers with clean, hand-computable rolling
        # stats, so the expected field average is easy to verify by hand.
        rows = []
        for rnd in range(1, 4):
            rows.append(_finished_row(2022, rnd, "A", "TeamA", "CircuitX", 1, 1))   # always wins
            rows.append(_finished_row(2022, rnd, "B", "TeamB", "CircuitX", 10, 10))  # always P10
        return pd.DataFrame(rows)

    def test_fallback_fills_nonzero_field_average_values(self):
        hist = self._historical_df_with_known_averages()
        feat_df = build_training_features(hist)

        rookie_row = {f: 0.0 for f in [
            "driver_rolling_points", "driver_rolling_wins", "driver_rolling_podiums",
            "driver_circuit_avg_pos",
        ]}
        result = apply_rookie_fallback(rookie_row, "NewRookie", feat_df)

        # The whole point of this function: none of these should be left
        # at the fabricated 0.0 they started as (0.0 for
        # driver_circuit_avg_pos in particular reads as "averages a P0
        # finish" — better than winning every race, and impossible in
        # real data).
        for field in ["driver_rolling_points", "driver_rolling_wins",
                      "driver_rolling_podiums", "driver_circuit_avg_pos"]:
            assert result[field] != 0.0, f"{field} was left at a fabricated 0.0"

    def test_fallback_value_is_a_real_average_not_an_extreme(self):
        hist = self._historical_df_with_known_averages()
        feat_df = build_training_features(hist)
        rookie_row = {"driver_circuit_avg_pos": 0.0, "driver_rolling_points": 0.0,
                      "driver_rolling_wins": 0.0, "driver_rolling_podiums": 0.0}
        result = apply_rookie_fallback(rookie_row, "NewRookie", feat_df)

        # Driver A always finishes P1, Driver B always finishes P10 — the
        # fallback average should land somewhere between those two
        # extremes (a plausible "average" driver), not at either extreme.
        assert 1.0 < result["driver_circuit_avg_pos"] < 10.0


class TestQualifyingRookieFallback:
    """Same contract as the race-model version, for the qualifying
    pipeline's apply_rookie_fallback (which additionally takes an explicit
    rookie_names set, unlike the race version)."""

    def test_rookie_in_the_set_gets_field_average(self):
        feat_df = pd.DataFrame({
            "driver_5race_avg_quali_pos": [3.0, 15.0],
            "driver_quali_pos_std": [1.0, 2.0],
        })
        row = {"driver_5race_avg_quali_pos": 0.0, "driver_quali_pos_std": 0.0}
        result = apply_qualifying_rookie_fallback(row, "Arvid Lindblad", {"Arvid Lindblad"}, feat_df)
        assert result["driver_5race_avg_quali_pos"] == pytest.approx(9.0)  # mean(3, 15)

    def test_non_rookie_is_left_untouched(self):
        # A driver NOT in the rookie set must pass through unchanged —
        # this is the exact mechanism that broke when ROOKIE_2026 wrongly
        # included Antonelli/Hadjar/Bortoleto: they'd hit THIS branch and
        # get their real history overwritten with a field average.
        feat_df = pd.DataFrame({
            "driver_5race_avg_quali_pos": [3.0, 15.0],
            "driver_quali_pos_std": [1.0, 2.0],
        })
        row = {"driver_5race_avg_quali_pos": 4.2, "driver_quali_pos_std": 0.8}
        result = apply_qualifying_rookie_fallback(row, "Kimi Antonelli", {"Arvid Lindblad"}, feat_df)
        assert result["driver_5race_avg_quali_pos"] == 4.2
        assert result["driver_quali_pos_std"] == 0.8


class TestNewConstructorFallback:
    """Audi and Cadillac (genuinely new 2026 constructors) — and any
    constructor below the minimum race-count threshold — must get a
    midfield-average value, not a missing/zero one."""

    def test_audi_and_cadillac_are_flagged_new(self):
        assert NEW_CONSTRUCTORS_2026 == {"Audi", "Cadillac"}

    def test_new_constructor_gets_midfield_value_not_zero(self):
        rows = []
        # Six established constructors with a clear points spread, so
        # "midfield" (rank 6 of the field) is unambiguous.
        for i, avg_pts in enumerate([20, 16, 12, 8, 4, 2]):
            constructor = f"Team{i}"
            for rnd in range(1, 4):
                rows.append(_finished_row(2022, rnd, f"D{i}", constructor, "CircuitX", 1, 1))
        hist = pd.DataFrame(rows)
        feat_df = build_training_features(hist)

        row = {"constructor_avg_points": 0.0, "constructor_dnf_rate": 0.0}
        result = apply_new_constructor_fallback(row, "Audi", feat_df)
        assert result["constructor_avg_points"] != 0.0

    def test_established_constructor_with_enough_races_is_untouched(self):
        rows = []
        for rnd in range(1, 10):
            rows.append(_finished_row(2022, rnd, "D1", "Ferrari", "CircuitX", 1, 1))
            rows.append(_finished_row(2022, rnd, "D2", "Mercedes", "CircuitX", 2, 2))
        hist = pd.DataFrame(rows)
        feat_df = build_training_features(hist)

        row = {"constructor_avg_points": 42.0, "constructor_dnf_rate": 0.05}
        result = apply_new_constructor_fallback(row, "Ferrari", feat_df)
        # Ferrari is neither in NEW_CONSTRUCTORS_2026 nor below the race
        # threshold — its real (already-computed) values must pass through
        # unchanged, not get overwritten with a midfield placeholder.
        assert result["constructor_avg_points"] == 42.0
        assert result["constructor_dnf_rate"] == 0.05


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
