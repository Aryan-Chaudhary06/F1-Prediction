"""
tests/test_feature_leakage.py
────────────────────────────────
Regression guards for the leakage bugs found and fixed during the
RaceMindAI audit (see RaceMindAI_Audit_Phases1-5.md Phase 2.1 and
RaceMindAI_Redesign_Phases6-7.md §6.3):

  1. driver_circuit_avg_pos used to be a whole-dataset mean, so a
     driver's "average finish at this circuit" feature for an EARLY race
     silently included results from races that hadn't happened yet.
  2. constructor_avg_points used to average points_scored WITHIN the same
     (constructor, year, round) — i.e. across a constructor's two cars IN
     THE SAME RACE being predicted, encoding that race's own outcome into
     its own feature.
  3. race_predictor.train_model() used a random row-level train_test_split
     instead of a chronological, race-grouped split, allowing rows from
     the same race weekend (or even a LATER race) to leak into training
     while an earlier race ended up in the test set.

These tests build small synthetic result sets with a KNOWN correct answer
(computed by hand in each test) rather than asserting against real Jolpica
data, so they run offline and stay meaningful even as the training data
itself changes.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from app.models.feature_engineering import build_training_features
from app.models.race_predictor import chronological_split


def _finished_row(year, round_, driver, constructor, circuit, grid, position):
    return {
        "year": year, "round": round_, "driver": driver, "constructor": constructor,
        "circuit": circuit, "grid": grid, "position": position,
        "status": "Finished", "points": 0, "laps": 55,
    }


class TestDriverCircuitAvgPosNoLeakage:
    """driver_circuit_avg_pos must only ever reflect races STRICTLY BEFORE
    the row it's attached to."""

    def test_first_visit_to_a_circuit_is_nan(self):
        # A driver's very first race at a circuit has no prior history —
        # this MUST be NaN, not a fabricated 0.0 (which would read as a
        # superhuman "averages P0" per feature_engineering.py's own
        # rookie-fallback docstring) and not silently backfilled from a
        # LATER visit to the same circuit.
        df = pd.DataFrame([
            _finished_row(2022, 1, "X", "TeamX", "CircuitA", 1, 1),
        ])
        out = build_training_features(df)
        assert pd.isna(out.iloc[0]["driver_circuit_avg_pos"])

    def test_second_visit_only_uses_the_first_races_result(self):
        df = pd.DataFrame([
            _finished_row(2022, 1, "X", "TeamX", "CircuitA", 1, 1),   # P1
            _finished_row(2023, 1, "X", "TeamX", "CircuitA", 5, 5),   # P5
        ])
        out = build_training_features(df).sort_values("year").reset_index(drop=True)
        assert pd.isna(out.iloc[0]["driver_circuit_avg_pos"])
        # The SECOND row's feature must equal the FIRST row's actual
        # position (1.0) — not include its own position (5), and not be
        # some other value.
        assert out.iloc[1]["driver_circuit_avg_pos"] == pytest.approx(1.0)

    def test_third_visit_averages_only_the_first_two_not_itself(self):
        df = pd.DataFrame([
            _finished_row(2022, 1, "X", "TeamX", "CircuitA", 1, 1),    # P1
            _finished_row(2023, 1, "X", "TeamX", "CircuitA", 5, 5),    # P5
            _finished_row(2024, 1, "X", "TeamX", "CircuitA", 10, 10),  # P10 — must NOT be included in its own feature
        ])
        out = build_training_features(df).sort_values("year").reset_index(drop=True)
        assert out.iloc[2]["driver_circuit_avg_pos"] == pytest.approx((1 + 5) / 2)

    def test_future_race_never_leaks_into_an_earlier_rows_feature(self):
        # Regression case for the ORIGINAL bug: a groupby().mean() over the
        # whole dataset would let a driver's 2024 result affect their 2022
        # row's feature. Explicitly assert that does NOT happen.
        df = pd.DataFrame([
            _finished_row(2022, 1, "X", "TeamX", "CircuitA", 1, 1),    # P1 — first ever, should be NaN
            _finished_row(2023, 1, "X", "TeamX", "CircuitA", 1, 20),   # P20 — a very bad future result
        ])
        out = build_training_features(df).sort_values("year").reset_index(drop=True)
        # The 2022 row must stay NaN regardless of how bad the driver's
        # LATER (2023) result at this circuit was.
        assert pd.isna(out.iloc[0]["driver_circuit_avg_pos"])


class TestConstructorAvgPointsNoLeakage:
    """constructor_avg_points must never include the current race's own
    result — not even a teammate's result from the SAME race."""

    def test_first_race_for_a_constructor_is_nan(self):
        df = pd.DataFrame([
            _finished_row(2022, 1, "X", "TeamX", "CircuitA", 1, 1),
            _finished_row(2022, 1, "Y", "TeamX", "CircuitA", 2, 20),
        ])
        out = build_training_features(df)
        assert out["constructor_avg_points"].isna().all()

    def test_teammates_same_race_result_does_not_leak_into_either_row(self):
        # Regression case for the ORIGINAL bug: constructor_avg_points used
        # to be computed via groupby(["constructor","year","round"]).mean(),
        # which averages points WITHIN the same race — X's feature row
        # would include Y's result from the SAME race, and vice versa.
        # After the fix, round 2's constructor_avg_points must be based
        # ONLY on round 1's results, not round 2's own (including a
        # teammate's) result.
        df = pd.DataFrame([
            _finished_row(2022, 1, "X", "TeamX", "CircuitA", 1, 1),    # P1 = 25 pts
            _finished_row(2022, 1, "Y", "TeamX", "CircuitA", 2, 20),   # P20 = 0 pts
            _finished_row(2022, 2, "X", "TeamX", "CircuitB", 1, 1),    # P1 = 25 pts (this round's OWN result)
            _finished_row(2022, 2, "Y", "TeamX", "CircuitB", 2, 1),    # Y suddenly wins too — must not affect X's round-2 feature
        ])
        out = build_training_features(df)
        round2_x = out[(out["driver"] == "X") & (out["round"] == 2)].iloc[0]
        # Expected: round 1's constructor average = (25 + 0) / 2 = 12.5,
        # rolled forward as the ONLY prior data point — round 2's own
        # results (X's own win AND Y's surprise win) must not appear here.
        assert round2_x["constructor_avg_points"] == pytest.approx(12.5)


class TestChronologicalSplit:
    """race_predictor.chronological_split() must group by whole races and
    never let a later race end up in train while an earlier one is in
    test."""

    def _build_multi_season_df(self, n_races=10):
        rows = []
        for rnd in range(1, n_races + 1):
            rows.append(_finished_row(2022, rnd, "X", "TeamX", "CircuitA", 1, 1))
            rows.append(_finished_row(2022, rnd, "Y", "TeamX", "CircuitA", 2, 2))
        return pd.DataFrame(rows)

    def test_test_set_is_strictly_the_most_recent_races(self):
        df = self._build_multi_season_df(n_races=10).sort_values(["year", "round"]).reset_index(drop=True)
        train_mask, test_mask = chronological_split(df, test_size=0.2)

        train_rounds = set(df.loc[train_mask, "round"])
        test_rounds = set(df.loc[test_mask, "round"])

        # No overlap between the two sets of ROUNDS at all.
        assert train_rounds.isdisjoint(test_rounds)
        # Every round in the test set must be chronologically AFTER every
        # round in the train set — this is the property a random
        # row-level split does NOT guarantee, and is the whole point of
        # this function existing.
        assert max(train_rounds) < min(test_rounds)

    def test_no_row_from_the_same_race_is_split_across_train_and_test(self):
        df = self._build_multi_season_df(n_races=10).sort_values(["year", "round"]).reset_index(drop=True)
        train_mask, test_mask = chronological_split(df, test_size=0.2)

        for rnd in df["round"].unique():
            rows_for_round = df[df["round"] == rnd]
            round_train_mask = train_mask[rows_for_round.index]
            round_test_mask = test_mask[rows_for_round.index]
            # A single race's rows must be ENTIRELY in train or ENTIRELY
            # in test — never split (which a naive row-level split could
            # do, since it doesn't know about race grouping at all).
            assert round_train_mask.all() or round_test_mask.all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
