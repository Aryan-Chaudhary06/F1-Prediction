"""
tests/test_practice_pace_features.py
────────────────────────────────────────
Regression tests for practice_pace_features.py (RaceMindAI_Redesign_
Phases6-7.md §6.5 — designed in the roadmap, built and tested here).

All tests build fabricated lap data shaped exactly like
fastf1_client.get_lap_times()'s documented output schema (Driver,
LapNumber, LapTimeSeconds, Stint, Compound, TyreLife) rather than hitting
a real FastF1 session, so these run offline. See the module's own
docstring for the one thing this suite can't cover: an actual live smoke
test against a real completed session.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import pandas as pd
import pytest
from unittest.mock import patch

from app.models.practice_pace_features import (
    _extract_long_run_laps,
    _session_pace_summary,
    build_practice_pace_lookup,
    attach_practice_pace_features,
    default_practice_pace_row,
    get_session_practice_pace,
    MIN_LONG_RUN_LAPS,
)


def _stint_rows(driver, times, stint=1, compound="MEDIUM"):
    return [
        {"Driver": driver, "LapNumber": i + 1, "LapTimeSeconds": t,
         "Stint": stint, "Compound": compound, "TyreLife": i + 1}
        for i, t in enumerate(times)
    ]


class TestLongRunFiltering:

    def test_short_stint_is_excluded_entirely(self):
        # A qualifying-simulation stint (fewer than MIN_LONG_RUN_LAPS) is
        # fast but NOT representative of race pace — must not appear at all.
        assert MIN_LONG_RUN_LAPS >= 4  # sanity check the constant itself is sane
        short_stint = _stint_rows("C", [88.0, 87.8, 87.9])  # 3 laps
        laps_df = pd.DataFrame(short_stint)
        result = _extract_long_run_laps(laps_df)
        assert result.empty

    def test_out_lap_is_removed_from_a_long_run(self):
        # First lap of a stint is unrepresentative (still finding grip
        # after the pit stop) — must be dropped even from a long run.
        laps = _stint_rows("A", [95.0, 90.2, 90.1, 90.3, 90.0, 90.4])  # first lap is slow/atypical
        result = _extract_long_run_laps(pd.DataFrame(laps))
        assert 95.0 not in result["LapTimeSeconds"].values

    def test_genuine_outlier_lap_is_removed(self):
        # A lap far slower than the stint's own median (traffic, yellow
        # flag, mistake) is noise, not pace signal.
        laps = _stint_rows("A", [90.5, 90.2, 90.1, 90.3, 110.0, 90.0, 90.4, 90.2])
        result = _extract_long_run_laps(pd.DataFrame(laps))
        assert 110.0 not in result["LapTimeSeconds"].values

    def test_mild_deviation_within_threshold_is_kept(self):
        # A lap only slightly slower than the median (realistic normal
        # variation, not an outlier) should NOT be discarded — the filter
        # shouldn't be so aggressive it throws away real signal.
        laps = _stint_rows("A", [90.5, 90.2, 90.1, 90.3, 95.0, 90.0, 90.4, 90.2])  # ~5% deviation
        result = _extract_long_run_laps(pd.DataFrame(laps))
        assert 95.0 in result["LapTimeSeconds"].values

    def test_empty_input_returns_empty_output(self):
        result = _extract_long_run_laps(pd.DataFrame(columns=["Driver", "LapNumber", "LapTimeSeconds", "Stint"]))
        assert result.empty


class TestPaceSummary:

    def test_fastest_driver_gets_zero_delta(self):
        laps = pd.DataFrame(
            _stint_rows("A", [90.0] * 6) + _stint_rows("B", [91.0] * 6)
        )
        summary = _session_pace_summary(laps)
        a_row = summary[summary["driver"] == "A"].iloc[0]
        assert a_row["practice_pace_delta_pct"] == 0.0

    def test_slower_driver_gets_positive_delta(self):
        laps = pd.DataFrame(
            _stint_rows("A", [90.0] * 6) + _stint_rows("B", [99.0] * 6)  # 10% slower
        )
        summary = _session_pace_summary(laps)
        b_row = summary[summary["driver"] == "B"].iloc[0]
        assert b_row["practice_pace_delta_pct"] == pytest.approx(10.0, abs=0.1)

    def test_lap_count_reflects_surviving_laps(self):
        laps = pd.DataFrame(_stint_rows("A", [90.0] * 6))
        summary = _session_pace_summary(laps)
        assert summary.iloc[0]["practice_lap_count"] == 6

    def test_empty_input_returns_empty_dataframe_with_expected_columns(self):
        summary = _session_pace_summary(pd.DataFrame(columns=["Driver", "LapTimeSeconds"]))
        assert summary.empty
        assert list(summary.columns) == ["driver", "practice_pace_delta_pct", "practice_lap_count"]


class TestBuildPracticePaceLookup:

    def test_future_race_is_skipped_without_fetching(self):
        today = datetime.date.today()
        schedule = pd.DataFrame([
            {"year": 2026, "round": 1, "date": str(today - datetime.timedelta(days=10))},
            {"year": 2026, "round": 20, "date": str(today + datetime.timedelta(days=60))},
        ])

        call_log = []
        def fake_get_lap_times(year, round_, session):
            call_log.append(round_)
            return pd.DataFrame(_stint_rows("A", [90.0] * 6))

        with patch("app.models.practice_pace_features.get_lap_times", side_effect=fake_get_lap_times):
            lookup = build_practice_pace_lookup(schedule)

        assert call_log == [1], "Only the past race should have been fetched, not the future one"
        assert len(lookup) == 1
        assert lookup.iloc[0]["round"] == 1

    def test_fetch_failure_is_soft_and_does_not_raise(self):
        today = datetime.date.today()
        schedule = pd.DataFrame([
            {"year": 2026, "round": 1, "date": str(today - datetime.timedelta(days=10))},
        ])

        def failing_get_lap_times(*a, **k):
            raise RuntimeError("simulated FastF1 fetch failure")

        with patch("app.models.practice_pace_features.get_lap_times", side_effect=failing_get_lap_times):
            lookup = build_practice_pace_lookup(schedule)  # must not raise
        assert lookup.empty


class TestAttachPracticePaceFeatures:

    def test_missing_driver_gets_default_row(self):
        lookup = pd.DataFrame([
            {"year": 2026, "round": 1, "driver": "A", "practice_pace_delta_pct": 0.0, "practice_lap_count": 8},
        ])
        feat_df = pd.DataFrame({"year": [2026, 2026], "round": [1, 1], "driver": ["A", "B"]})
        out = attach_practice_pace_features(feat_df, lookup)

        b_row = out[out["driver"] == "B"].iloc[0]
        defaults = default_practice_pace_row()
        assert b_row["practice_pace_delta_pct"] == defaults["practice_pace_delta_pct"]
        assert b_row["practice_lap_count"] == defaults["practice_lap_count"]

    def test_matched_driver_keeps_real_value_not_default(self):
        lookup = pd.DataFrame([
            {"year": 2026, "round": 1, "driver": "A", "practice_pace_delta_pct": 1.23, "practice_lap_count": 8},
        ])
        feat_df = pd.DataFrame({"year": [2026], "round": [1], "driver": ["A"]})
        out = attach_practice_pace_features(feat_df, lookup)
        assert out.iloc[0]["practice_pace_delta_pct"] == pytest.approx(1.23)


class TestGetSessionPracticePace:

    def test_returns_empty_dict_on_fetch_failure_rather_than_raising(self):
        with patch("app.models.practice_pace_features.get_lap_times",
                   side_effect=RuntimeError("simulated failure")):
            result = get_session_practice_pace(2026, 5)
        assert result == {}

    def test_returns_per_driver_dict_on_success(self):
        def fake_get_lap_times(year, round_, session):
            return pd.DataFrame(
                _stint_rows("A", [90.0] * 6) + _stint_rows("B", [92.0] * 6)
            )
        with patch("app.models.practice_pace_features.get_lap_times", side_effect=fake_get_lap_times):
            result = get_session_practice_pace(2026, 5)

        assert "A" in result and "B" in result
        assert result["A"]["practice_pace_delta_pct"] == 0.0
        assert result["B"]["practice_pace_delta_pct"] > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
