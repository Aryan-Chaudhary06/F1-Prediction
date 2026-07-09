"""
tests/test_concurrent_client.py
────────────────────────────────
Unit tests for app/data/concurrent_client.py.

Tests verify:
  1. All three coroutines fire concurrently (wall time ≈ max, not sum).
  2. Parsed DataFrames have the correct schema.
  3. A single API failure degrades gracefully (returns empty DataFrame,
     not an exception) while the other two results survive.
  4. Concurrent fetch is faster than sequential mock by >40%.

Run:
    pip install pytest pytest-asyncio aiohttp
    pytest tests/test_concurrent_client.py -v

No network is required — all HTTP is mocked with unittest.mock.
"""

import asyncio
import time
import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: canned API responses (minimal valid payloads)
# ─────────────────────────────────────────────────────────────────────────────

DRIVER_STANDINGS_PAYLOAD = {
    "MRData": {
        "StandingsTable": {
            "StandingsLists": [{
                "DriverStandings": [
                    {
                        "position": "1",
                        "points": "150",
                        "wins": "5",
                        "Driver": {
                            "code": "VER",
                            "givenName": "Max",
                            "familyName": "Verstappen",
                        },
                        "Constructors": [{"name": "Red Bull"}],
                    },
                    {
                        "position": "2",
                        "points": "120",
                        "wins": "3",
                        "Driver": {
                            "code": "NOR",
                            "givenName": "Lando",
                            "familyName": "Norris",
                        },
                        "Constructors": [{"name": "McLaren"}],
                    },
                ]
            }]
        }
    }
}

CONSTRUCTOR_STANDINGS_PAYLOAD = {
    "MRData": {
        "StandingsTable": {
            "StandingsLists": [{
                "ConstructorStandings": [
                    {"position": "1", "points": "300", "wins": "8",
                     "Constructor": {"name": "Red Bull"}},
                    {"position": "2", "points": "240", "wins": "4",
                     "Constructor": {"name": "McLaren"}},
                ]
            }]
        }
    }
}

SCHEDULE_PAYLOAD = {
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "round": "1",
                    "raceName": "Bahrain Grand Prix",
                    "date": "2026-03-02",
                    "Circuit": {
                        "circuitName": "Bahrain International Circuit",
                        "Location": {"country": "Bahrain"},
                    },
                },
                {
                    "round": "2",
                    "raceName": "Saudi Arabian Grand Prix",
                    "date": "2026-03-16",
                    "Circuit": {
                        "circuitName": "Jeddah Corniche Circuit",
                        "Location": {"country": "Saudi Arabia"},
                    },
                },
            ]
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_response(payload: dict, delay: float = 0.0):
    """
    Build a mock aiohttp response that returns `payload` as JSON
    and optionally sleeps for `delay` seconds (simulates latency).
    """
    async def _json(*args, **kwargs):
        if delay:
            await asyncio.sleep(delay)
        return payload

    mock_resp = AsyncMock()
    mock_resp.json = _json
    mock_resp.raise_for_status = MagicMock()
    # Support async context manager
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__  = AsyncMock(return_value=False)
    return mock_resp


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentClient:

    def _make_session_mock(self, drv_delay=0.1, con_delay=0.1, sch_delay=0.1):
        """
        Returns a mock aiohttp.ClientSession whose .get() dispatches the
        correct canned payload based on URL substring.
        """
        drv_resp = _make_mock_response(DRIVER_STANDINGS_PAYLOAD, drv_delay)
        con_resp = _make_mock_response(CONSTRUCTOR_STANDINGS_PAYLOAD, con_delay)
        sch_resp = _make_mock_response(SCHEDULE_PAYLOAD, sch_delay)

        def get_side_effect(url, **kwargs):
            if "driverStandings" in url:
                return drv_resp
            elif "constructorStandings" in url:
                return con_resp
            else:
                return sch_resp

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=get_side_effect)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__  = AsyncMock(return_value=False)
        return mock_session

    # ── 1. Schema tests ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_driver_standings_schema(self):
        """Parsed driver DataFrame has the correct columns and types."""
        import aiohttp
        from app.data.concurrent_client import _fetch_standings_page_async

        mock_session = self._make_session_mock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _fetch_standings_page_async(2026)

        df = result["drivers"]
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ["position", "driver", "full_name", "constructor", "points", "wins"]:
            assert col in df.columns, f"Missing column: {col}"
        assert df["position"].dtype in [int, "int64"]
        assert df["points"].dtype in [float, "float64"]
        assert df.iloc[0]["driver"] == "VER"

    @pytest.mark.asyncio
    async def test_constructor_standings_schema(self):
        """Parsed constructor DataFrame has the correct columns."""
        import aiohttp
        from app.data.concurrent_client import _fetch_standings_page_async

        mock_session = self._make_session_mock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _fetch_standings_page_async(2026)

        df = result["constructors"]
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ["position", "constructor", "points", "wins"]:
            assert col in df.columns

    @pytest.mark.asyncio
    async def test_schedule_schema(self):
        """Parsed schedule DataFrame has the correct columns."""
        import aiohttp
        from app.data.concurrent_client import _fetch_standings_page_async

        mock_session = self._make_session_mock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _fetch_standings_page_async(2026)

        df = result["schedule"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        for col in ["round", "gp_name", "circuit", "country", "date"]:
            assert col in df.columns

    # ── 2. Concurrency timing test ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_concurrent_is_faster_than_sequential(self):
        """
        Wall time for concurrent fetch must be < sum of individual latencies.
        With delays of 0.3s, 0.25s, 0.2s:
          sequential  ≈ 0.75s
          concurrent  ≈ 0.30s  (max of the three)
        """
        import aiohttp
        from app.data.concurrent_client import _fetch_standings_page_async

        # Simulate realistic API latency per endpoint
        mock_session = self._make_session_mock(
            drv_delay=0.30,
            con_delay=0.25,
            sch_delay=0.20,
        )

        # Sequential baseline (await one after another)
        t0 = time.perf_counter()
        await asyncio.sleep(0.30)  # driver
        await asyncio.sleep(0.25)  # constructor
        await asyncio.sleep(0.20)  # schedule
        seq_elapsed = time.perf_counter() - t0

        # Concurrent via asyncio.gather
        with patch("aiohttp.ClientSession", return_value=mock_session):
            t0 = time.perf_counter()
            await _fetch_standings_page_async(2026)
            con_elapsed = time.perf_counter() - t0

        reduction = (seq_elapsed - con_elapsed) / seq_elapsed * 100
        assert reduction > 40, (
            f"Expected >40% latency reduction, got {reduction:.1f}%"
        )

    # ── 3. Graceful degradation test ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_single_failure_degrades_gracefully(self):
        """
        If one endpoint raises an exception, the other two results must
        still be returned as valid DataFrames (not None, not exception).
        """
        import aiohttp
        from app.data.concurrent_client import _fetch_standings_page_async

        con_resp = _make_mock_response(CONSTRUCTOR_STANDINGS_PAYLOAD)
        sch_resp = _make_mock_response(SCHEDULE_PAYLOAD)

        # Make driver standings endpoint raise a timeout
        error_resp = AsyncMock()
        error_resp.raise_for_status = MagicMock(
            side_effect=Exception("Simulated timeout")
        )
        error_resp.__aenter__ = AsyncMock(return_value=error_resp)
        error_resp.__aexit__  = AsyncMock(return_value=False)

        def get_side_effect(url, **kwargs):
            if "driverStandings" in url:
                return error_resp
            elif "constructorStandings" in url:
                return con_resp
            else:
                return sch_resp

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=get_side_effect)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__  = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _fetch_standings_page_async(2026)

        # Drivers failed → empty DataFrame, not exception
        assert isinstance(result["drivers"], pd.DataFrame)
        assert result["drivers"].empty

        # Constructors and schedule still work
        assert not result["constructors"].empty
        assert not result["schedule"].empty

    # ── 4. Metadata test ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_latency_metadata_returned(self):
        """Result dict must include latency_s and per_call_latency keys."""
        import aiohttp
        from app.data.concurrent_client import _fetch_standings_page_async

        mock_session = self._make_session_mock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _fetch_standings_page_async(2026)

        assert "latency_s" in result
        assert "per_call_latency" in result
        assert isinstance(result["latency_s"], float)
        assert result["latency_s"] > 0

    # ── 5. Public sync wrapper ────────────────────────────────────────────────

    def test_sync_wrapper_returns_dataframes(self):
        """
        fetch_standings_page() (sync) must work from non-async context
        (i.e. exactly how Streamlit calls it).
        """
        import aiohttp
        from app.data.concurrent_client import fetch_standings_page

        mock_session = self._make_session_mock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = fetch_standings_page(2026)

        assert isinstance(result["drivers"],      pd.DataFrame)
        assert isinstance(result["constructors"], pd.DataFrame)
        assert isinstance(result["schedule"],     pd.DataFrame)
        assert not result["drivers"].empty


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner  (python tests/test_concurrent_client.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
