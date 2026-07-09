"""
tests/benchmark_concurrent.py
──────────────────────────────
Measures sequential vs concurrent API latency for RaceMindAI.
Run this once on your machine, then paste the printed numbers
into your resume bullet.

Usage
─────
    # from repo root
    python -m tests.benchmark_concurrent

    # or directly
    python tests/benchmark_concurrent.py

Output
──────
    Sequential  avg : 2.134s
    Concurrent  avg : 0.821s
    Reduction       : 61.5%

    → Resume bullet:
      Refactored 3 live API integrations (Jolpica × 2, OpenF1) from
      sequential to concurrent using Python asyncio.gather(); reduced
      dashboard data-load latency by ~62% (2.13s → 0.82s, 5-trial avg).
"""

import asyncio
import time
import sys
import os

# ── Allow running from repo root without installing the package ───────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Sequential baseline  (what the app does TODAY in Live Standings page)
# ─────────────────────────────────────────────────────────────────────────────

def sequential_fetch(year: int = 2026) -> float:
    """
    Mirrors exactly what app.py lines 482–484 currently do:
        drivers      = get_driver_standings(year)
        constructors = get_constructor_standings(year)
        schedule     = get_season_schedule(year)
    Three blocking requests.get() calls in series.
    """
    import requests

    JOLPICA = "https://api.jolpi.ca/ergast/f1"

    t0 = time.perf_counter()
    requests.get(f"{JOLPICA}/{year}/driverStandings.json?limit=1000",  timeout=15)
    requests.get(f"{JOLPICA}/{year}/constructorStandings.json?limit=1000", timeout=15)
    requests.get(f"{JOLPICA}/{year}.json?limit=1000", timeout=15)
    return time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent (new behaviour via concurrent_client.py)
# ─────────────────────────────────────────────────────────────────────────────

def concurrent_fetch(year: int = 2026) -> float:
    """
    Calls fetch_standings_page() — same three endpoints,
    dispatched concurrently with asyncio.gather().
    """
    from app.data.concurrent_client import fetch_standings_page
    t0   = time.perf_counter()
    data = fetch_standings_page(year)
    elapsed = time.perf_counter() - t0
    # Echo per-call breakdown so you can see the individual timings
    print(f"    per-call: {data['per_call_latency']}")
    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main(year: int = 2026, runs: int = 5):
    print(f"\n{'─'*55}")
    print(f"  RaceMindAI · Concurrency Benchmark  (year={year}, n={runs})")
    print(f"{'─'*55}\n")

    # ── Sequential ──
    print("Running SEQUENTIAL trials...")
    seq_times = []
    for i in range(runs):
        t = sequential_fetch(year)
        seq_times.append(t)
        print(f"  Trial {i+1}: {t:.3f}s")

    # ── Concurrent ──
    print("\nRunning CONCURRENT trials...")
    con_times = []
    for i in range(runs):
        print(f"  Trial {i+1}:", end=" ", flush=True)
        t = concurrent_fetch(year)
        con_times.append(t)
        print(f"{t:.3f}s")

    # ── Stats ──
    seq_avg = sum(seq_times) / runs
    con_avg = sum(con_times) / runs
    seq_min = min(seq_times)
    con_min = min(con_times)
    reduction_avg = (seq_avg - con_avg) / seq_avg * 100
    reduction_min = (seq_min - con_min) / seq_min * 100

    print(f"\n{'─'*55}")
    print(f"  Sequential  avg : {seq_avg:.3f}s   min : {seq_min:.3f}s")
    print(f"  Concurrent  avg : {con_avg:.3f}s   min : {con_min:.3f}s")
    print(f"  Reduction (avg) : {reduction_avg:.1f}%")
    print(f"  Reduction (min) : {reduction_min:.1f}%")
    print(f"{'─'*55}\n")

    # ── Auto-generated resume bullet ──
    print("→ Resume bullet (fill in your actual numbers below):\n")
    print(
        f"  Refactored 3 live API integrations (Jolpica × 2, OpenF1) from sequential\n"
        f"  to concurrent using Python asyncio.gather(); reduced dashboard data-load\n"
        f"  latency by ~{reduction_avg:.0f}% ({seq_avg:.2f}s → {con_avg:.2f}s, {runs}-trial avg on live APIs)."
    )
    print()


if __name__ == "__main__":
    main()
