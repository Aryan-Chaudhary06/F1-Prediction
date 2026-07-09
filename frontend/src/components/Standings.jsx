import { useState, useEffect, useMemo } from "react";
import {
  getSchedule, getDriverStandings, getConstructorStandings,
  getGridDefaults, runSimulation, getRaceResults,
} from "../api";
import CarWidget from "./CarWidget";
import "./Standings.css";

const YEAR = 2026;

// Ergast/Jolpica returns verbose official entrant names ("Alpine F1 Team",
// "RB F1 Team", "Haas F1 Team", ...) that don't match the short team keys
// used in drivers_2026.py / CarWidget's TEAM_IMAGES ("Alpine", "Racing
// Bulls", "Haas"). Without this, those teams' car icons silently fail to
// resolve. Add to this list if another season introduces a new mismatch.
const TEAM_NAME_ALIASES = {
  "Alpine F1 Team": "Alpine",
  "RB F1 Team": "Racing Bulls",
  "Haas F1 Team": "Haas",
  "Red Bull Racing": "Red Bull",
  "Cadillac F1 Team": "Cadillac",
  "Cadillac F1": "Cadillac",
  "Audi F1 Team": "Audi",
};
const resolveTeamKey = (ergastName, knownTeamNames) => {
  if (TEAM_NAME_ALIASES[ergastName]) return TEAM_NAME_ALIASES[ergastName];
  // Fallback: case-insensitive substring match against the known team
  // names from drivers_2026.py, for any Ergast/Jolpica naming variant not
  // covered by the alias map above (e.g. an entrant name we haven't seen
  // yet). "Cadillac F1 Team" contains "cadillac" either direction.
  const lower = ergastName.toLowerCase();
  const match = knownTeamNames.find((name) => {
    const nameLower = name.toLowerCase();
    return lower.includes(nameLower) || nameLower.includes(lower);
  });
  return match || ergastName;
};

// Small "?" info widget — click to reveal a plain-language explainer.
function InfoTooltip({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="stand-info-wrap">
      <button
        type="button"
        className="stand-info-btn"
        aria-label="What does this control do?"
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open && <div className="stand-info-panel">{children}</div>}
    </span>
  );
}

export default function Standings() {
  const [schedule, setSchedule] = useState([]);
  const [drivers, setDrivers] = useState(null);       // current standings
  const [prevDrivers, setPrevDrivers] = useState(null); // standings as of previous round
  const [constructors, setConstructors] = useState(null);
  const [driversMeta, setDriversMeta] = useState({});  // code -> {team_color, number, name, team}
  const [sim, setSim] = useState(null);
  const [simLoading, setSimLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hoverRound, setHoverRound] = useState(null);
  const [roundWinners, setRoundWinners] = useState({}); // round -> winner name (lazy-loaded on hover)
  const [showAllDrivers, setShowAllDrivers] = useState(false);

  // ── Championship forecast controls (mirrors the old Streamlit page) ──
  const [numSimulations, setNumSimulations] = useState(5000);
  const [chaosLevel, setChaosLevel] = useState(0.20);       // noise_std
  const [safetyCarFreq, setSafetyCarFreq] = useState(1.0);  // safety_car_multiplier
  const [simulateDNFs, setSimulateDNFs] = useState(true);   // use_dnf_modeling
  const [forecastRunning, setForecastRunning] = useState(false);

  // ── Load everything ──────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const sched = await getSchedule(YEAR);
        if (cancelled) return;
        const sorted = [...sched].sort((a, b) => a.round - b.round);
        setSchedule(sorted);

        const now = new Date();
        const pastRounds = sorted.filter((r) => r.date && new Date(r.date) < now);
        const currentRoundNum = pastRounds.length ? pastRounds[pastRounds.length - 1].round : 1;

        const [ds, cs, meta] = await Promise.all([
          getDriverStandings(YEAR),
          getConstructorStandings(YEAR),
          getGridDefaults(YEAR),
        ]);
        if (cancelled) return;
        setDrivers(ds);
        setConstructors(cs);
        const metaMap = {};
        meta.forEach((d) => { metaMap[d.code] = d; });
        setDriversMeta(metaMap);

        // Previous round, for trend arrows / biggest movers — best effort,
        // silently skipped if unavailable (e.g. round 1, or API hiccup).
        if (currentRoundNum > 1) {
          getDriverStandings(YEAR, currentRoundNum - 1)
            .then((prev) => { if (!cancelled) setPrevDrivers(prev); })
            .catch(() => {});
        }

        // Championship simulation — reduced sample count from the API
        // default (10k) for a snappier page load; still statistically
        // reasonable for a probability readout.
        // Championship simulation, run once on load with the default
        // forecast settings; the Championship Forecast panel below lets
        // the person re-run with their own settings afterward.
        runSimulation({
          year: YEAR, current_round: currentRoundNum,
          n_simulations: numSimulations, noise_std: chaosLevel,
          safety_car_multiplier: safetyCarFreq, use_dnf_modeling: simulateDNFs,
        })
          .then((res) => { if (!cancelled) { setSim(res); setSimLoading(false); } })
          .catch((e) => { if (!cancelled) { setSimLoading(false); console.error(e); } });
      } catch (e) {
        if (!cancelled) setError(e.response?.data?.detail || e.message);
      }
    })();

    return () => { cancelled = true; };
  }, []);

  const currentRoundNum = useMemo(() => {
    const now = new Date();
    const past = schedule.filter((r) => r.date && new Date(r.date) < now);
    return past.length ? past[past.length - 1].round : (schedule[0]?.round ?? 1);
  }, [schedule]);

  const nextRace = useMemo(
    () => schedule.find((r) => r.round === currentRoundNum + 1),
    [schedule, currentRoundNum]
  );

  const prevPosByDriver = useMemo(() => {
    if (!prevDrivers) return null;
    const m = {};
    prevDrivers.forEach((d) => { m[d.driver] = d.position; });
    return m;
  }, [prevDrivers]);

  // ── Derived: biggest movers ──────────────────────────────────────────
  const biggestMovers = useMemo(() => {
    if (!drivers || !prevPosByDriver) return null;
    let gainer = null, loser = null;
    drivers.forEach((d) => {
      const prevPos = prevPosByDriver[d.driver];
      if (prevPos == null) return;
      const delta = prevPos - d.position; // positive = moved up
      if (!gainer || delta > gainer.delta) gainer = { driver: d, delta };
      if (!loser || delta < loser.delta) loser = { driver: d, delta };
    });
    return { gainer, loser };
  }, [drivers, prevPosByDriver]);

  // ── Derived: live stat strip ──────────────────────────────────────────
  const liveStats = useMemo(() => {
    if (!drivers || drivers.length < 2) return null;
    const points = drivers.map((d) => d.points);
    const largestGap = drivers[0].points - drivers[1].points;
    let closest = Infinity;
    for (let i = 0; i < drivers.length - 1; i++) {
      closest = Math.min(closest, drivers[i].points - drivers[i + 1].points);
    }
    const avg = points.reduce((a, b) => a + b, 0) / points.length;
    const winnersCount = drivers.filter((d) => d.wins > 0).length;
    return { largestGap, closest, avg, winnersCount };
  }, [drivers]);

  // ── Derived: WCC probability (approximated — see note in UI) ────────
  const wccApprox = useMemo(() => {
    if (!sim?.results || !driversMeta) return null;
    const byTeam = {};
    sim.results.forEach((r) => {
      const team = driversMeta[r.driver]?.team || "Unknown";
      byTeam[team] = (byTeam[team] || 0) + r.avg_final_points;
    });
    const total = Object.values(byTeam).reduce((a, b) => a + b, 0) || 1;
    return Object.entries(byTeam)
      .map(([team, pts]) => ({ team, pct: (pts / total) * 100 }))
      .sort((a, b) => b.pct - a.pct);
  }, [sim, driversMeta]);

  const projectedStandings = useMemo(() => {
    if (!sim?.results) return null;
    return [...sim.results].sort((a, b) => b.avg_final_points - a.avg_final_points);
  }, [sim]);

  const confidenceFor = (row, idx, arr) => {
    const next = arr[idx + 1];
    const gap = next ? row.avg_final_points - next.avg_final_points : 999;
    if (gap > 40) return "high";
    if (gap > 15) return "medium";
    return "low";
  };

  const handleRunForecast = async () => {
    setForecastRunning(true);
    try {
      const res = await runSimulation({
        year: YEAR, current_round: currentRoundNum,
        n_simulations: numSimulations, noise_std: chaosLevel,
        safety_car_multiplier: safetyCarFreq, use_dnf_modeling: simulateDNFs,
      });
      setSim(res);
    } catch (e) {
      console.error(e);
    } finally {
      setForecastRunning(false);
    }
  };

  const handleHoverRound = (round, gpName) => {
    setHoverRound(round);
    if (roundWinners[round] !== undefined) return;
    getRaceResults(YEAR, gpName)
      .then((results) => {
        const winner = results.find((r) => Number(r.Position) === 1);
        setRoundWinners((m) => ({ ...m, [round]: winner ? (winner.Abbreviation || winner.FullName) : null }));
      })
      .catch(() => setRoundWinners((m) => ({ ...m, [round]: null })));
  };

  if (error) return <div className="stand-error">Couldn't load standings — {error}</div>;
  if (!drivers || !constructors) return <div className="stand-loading">Loading championship data…</div>;

  const leader = drivers[0];
  const p2 = drivers[1];
  const leaderMeta = driversMeta[leader.driver];
  const maxConstructorPts = constructors[0]?.points || 1;
  const topSim = sim?.results?.[0];

  return (
    <div className="stand-page">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="stand-hero">
        <div className="stand-hero-text">
          <div className="stand-hero-label">
            <span className="stand-hero-live">Live · {YEAR} Season</span>
          </div>
          <h1 className="stand-hero-heading">
            Championship<br /><span className="accent">Standings</span>
          </h1>
          <div className="stand-hero-sub">
            {YEAR} Formula 1 Season · Round {currentRoundNum} / {schedule.length || 24}
          </div>
        </div>
        {leaderMeta && (
          <div className="stand-hero-car">
            <CarWidget driver={leaderMeta} size="hero" />
          </div>
        )}
      </section>

      {/* ── Leader card ──────────────────────────────────────────────── */}
      <section className="stand-section">
        <div className="stand-leader-card">
          <div className="stand-leader-car-wrap">
            {leaderMeta && <CarWidget driver={leaderMeta} size="card" />}
          </div>
          <div className="stand-leader-info">
            <div className="stand-leader-tag">Championship Leader</div>
            <div className="stand-leader-name" style={{ marginTop: "1.4rem" }}>
              {leader.full_name}
            </div>
            <div className="stand-leader-team">{leader.constructor}</div>
          </div>
          <div className="stand-leader-points">
            <span className="stand-leader-points-num">{leader.points}</span>
            <span className="stand-leader-points-unit">PTS</span>
            {p2 && <div className="stand-leader-gap">+{(leader.points - p2.points).toFixed(0)} PTS OVER P2</div>}
          </div>
          <div className="stand-leader-stats">
            <div className="stand-leader-stat">
              <div className="stand-leader-stat-num">{leader.wins}</div>
              <div className="stand-leader-stat-label">Wins</div>
            </div>
            <div className="stand-leader-stat">
              <div className="stand-leader-stat-num">{leader.position}</div>
              <div className="stand-leader-stat-label">Position</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Podium (top 3) ───────────────────────────────────────────── */}
      <section className="stand-section">
        <div className="stand-section-head">
          <span className="stand-section-label">Top 3</span>
        </div>
        <div className="stand-podium-row">
          {drivers.slice(0, 3).map((d) => {
            const meta = driversMeta[d.driver];
            const gap = d.position === 1 ? null : `+${(leader.points - d.points).toFixed(0)}`;
            return (
              <div key={d.driver} className={`stand-podium-card ${d.position === 1 ? "stand-podium-card--leader" : ""}`}>
                <span className="stand-podium-pos-badge">P{d.position}</span>
                {gap && <span className="stand-podium-gap">{gap}</span>}
                {meta && <CarWidget driver={meta} size="card" />}
                <div className="stand-podium-name">{d.full_name}</div>
                <div className="stand-podium-team">{d.constructor}</div>
                <div className="stand-podium-pts">{d.points}<span className="stand-podium-pts-unit"> pts</span></div>
                {d.position === 1 && <div className="stand-podium-leader-chip">Leader</div>}
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Driver / Constructor standings ───────────────────────────── */}
      <section className="stand-section">
        <div className="stand-two-col">
          <div>
            <div className="stand-section-head">
              <span className="stand-section-label">Driver Standings</span>
              <button className="stand-view-full" onClick={() => setShowAllDrivers((s) => !s)}>
                {showAllDrivers ? "Show Top 10 →" : "View Full →"}
              </button>
            </div>
            <div className="stand-list">
              {(showAllDrivers ? drivers : drivers.slice(0, 10)).map((d) => {
                const meta = driversMeta[d.driver];
                const prevPos = prevPosByDriver?.[d.driver];
                const trend = prevPos == null ? "flat" : prevPos > d.position ? "up" : prevPos < d.position ? "down" : "flat";
                const gap = d.position === 1 ? "–" : `+${(leader.points - d.points).toFixed(0)}`;
                return (
                  <div className="stand-list-row" key={d.driver}>
                    <span className="stand-list-pos">{d.position}</span>
                    {meta ? <CarWidget driver={meta} size="row-sm" /> : <span />}
                    <span className="stand-list-name-wrap">
                      <div className="stand-list-name">{d.full_name}</div>
                      <div className="stand-list-team">{d.constructor}</div>
                    </span>
                    <span className="stand-list-pts">{d.points}</span>
                    <span className={`stand-list-trend ${trend}`}>
                      {trend === "up" ? "▲" : trend === "down" ? "▼" : "–"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div>
            <div className="stand-section-head">
              <span className="stand-section-label">Constructor Standings</span>
              <button className="stand-view-full">View Full →</button>
            </div>
            <div className="stand-list">
              {constructors.map((c) => {
                const knownTeamNames = [...new Set(Object.values(driversMeta).map((d) => d.team))];
                const teamKey = resolveTeamKey(c.constructor, knownTeamNames);
                const teamCode = Object.values(driversMeta).find((d) => d.team === teamKey);
                const color = teamCode?.team_color || "var(--gray)";
                return (
                  <div className="stand-con-row" key={c.constructor}>
                    <span className="stand-list-pos">{c.position}</span>
                    {teamCode ? <CarWidget driver={teamCode} size="row-sm" /> : <span />}
                    <span className="stand-con-name-wrap"><div className="stand-con-name">{c.constructor}</div></span>
                    <span className="stand-con-pts">{c.points}</span>
                    <span className="stand-con-bar-track">
                      <span className="stand-con-bar-fill" style={{ width: `${(c.points / maxConstructorPts) * 100}%`, background: color }} />
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* ── Season progress + live stats ─────────────────────────────── */}
      <section className="stand-section">
        <div className="stand-stat-strip">
          <div className="stand-stat-card">
            <div className="stand-stat-label">Season Progress</div>
            <div className="stand-stat-value">{currentRoundNum} / {schedule.length || 24}</div>
            <div className="stand-stat-sub">Races Completed</div>
          </div>
          {nextRace && (
            <div className="stand-stat-card">
              <div className="stand-stat-label">Next Race</div>
              <div className="stand-stat-value" style={{ fontSize: "1.1rem" }}>{nextRace.gp_name}</div>
              <div className="stand-stat-sub">{nextRace.circuit}</div>
            </div>
          )}
          {biggestMovers?.gainer && (
            <div className="stand-stat-card">
              <div className="stand-stat-label">Biggest Gainer</div>
              <div className="stand-stat-value lime"><span className="stand-stat-arrow">▲</span>{biggestMovers.gainer.delta}</div>
              <div className="stand-stat-sub">{biggestMovers.gainer.driver.full_name}</div>
            </div>
          )}
          {biggestMovers?.loser && biggestMovers.loser.delta < 0 && (
            <div className="stand-stat-card">
              <div className="stand-stat-label">Biggest Loser</div>
              <div className="stand-stat-value red"><span className="stand-stat-arrow">▼</span>{Math.abs(biggestMovers.loser.delta)}</div>
              <div className="stand-stat-sub">{biggestMovers.loser.driver.full_name}</div>
            </div>
          )}
          {liveStats && (
            <div className="stand-stat-card">
              <div className="stand-stat-label">Largest Gap</div>
              <div className="stand-stat-value">{liveStats.largestGap.toFixed(0)} <span style={{ fontSize: "0.9rem", color: "var(--gray)" }}>pts</span></div>
              <div className="stand-stat-sub">{leader.driver} → {p2?.driver}</div>
            </div>
          )}
        </div>

        <div className="stand-section-head" style={{ marginTop: "2.5rem" }}>
          <span className="stand-section-label">Season Timeline</span>
        </div>
        <div className="stand-timeline">
          {schedule.map((r) => {
            const isDone = r.round < currentRoundNum;
            const isCurrent = r.round === currentRoundNum;
            const short = (r.gp_name || "").replace(" Grand Prix", "").slice(0, 3).toUpperCase();
            return (
              <div
                key={r.round}
                className={`stand-timeline-item ${isDone ? "done" : ""} ${isCurrent ? "current" : ""}`}
                onMouseEnter={() => isDone && handleHoverRound(r.round, r.gp_name)}
                onMouseLeave={() => setHoverRound(null)}
              >
                <span className="stand-timeline-line" />
                <span className="stand-timeline-dot" />
                <span className="stand-timeline-code">{short}</span>
                {hoverRound === r.round && (
                  <span className="stand-timeline-tooltip">
                    {roundWinners[r.round] === undefined ? "Loading…" : roundWinners[r.round] ? `Winner: ${roundWinners[r.round]}` : "Unavailable"}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Championship forecast (configurable Monte Carlo) ─────────── */}
      <section className="stand-section">
        <div className="stand-section-head">
          <span className="stand-section-label">Championship Forecast</span>
        </div>
        <div className="stand-forecast-sub">Monte Carlo simulation over the remaining {schedule.length - currentRoundNum} races.</div>

        <div className="stand-forecast-controls">
          <div className="stand-forecast-field">
            <div className="stand-field-label">Rounds Completed <span className="stand-field-note">(live)</span></div>
            <div className="stand-forecast-readout">{currentRoundNum} <span className="stand-forecast-readout-unit">/ {schedule.length || 24}</span></div>
          </div>

          <div className="stand-forecast-field">
            <div className="stand-field-label">
              Simulations
              <InfoTooltip>
                <p>
                  How many complete seasons the model plays out. More simulations give a more
                  stable probability estimate but take longer to compute — 5,000 is a good
                  balance; 10,000 is closer to the statistical ceiling for this model.
                </p>
              </InfoTooltip>
            </div>
            <select className="stand-forecast-select" value={numSimulations} onChange={(e) => setNumSimulations(Number(e.target.value))}>
              <option value={1000}>1,000</option>
              <option value={3000}>3,000</option>
              <option value={5000}>5,000</option>
              <option value={10000}>10,000</option>
            </select>
          </div>

          <div className="stand-forecast-field">
            <div className="stand-field-label">
              Upset Factor (Chaos Level)
              <InfoTooltip>
                <p>
                  How much random race-to-race variance gets added on top of each driver's
                  underlying pace. Low values mostly play out "as expected" based on current
                  form; higher values let midfield drivers occasionally out-score the favorites,
                  modeling messier, less predictable races.
                </p>
              </InfoTooltip>
            </div>
            <div className="stand-forecast-slider-row">
              <input type="range" min="0" max="0.5" step="0.01" value={chaosLevel} onChange={(e) => setChaosLevel(Number(e.target.value))} />
              <span className="stand-forecast-slider-val">{chaosLevel.toFixed(2)}</span>
            </div>
          </div>

          <div className="stand-forecast-field">
            <div className="stand-field-label">
              Safety Car Frequency
              <InfoTooltip>
                <p>
                  A multiplier on each circuit's baseline safety-car probability. 1.0 uses the
                  historical average for each track; higher values simulate a season with more
                  safety cars and virtual safety cars, which tend to bunch the field and create
                  more upset opportunities.
                </p>
              </InfoTooltip>
            </div>
            <div className="stand-forecast-slider-row">
              <input type="range" min="0" max="2" step="0.1" value={safetyCarFreq} onChange={(e) => setSafetyCarFreq(Number(e.target.value))} />
              <span className="stand-forecast-slider-val">{safetyCarFreq.toFixed(2)}</span>
            </div>
          </div>
        </div>

        <label className="stand-forecast-checkbox">
          <input type="checkbox" checked={simulateDNFs} onChange={(e) => setSimulateDNFs(e.target.checked)} />
          Simulate DNFs / retirements
          <InfoTooltip>
            <p>
              When enabled, each driver has a per-race chance of retiring based on their
              constructor's historical DNF rate, which removes them from the points for that
              simulated race. When disabled, every driver finishes every remaining race —
              a cleaner but less realistic scenario.
            </p>
          </InfoTooltip>
        </label>

        <button className="stand-forecast-run-btn" onClick={handleRunForecast} disabled={forecastRunning}>
          {forecastRunning ? "Running Simulation…" : "Run Championship Simulation"}
        </button>

        {/* ── Result: cinematic prediction card ───────────────────────── */}
        <div className="stand-predict-card" style={{ marginTop: "2rem" }}>
          <div>
            <div className="stand-predict-label">Model Prediction</div>
            {(simLoading || forecastRunning) ? (
              <div className="stand-loading">Running {numSimulations.toLocaleString()} season simulations…</div>
            ) : topSim ? (
              <>
                <div className="stand-predict-name">{driversMeta[topSim.driver]?.name || topSim.driver}</div>
                <div className="stand-predict-sub">to win the {YEAR} world championship</div>
              </>
            ) : (
              <div className="stand-loading">Simulation unavailable</div>
            )}
          </div>
          {driversMeta[topSim?.driver] && <CarWidget driver={driversMeta[topSim.driver]} size="card" />}
          {topSim && (
            <div>
              <div className="stand-predict-pct">{topSim.wdc_probability}%</div>
              <div className="stand-predict-pct-label">Probability</div>
            </div>
          )}
        </div>
      </section>

      {/* ── WDC / WCC probability ─────────────────────────────────────── */}
      {sim && (
        <section className="stand-section">
          <div className="stand-two-col">
            <div>
              <div className="stand-section-head"><span className="stand-section-label">WDC Probability</span></div>
              <div className="stand-list">
                {sim.results.slice(0, 6).map((r, i) => (
                  <div className="stand-prob-row" key={r.driver}>
                    <span className="stand-prob-pos">{i + 1}</span>
                    <span className="stand-prob-name">{driversMeta[r.driver]?.name || r.driver}</span>
                    <span className="stand-prob-pct">{r.wdc_probability}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="stand-section-head"><span className="stand-section-label">WCC Probability</span></div>
              <div className="stand-list">
                {wccApprox?.slice(0, 6).map((t, i) => (
                  <div className="stand-prob-row" key={t.team}>
                    <span className="stand-prob-pos">{i + 1}</span>
                    <span className="stand-prob-name">{t.team}</span>
                    <span className="stand-prob-pct">{t.pct.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div className="stand-approx-note">
                Approximated from per-driver simulation totals grouped by team — not a direct
                team-level Monte Carlo run. Ask if you'd like a true constructor simulation added
                to the backend.
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── Projected final standings ─────────────────────────────────── */}
      {projectedStandings && (
        <section className="stand-section">
          <div className="stand-section-head">
            <span className="stand-section-label">Projected Final Standings</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--gray-dim)" }}>
              Based on {numSimulations.toLocaleString()} simulations
            </span>
          </div>
          <div className="stand-proj-cols">
            <div>
              {projectedStandings.slice(0, 10).map((r, i, arr) => (
                <div className="stand-proj-row" key={r.driver}>
                  <span className="stand-proj-pos">{i + 1}</span>
                  <span className="stand-proj-name">{driversMeta[r.driver]?.name || r.driver}</span>
                  <span className="stand-proj-pts">{r.avg_final_points.toFixed(1)}</span>
                  <span className={`stand-proj-conf ${confidenceFor(r, i, arr)}`}>{confidenceFor(r, i, arr)}</span>
                </div>
              ))}
            </div>
            <div>
              {projectedStandings.slice(10, 20).map((r, i, arr) => (
                <div className="stand-proj-row" key={r.driver}>
                  <span className="stand-proj-pos">{i + 11}</span>
                  <span className="stand-proj-name">{driversMeta[r.driver]?.name || r.driver}</span>
                  <span className="stand-proj-pts">{r.avg_final_points.toFixed(1)}</span>
                  <span className={`stand-proj-conf ${confidenceFor(r, i, arr)}`}>{confidenceFor(r, i, arr)}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      <div className="stand-footer-note">
        <span className="stand-footer-dot" /> Live data powered by FastF1 &amp; Jolpica
      </div>
    </div>
  );
}
