import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { getDriverDynamics, getGridDefaults, getDriverStandings, getSchedule } from "../api";
import CarWidget from "./CarWidget";
import RadarChart from "./RadarChart";
import "./DriverDynamics.css";

const YEAR = 2026;

const DIMENSIONS = ["street", "power", "technical", "high_downforce", "consistency", "race_craft"];
const DIM_LABELS = { street: "Street Circuits", power: "Power Tracks", technical: "Technical Circuits", high_downforce: "High Downforce", consistency: "Consistency", race_craft: "Race Craft" };
const DIM_ICON = { street: "🏙", power: "⚡", technical: "🔧", high_downforce: "🌀", consistency: "📊", race_craft: "🎯" };

const SERIES_COLORS = ["#E10600", "#00D4AA", "#7C4DFF", "#FF8C00", "#00B0FF", "#FF4081"];

const DEFAULT_CODES = ["VER", "NOR", "LEC", "RUS", "HAM"];

export default function DriverDynamics() {
  const [dna, setDna] = useState(null);
  const [driversMeta, setDriversMeta] = useState({});
  const [standings, setStandings] = useState(null);
  const [racesCompleted, setRacesCompleted] = useState(null);
  const [error, setError] = useState(null);

  const [selected, setSelected] = useState([]);
  const [scorecardView, setScorecardView] = useState("bars"); // "bars" | "numbers"
  const [addDriverValue, setAddDriverValue] = useState("");

  useEffect(() => {
    Promise.all([
      getDriverDynamics(),
      getGridDefaults(YEAR),
      getDriverStandings(YEAR),
      getSchedule(YEAR),
    ])
      .then(([dnaRows, meta, standingsRows, schedule]) => {
        const metaMap = {};
        meta.forEach((d) => { metaMap[d.code] = d; });
        setDriversMeta(metaMap);

        // driver_dna.py builds its dataset from historical results (2022-2026),
        // which includes drivers no longer on the grid (retired, swapped
        // teams out of F1, etc.). Filter down to just the current season's
        // roster so retired/former drivers never show up to compare.
        const currentGridDna = dnaRows.filter((r) => metaMap[r.driver]);
        setDna(currentGridDna);
        setStandings(standingsRows);

        const now = new Date();
        const past = [...schedule].filter((r) => r.date && new Date(r.date) < now);
        setRacesCompleted(past.length || 1);

        const available = currentGridDna.map((r) => r.driver);
        setSelected(DEFAULT_CODES.filter((c) => available.includes(c)).slice(0, 5));
      })
      .catch((e) => setError(e.response?.data?.detail || e.message));
  }, []);

  const dnaByDriver = useMemo(() => {
    if (!dna) return {};
    const m = {};
    dna.forEach((r) => { m[r.driver] = r; });
    return m;
  }, [dna]);

  const standingsByDriver = useMemo(() => {
    if (!standings) return {};
    const m = {};
    standings.forEach((s) => { m[s.driver] = s; });
    return m;
  }, [standings]);

  const toggleDriver = (code) => {
    setSelected((sel) => {
      if (sel.includes(code)) return sel.filter((c) => c !== code);
      if (sel.length >= 6) return sel; // capped per spec
      return [...sel, code];
    });
  };

  const availableToAdd = useMemo(() => {
    if (!dna) return [];
    return dna.map((r) => r.driver).filter((c) => !selected.includes(c));
  }, [dna, selected]);

  const radarAxes = DIMENSIONS.map((key) => ({ key, label: DIM_LABELS[key] }));
  const radarSeries = selected.map((code, i) => ({
    label: code,
    color: driversMeta[code]?.team_color || SERIES_COLORS[i % SERIES_COLORS.length],
    values: dnaByDriver[code] || {},
  }));

  // ── Quick insight cards ──────────────────────────────────────────────
  // Only built from metrics that legitimately exist (standings points/wins,
  // and the 6 real DNA dimensions) — computed across the SELECTED drivers.
  // Average Finish, Qualifying Gap, Podium Rate, and DNF Rate are NOT
  // available from any current endpoint, so they're intentionally left out
  // rather than filled with invented numbers.
  const insightCards = useMemo(() => {
    if (selected.length === 0 || !racesCompleted) return [];
    const withStats = selected.map((code) => {
      const st = standingsByDriver[code];
      const d = dnaByDriver[code];
      return {
        code,
        pointsPerRace: st ? st.points / racesCompleted : null,
        winRate: st ? (st.wins / racesCompleted) * 100 : null,
        raceCraft: d?.race_craft,
        consistency: d?.consistency,
        overall: d ? DIMENSIONS.reduce((a, k) => a + (d[k] || 0), 0) / DIMENSIONS.length : null,
      };
    });
    const best = (key) => withStats.reduce((a, b) => ((b[key] ?? -Infinity) > (a[key] ?? -Infinity) ? b : a));
    return [
      { label: "Points / Race", icon: "🏁", key: "pointsPerRace", pick: best("pointsPerRace"), fmt: (v) => v.toFixed(1) },
      { label: "Win Rate", icon: "🏆", key: "winRate", pick: best("winRate"), fmt: (v) => `${v.toFixed(0)}%` },
      { label: "Race Craft", icon: "🎯", key: "raceCraft", pick: best("raceCraft"), fmt: (v) => v.toFixed(0) },
      { label: "Consistency", icon: "📊", key: "consistency", pick: best("consistency"), fmt: (v) => v.toFixed(0) },
      { label: "Overall Rating", icon: "⭐", key: "overall", pick: best("overall"), fmt: (v) => v.toFixed(0) },
    ];
  }, [selected, standingsByDriver, dnaByDriver, racesCompleted]);

  const statusBadge = (value) => {
    if (value >= 75) return { label: "Best", cls: "best" };
    if (value >= 50) return { label: "Average", cls: "average" };
    return { label: "Needs Work", cls: "needs-work" };
  };

  if (error) return <div className="dyn-error">Couldn't load driver dynamics — {error}</div>;
  if (!dna) return <div className="dyn-loading">Loading driver dynamics…</div>;

  return (
    <div className="dyn-page">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="dyn-hero">
        <div>
          <h1 className="dyn-hero-heading">Driver<br /><span className="outline">Insights</span></h1>
          <p className="dyn-hero-sub">Deep dive into driver performance, strengths, weaknesses and race characteristics.</p>
        </div>
        <div className="dyn-season-select">{YEAR} ▾</div>
      </section>

      {/* ── Compare drivers + radar (single stacked card) ─────────────── */}
      <section className="dyn-section">
        <motion.div className="dyn-card dyn-compare-card" initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.5 }}>
          <div className="dyn-card-label">Compare Drivers</div>

          <div className="dyn-pills">
            {selected.map((code, i) => (
              <span key={code} className="dyn-pill" style={{ "--pill-color": driversMeta[code]?.team_color || SERIES_COLORS[i % SERIES_COLORS.length] }}>
                {code}
                <button onClick={() => toggleDriver(code)} aria-label={`Remove ${code}`}>×</button>
              </span>
            ))}
            {selected.length === 0 && <span className="dyn-pills-empty">No drivers selected</span>}
          </div>

          {selected.length < 6 && (
            <select
              className="dyn-add-select"
              value={addDriverValue}
              onChange={(e) => { toggleDriver(e.target.value); setAddDriverValue(""); }}
            >
              <option value="">+ Add driver ({selected.length}/6)</option>
              {availableToAdd.map((code) => (
                <option key={code} value={code}>{driversMeta[code]?.name || code}</option>
              ))}
            </select>
          )}

          <div className="dyn-helper-text">All statistics are season averages.</div>

          <div className="dyn-radar-divider" />

          <div className="dyn-card-label">Performance Radar</div>
          {selected.length === 0 ? (
            <div className="dyn-empty-state">Select at least one driver to see their profile.</div>
          ) : (
            <div className="dyn-radar-layout">
              <RadarChart axes={radarAxes} series={radarSeries} />
              <div className="dyn-radar-legend">
                {radarSeries.map((s) => {
                  const avg = radarAxes.reduce((a, ax) => a + (s.values[ax.key] || 0), 0) / radarAxes.length;
                  return (
                    <div key={s.label} className="dyn-legend-row">
                      <span className="dyn-legend-dot" style={{ background: s.color }} />
                      <span className="dyn-legend-name">{s.label}</span>
                      <span className="dyn-legend-avg">{avg.toFixed(0)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </motion.div>
      </section>

      {/* ── Quick insight cards ───────────────────────────────────────── */}
      <section className="dyn-section">
        <div className="dyn-card-label">Quick Insights</div>
        {insightCards.length === 0 ? (
          <div className="dyn-empty-state">Select drivers above to see quick insights.</div>
        ) : (
          <div className="dyn-insight-grid">
            {insightCards.map((c) => {
              const rawValue = c.pick[c.key];
              const badge = statusBadge(rawValue ?? 0);
              return (
                <motion.div key={c.label} className="dyn-insight-card" whileHover={{ y: -4 }}>
                  <div className="dyn-insight-icon">{c.icon}</div>
                  <div className="dyn-insight-value">{rawValue != null ? c.fmt(rawValue) : "—"}</div>
                  <div className="dyn-insight-label">{c.label}</div>
                  <div className="dyn-insight-driver">{driversMeta[c.pick.code]?.name || c.pick.code}</div>
                  <span className={`dyn-badge ${badge.cls}`}>{badge.label}</span>
                </motion.div>
              );
            })}
          </div>
        )}
      </section>

      {/* ── Driver scorecards ─────────────────────────────────────────── */}
      <section className="dyn-section">
        <div className="dyn-section-head">
          <span className="dyn-card-label">Driver Scorecards</span>
          <div className="dyn-toggle-pair">
            <button className={scorecardView === "bars" ? "active" : ""} onClick={() => setScorecardView("bars")}>Bars</button>
            <button className={scorecardView === "numbers" ? "active" : ""} onClick={() => setScorecardView("numbers")}>Numbers</button>
          </div>
        </div>

        {selected.length === 0 ? (
          <div className="dyn-empty-state">Select drivers above to see scorecards.</div>
        ) : (
          <div className="dyn-scorecard-grid">
            {selected.map((code, i) => {
              const d = dnaByDriver[code];
              if (!d) return null;
              const meta = driversMeta[code];
              const color = meta?.team_color || SERIES_COLORS[i % SERIES_COLORS.length];
              const overall = DIMENSIONS.reduce((a, k) => a + (d[k] || 0), 0) / DIMENSIONS.length;
              const sorted = [...DIMENSIONS].sort((a, b) => d[b] - d[a]);
              const strengths = sorted.slice(0, 2);
              const weaknesses = sorted.slice(-2);
              return (
                <motion.div
                  key={code}
                  className="dyn-scorecard"
                  style={{ "--card-color": color }}
                  whileHover={{ y: -5, boxShadow: `0 12px 30px ${color}22` }}
                >
                  <div className="dyn-scorecard-head">
                    {meta && <CarWidget driver={meta} size="row-sm" />}
                    <div>
                      <div className="dyn-scorecard-code">{code}</div>
                      <div className="dyn-scorecard-team">{meta?.team}</div>
                    </div>
                    <div className="dyn-scorecard-score">{overall.toFixed(0)}</div>
                  </div>

                  <div className="dyn-scorecard-bars">
                    {DIMENSIONS.map((dim) => (
                      <div key={dim} className="dyn-bar-row">
                        <span className="dyn-bar-label">{DIM_ICON[dim]} {DIM_LABELS[dim]}</span>
                        {scorecardView === "bars" ? (
                          <span className="dyn-bar-track">
                            <span className="dyn-bar-fill" style={{ width: `${d[dim]}%`, background: color }} />
                          </span>
                        ) : (
                          <span className="dyn-bar-number">{d[dim].toFixed(0)}</span>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="dyn-scorecard-profile">
                    <div className="dyn-profile-col">
                      <div className="dyn-profile-label">Strengths</div>
                      {strengths.map((dim) => <div key={dim} className="dyn-profile-item strength">● {DIM_LABELS[dim]}</div>)}
                    </div>
                    <div className="dyn-profile-col">
                      <div className="dyn-profile-label">Weaknesses</div>
                      {weaknesses.map((dim) => <div key={dim} className="dyn-profile-item weakness">● {DIM_LABELS[dim]}</div>)}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </section>

      {/* ── All drivers table (current grid only — dna is pre-filtered) ── */}
      <section className="dyn-section">
        <div className="dyn-card-label">All Drivers</div>
        <div className="dyn-table-note">
          Average Finish, Qualifying Gap, Podiums, DNF %, and Fastest Laps aren't available from
          the current API (only points/wins and the 6 DNA dimensions are) — those columns aren't
          shown rather than filled with placeholder numbers.
        </div>
        <div className="dyn-table-wrap">
          <table className="dyn-table">
            <thead>
              <tr>
                <th>Driver</th><th>Team</th><th>Points</th><th>Wins</th><th>Pts / Race</th>
                {DIMENSIONS.map((dim) => <th key={dim}>{DIM_LABELS[dim]}</th>)}
              </tr>
            </thead>
            <tbody>
              {dna.map((row, i) => {
                const st = standingsByDriver[row.driver];
                const meta = driversMeta[row.driver];
                return (
                  <tr key={row.driver} className={i % 2 === 1 ? "zebra" : ""}>
                    <td className="dyn-table-driver">
                      {meta && <CarWidget driver={meta} size="row-sm" />}
                      {meta?.name || row.driver}
                    </td>
                    <td>{meta?.team || "—"}</td>
                    <td>{st?.points ?? "—"}</td>
                    <td>{st?.wins ?? "—"}</td>
                    <td>{st && racesCompleted ? (st.points / racesCompleted).toFixed(1) : "—"}</td>
                    {DIMENSIONS.map((dim) => <td key={dim}>{row[dim]?.toFixed(0)}</td>)}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
