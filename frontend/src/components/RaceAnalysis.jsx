import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { getSchedule, getRaceLaps, getRaceResults } from "../api";
import CircuitMap from "./CircuitMap";
import LapTimeChart from "./LapTimeChart";
import TireStrategyChart, { COMPOUND_COLORS } from "./TireStrategyChart";
import "./RaceAnalysis.css";

const YEAR = 2026;

const SESSION_TYPES = [
  { value: "R", label: "Race" },
  { value: "Q", label: "Qualifying" },
  { value: "FP1", label: "Practice 1" },
  { value: "FP2", label: "Practice 2" },
  { value: "FP3", label: "Practice 3" },
];

const LINE_COLORS = ["#e10600", "#ff8c00", "#00d4aa", "#7c4dff", "#00b0ff", "#ff4081", "#69f0ae", "#ffeb3b", "#40c4ff", "#ff6d00"];

// Ports GP_MAP from the existing Streamlit app.py exactly, so the FastF1
// `gp` param matches what the backend actually expects — same mapping,
// same fallback (strip " Grand Prix") for anything not listed.
const GP_MAP = {
  "Australian Grand Prix": "Australia", "Chinese Grand Prix": "China", "Japanese Grand Prix": "Japan",
  "Miami Grand Prix": "Miami", "Canadian Grand Prix": "Canada", "Monaco Grand Prix": "Monaco",
  "Barcelona Grand Prix": "Spain", "Austrian Grand Prix": "Austria", "British Grand Prix": "Great Britain",
  "Belgian Grand Prix": "Belgium", "Hungarian Grand Prix": "Hungary", "Dutch Grand Prix": "Netherlands",
  "Italian Grand Prix": "Monza", "Spanish Grand Prix": "Spain", "Azerbaijan Grand Prix": "Azerbaijan",
  "Singapore Grand Prix": "Singapore", "United States Grand Prix": "United States", "Mexico City Grand Prix": "Mexico",
  "São Paulo Grand Prix": "São Paulo", "Brazilian Grand Prix": "São Paulo", "Las Vegas Grand Prix": "Las Vegas",
  "Qatar Grand Prix": "Qatar", "Abu Dhabi Grand Prix": "Abu Dhabi", "Bahrain Grand Prix": "Bahrain",
  "Saudi Arabian Grand Prix": "Saudi Arabia", "Emilia Romagna Grand Prix": "Emilia Romagna",
  "Emilia-Romagna Grand Prix": "Emilia Romagna",
};
const gpKeyFor = (gpName) => GP_MAP[gpName] || gpName.replace(" Grand Prix", "");

function formatLapTime(seconds) {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(3).padStart(6, "0");
  return `${m}:${s}`;
}

export default function RaceAnalysis() {
  const [schedule, setSchedule] = useState([]);
  const [selectedGp, setSelectedGp] = useState("");
  const [sessionType, setSessionType] = useState("R");

  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [laps, setLaps] = useState(null);
  const [results, setResults] = useState(null);

  const [selectedDrivers, setSelectedDrivers] = useState([]);
  const [addDriverValue, setAddDriverValue] = useState("");

  useEffect(() => {
    getSchedule(YEAR)
      .then((rows) => {
        const sorted = [...rows].sort((a, b) => a.round - b.round);
        setSchedule(sorted);
        if (sorted.length) setSelectedGp(sorted[0].gp_name);
      })
      .catch(() => {});
  }, []);

  const selectedRace = useMemo(
    () => schedule.find((r) => r.gp_name === selectedGp),
    [schedule, selectedGp]
  );

  const handleLoadSession = async () => {
    if (!selectedGp) return;
    setLoading(true);
    setLoadError(null);
    setLaps(null);
    setResults(null);
    try {
      const gpKey = gpKeyFor(selectedGp);
      const [lapsData, resultsData] = await Promise.all([
        getRaceLaps(YEAR, gpKey, sessionType),
        getRaceResults(YEAR, gpKey),
      ]);
      setLaps(lapsData);
      setResults(resultsData);
      const avail = [...new Set(lapsData.map((l) => l.Driver))].sort();
      setSelectedDrivers(avail.slice(0, 5));
    } catch (e) {
      setLoadError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const driverColors = useMemo(() => {
    if (!laps) return {};
    const avail = [...new Set(laps.map((l) => l.Driver))].sort();
    const m = {};
    avail.forEach((code, i) => { m[code] = LINE_COLORS[i % LINE_COLORS.length]; });
    return m;
  }, [laps]);

  const availableDrivers = useMemo(() => {
    if (!laps) return [];
    return [...new Set(laps.map((l) => l.Driver))].sort();
  }, [laps]);

  const availableToAdd = availableDrivers.filter((c) => !selectedDrivers.includes(c));

  const toggleDriver = (code) => {
    setSelectedDrivers((sel) => (sel.includes(code) ? sel.filter((c) => c !== code) : [...sel, code]));
  };

  const summary = useMemo(() => {
    if (!laps || laps.length === 0) return null;
    const valid = laps.filter((l) => l.LapTimeSeconds > 0);
    const fastest = valid.reduce((a, b) => (b.LapTimeSeconds < a.LapTimeSeconds ? b : a), valid[0]);
    const driverCount = new Set(laps.map((l) => l.Driver)).size;
    return { totalLaps: laps.length, fastest, driverCount };
  }, [laps]);

  return (
    <div className="race-page">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="race-hero">
        <div>
          <h1 className="race-hero-heading">Race<br /><span className="outline">Analysis</span></h1>
          <p className="race-hero-sub">Deep dive into lap pace, tyre strategy and race performance.</p>
        </div>

        {selectedRace && (
          <div className="race-info-card">
            <div className="race-info-circuit">
              <CircuitMap circuitName={selectedRace.circuit} mini />
            </div>
            <div className="race-info-gp">{selectedRace.gp_name}</div>
            <div className="race-info-meta">
              Round {selectedRace.round} · {selectedRace.circuit}
            </div>
            <div className="race-info-country">{selectedRace.country}</div>
          </div>
        )}
      </section>

      {/* ── Session selection ────────────────────────────────────────── */}
      <section className="race-section">
        <motion.div className="race-panel-card race-session-card" initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.5 }}>
          <div className="race-session-fields">
            <div className="race-field">
              <div className="race-field-label">Grand Prix</div>
              <select className="race-select" value={selectedGp} onChange={(e) => setSelectedGp(e.target.value)}>
                {schedule.map((r) => <option key={r.round} value={r.gp_name}>{r.gp_name}</option>)}
              </select>
            </div>
            <div className="race-field">
              <div className="race-field-label">Session</div>
              <select className="race-select" value={sessionType} onChange={(e) => setSessionType(e.target.value)}>
                {SESSION_TYPES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </div>
            <button className="race-load-btn" onClick={handleLoadSession} disabled={loading || !selectedGp}>
              {loading ? "Loading…" : "Load Session"}
            </button>
          </div>

          {loadError && <div className="race-status-card error">Could not load session — {loadError}</div>}
          {laps && !loading && (
            <div className="race-status-card success">
              <span className="race-status-check">✓</span>
              Loaded {laps.length} laps
              <span className="race-status-sub">{selectedGp}</span>
            </div>
          )}
        </motion.div>
      </section>

      {laps && summary && (
        <>
          {/* ── Race summary ─────────────────────────────────────────── */}
          <section className="race-section">
            <div className="race-summary-grid">
              <motion.div className="race-stat-card" whileHover={{ y: -4 }}>
                <div className="race-stat-icon">🏁</div>
                <div className="race-stat-value">{summary.totalLaps}</div>
                <div className="race-stat-sub">Total Laps — across all drivers</div>
              </motion.div>
              <motion.div className="race-stat-card" whileHover={{ y: -4 }}>
                <div className="race-stat-icon">⏱</div>
                <div className="race-stat-value">{formatLapTime(summary.fastest?.LapTimeSeconds)}</div>
                <div className="race-stat-sub">Fastest Lap — {summary.fastest?.Driver}</div>
              </motion.div>
              <motion.div className="race-stat-card" whileHover={{ y: -4 }}>
                <div className="race-stat-icon">👤</div>
                <div className="race-stat-value">{summary.driverCount}</div>
                <div className="race-stat-sub">Drivers in session</div>
              </motion.div>
            </div>
          </section>

          {/* ── Lap Time Evolution ───────────────────────────────────── */}
          <section className="race-section">
            <div className="race-section-label">Lap Time Evolution</div>
            <motion.div className="race-panel-card" initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.5 }}>
              <div className="race-pills">
                {selectedDrivers.map((code) => (
                  <span key={code} className="race-pill" style={{ "--pill-color": driverColors[code] }}>
                    {code}
                    <button onClick={() => toggleDriver(code)} aria-label={`Remove ${code}`}>×</button>
                  </span>
                ))}
              </div>
              {availableToAdd.length > 0 && (
                <select className="race-add-select" value={addDriverValue} onChange={(e) => { toggleDriver(e.target.value); setAddDriverValue(""); }}>
                  <option value="">+ Add driver</option>
                  {availableToAdd.map((code) => <option key={code} value={code}>{code}</option>)}
                </select>
              )}
              <div className="race-helper-text">Compare up to multiple drivers.</div>

              {selectedDrivers.length > 0 ? (
                <div className="race-chart-wrap">
                  <LapTimeChart laps={laps} drivers={selectedDrivers} colors={driverColors} />
                </div>
              ) : (
                <div className="race-empty-state">Select at least one driver to see lap times.</div>
              )}
            </motion.div>
          </section>

          {/* ── Tire Strategy ─────────────────────────────────────────── */}
          <section className="race-section">
            <div className="race-section-label">Tire Strategy</div>
            <p className="race-section-desc">Which compound each driver ran, lap by lap.</p>
            <motion.div className="race-panel-card" initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.5 }}>
              <div className="race-chart-wrap">
                <TireStrategyChart laps={laps} drivers={availableDrivers} />
              </div>
              <div className="race-compound-legend">
                {Object.entries(COMPOUND_COLORS).map(([name, color]) => (
                  <span key={name} className="race-compound-chip">
                    <span className="race-compound-dot" style={{ background: color }} />
                    {name}
                  </span>
                ))}
              </div>
            </motion.div>
          </section>

          {/* ── Race Results ──────────────────────────────────────────── */}
          {sessionType === "R" && results && (
            <section className="race-section">
              <div className="race-section-label">Race Results</div>
              <div className="race-table-wrap">
                <table className="race-table">
                  <thead>
                    <tr>
                      <th>Pos</th><th>Driver</th><th>Team</th><th>Grid</th><th>Points</th><th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={r.Abbreviation || i} className={i % 2 === 1 ? "zebra" : ""}>
                        <td className="race-table-pos">{r.Position}</td>
                        <td className="race-table-driver">{r.Abbreviation}</td>
                        <td>{r.TeamName}</td>
                        <td>{r.GridPosition}</td>
                        <td>{r.Points}</td>
                        <td>
                          <span className={`race-status-badge ${r.Status === "Finished" ? "finished" : "dnf"}`}>
                            {r.Status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
