import { useState, useEffect, useMemo } from "react";
import {
  getPredictorStatus, trainModel, getGridDefaults, getSchedule,
  predictRace, explainPrediction,
  getQualifyingStatus, trainQualifyingModel, predictQualifying,
} from "../api";
import StartingGrid from "./StartingGrid";
import "./Predictor.css";

const CIRCUIT_TYPE_LABEL = {
  high_downforce: "High Downforce", street: "Street Circuit",
  power: "Power Circuit", technical: "Technical", unknown: "Unclassified",
};

// Same podium arrangement (P2 / P1 / P3) as the homepage's finished-race
// cards. Block heights are fixed rem values (not percentages) so P2 is
// reliably exactly between P1 and P3 regardless of flex-item quirks.
const PODIUM_ORDER = [2, 1, 3];
const PODIUM_HEIGHT = { 1: "8.5rem", 2: "6.25rem", 3: "4rem" };

function PredictorPodium({ top3 }) {
  // top3: [{ driver: {code, name, number, team_color}, podium_probability }]
  const byPos = Object.fromEntries(top3.map((entry, i) => [i + 1, entry]));
  return (
    <div className="pred-podium-graphic">
      {PODIUM_ORDER.map((pos) => {
        const entry = byPos[pos];
        const color = entry?.driver?.team_color || "var(--gray)";
        return (
          <div key={pos} className="pred-podium-slot">
            {entry && (
              <>
                <div className="pred-podium-avatar" style={{ borderColor: color }}>
                  <span className="pred-podium-avatar-num" style={{ color }}>{entry.driver.number ?? "—"}</span>
                </div>
                <div className="pred-podium-driver">{entry.driver.name || entry.driver.code}</div>
                <div className="pred-podium-pct">{(entry.podium_probability * 100).toFixed(1)}%</div>
              </>
            )}
            <div
              className="pred-podium-block"
              style={{ height: PODIUM_HEIGHT[pos], background: `${color}22`, borderTop: `2px solid ${color}` }}
            >
              <span className="pred-podium-num">{pos}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StatusBanner({ label, status, error, training, onTrain }) {
  return (
    <div className="pred-status-row">
      <span className={`pred-status-dot ${status ? (status.is_stale ? "stale" : "ok") : ""}`} />
      <span className="pred-status-text">
        {error && <>{label} status unavailable — {error}</>}
        {!error && !status && `Checking ${label.toLowerCase()} model status…`}
        {!error && status && status.metadata && (
          <>
            {label} model trained <b>{status.metadata.trained_at?.slice(0, 10)}</b>
            {status.is_stale && " · over 7 days old"}
          </>
        )}
        {!error && status && !status.metadata && `No trained ${label.toLowerCase()} model yet.`}
      </span>
      <button className="pred-train-btn" onClick={onTrain} disabled={training}>
        {training ? "Training…" : "Train / Refresh"}
      </button>
    </div>
  );
}

// Small "?" info widget — click to reveal a plain-language explainer for
// the SHAP chart's features and values, without permanently taking up
// page space.
function InfoTooltip({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="pred-info-wrap">
      <button
        type="button"
        className="pred-info-btn"
        aria-label="What do these features mean?"
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open && <div className="pred-info-panel">{children}</div>}
    </span>
  );
}

export default function Predictor({ year = 2026 }) {
  const [schedule, setSchedule] = useState([]);
  const [selectedRound, setSelectedRound] = useState(null);
  const [weather, setWeather] = useState("Dry");
  const [drivers, setDrivers] = useState([]);

  const selectedRace = useMemo(
    () => schedule.find((r) => r.round === selectedRound),
    [schedule, selectedRound]
  );

  // ── Qualifying predictor ─────────────────────────────────────────
  const [qualiStatus, setQualiStatus] = useState(null);
  const [qualiStatusErr, setQualiStatusErr] = useState(null);
  const [qualiTraining, setQualiTraining] = useState(false);
  const [qualiLoading, setQualiLoading] = useState(false);
  const [qualiError, setQualiError] = useState(null);
  const [qualiResult, setQualiResult] = useState(null); // { circuit, circuit_type, predictions }

  // ── Race predictor ───────────────────────────────────────────────
  const [raceOrder, setRaceOrder] = useState([]); // array of driver codes, index 0 = P1
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);
  const [result, setResult] = useState(null);
  const [explainDriver, setExplainDriver] = useState(null);
  const [shapFactors, setShapFactors] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState(null);

  const [raceStatus, setRaceStatus] = useState(null);
  const [raceStatusErr, setRaceStatusErr] = useState(null);
  const [raceTraining, setRaceTraining] = useState(false);

  // ── Initial load ─────────────────────────────────────────────────
  useEffect(() => {
    getQualifyingStatus().then(setQualiStatus).catch((e) => setQualiStatusErr(e.message));
    getPredictorStatus().then(setRaceStatus).catch((e) => setRaceStatusErr(e.message));
    getSchedule(year)
      .then((rows) => {
        const sorted = [...rows].sort((a, b) => a.round - b.round);
        setSchedule(sorted);
        const now = new Date();
        const next = sorted.find((r) => !r.date || new Date(r.date) >= now);
        setSelectedRound((next || sorted[sorted.length - 1] || {}).round ?? null);
      })
      .catch(() => {});
    getGridDefaults(year)
      .then((list) => {
        setDrivers(list);
        setRaceOrder(list.map((d) => d.code));
      })
      .catch(() => {});
  }, [year]);

  const handleTrainQuali = async () => {
    setQualiTraining(true);
    try {
      await trainQualifyingModel(false);
      setQualiStatus(await getQualifyingStatus());
    } catch (e) { setQualiStatusErr(e.message); } finally { setQualiTraining(false); }
  };

  const handleTrainRace = async () => {
    setRaceTraining(true);
    try {
      await trainModel(false);
      setRaceStatus(await getPredictorStatus());
    } catch (e) { setRaceStatusErr(e.message); } finally { setRaceTraining(false); }
  };

  const handlePredictQuali = async () => {
    if (!selectedRound) return;
    setQualiLoading(true); setQualiError(null); setQualiResult(null);
    try {
      const res = await predictQualifying(year, selectedRound, weather);
      setQualiResult(res);
    } catch (e) {
      setQualiError(e.response?.data?.detail || e.message);
    } finally { setQualiLoading(false); }
  };

  const qualiOrder = useMemo(() => {
    if (!qualiResult) return null;
    return [...qualiResult.predictions]
      .sort((a, b) => a.predicted_quali_position - b.predicted_quali_position)
      .map((p) => p.driver);
  }, [qualiResult]);

  const useQualiAsGrid = () => {
    if (qualiOrder) setRaceOrder(qualiOrder);
  };

  const handlePredictRace = async () => {
    if (!selectedRound || raceOrder.length === 0) return;
    setPredicting(true); setPredictError(null); setResult(null); setShapFactors(null);
    try {
      const grid = raceOrder.map((code, i) => ({ driver: code, grid_position: i + 1 }));
      const res = await predictRace(year, selectedRound, grid);
      setResult(res);
      setExplainDriver(res.predictions?.[0]?.driver ?? null);
    } catch (e) {
      setPredictError(e.response?.data?.detail || e.message);
    } finally { setPredicting(false); }
  };

  useEffect(() => {
    if (!result || !explainDriver) return;
    setShapLoading(true); setShapError(null);
    explainPrediction(explainDriver, result._rows_used, 6)
      .then(setShapFactors)
      .catch((e) => setShapError(e.response?.data?.detail || e.message))
      .finally(() => setShapLoading(false));
  }, [explainDriver, result]);

  const driverByCode = useMemo(() => {
    const m = {}; drivers.forEach((d) => { m[d.code] = d; }); return m;
  }, [drivers]);

  const podiumTop3 = useMemo(() => {
    if (!result) return null;
    return [...result.predictions]
      .sort((a, b) => a.predicted_position - b.predicted_position)
      .slice(0, 3)
      .map((p) => ({ driver: driverByCode[p.driver] || { code: p.driver }, podium_probability: p.podium_probability }));
  }, [result, driverByCode]);

  const maxShapAbs = shapFactors ? Math.max(...shapFactors.map((f) => Math.abs(f.shap_value)), 0.001) : 1;
  const positiveFactors = (shapFactors || []).filter((f) => f.shap_value > 0);
  const negativeFactors = (shapFactors || []).filter((f) => f.shap_value < 0);

  return (
    <section className="pred-section" id="predictor">
      <div className="pred-label">Predictor</div>
      <h2 className="pred-heading">Grid To <span className="outline">Podium.</span></h2>
      <p className="pred-sub">
        Predict single-lap qualifying order, then feed it straight into the race model — or
        set your own grid by hand. Drag any car onto another box to swap positions.
      </p>

      {/* ── Shared race + circuit + weather selection, horizontal top bar ── */}
      <div className="pred-topbar">
        <div className="pred-topbar-field">
          <div className="pred-field-label">Select Race — {year}</div>
          <select
            className="pred-driver-select pred-topbar-select"
            value={selectedRound ?? ""}
            onChange={(e) => setSelectedRound(Number(e.target.value))}
          >
            {schedule.length === 0 && <option>Loading schedule…</option>}
            {schedule.map((r) => (
              <option key={r.round} value={r.round}>
                R{String(r.round).padStart(2, "0")} — {r.gp_name}
              </option>
            ))}
          </select>
        </div>

        <div className="pred-topbar-field">
          <div className="pred-field-label">Circuit</div>
          {selectedRace ? (
            <div className="pred-circuit-card">
              <div className="pred-circuit-name">{selectedRace.circuit}</div>
              <div className="pred-circuit-meta">
                {selectedRace.country}
                {qualiResult?.circuit_type && <> · {CIRCUIT_TYPE_LABEL[qualiResult.circuit_type] || qualiResult.circuit_type}</>}
              </div>
            </div>
          ) : (
            <div className="pred-circuit-card pred-circuit-card--empty">Select a race</div>
          )}
        </div>

        <div className="pred-topbar-field">
          <div className="pred-field-label">Weather</div>
          <select className="pred-driver-select pred-topbar-select" value={weather} onChange={(e) => setWeather(e.target.value)}>
            <option>Dry</option>
            <option>Wet</option>
            <option>Mixed</option>
          </select>
        </div>
      </div>

      {/* ── Qualifying predictor ─────────────────────────────────── */}
      <div className="pred-quali-block">
        <div className="pred-field-label" style={{ marginBottom: "0.7rem" }}>Qualifying Predictor</div>
        <StatusBanner label="Qualifying" status={qualiStatus} error={qualiStatusErr} training={qualiTraining} onTrain={handleTrainQuali} />

        <div className="pred-submit-row" style={{ marginTop: "1.4rem" }}>
          <button className="pred-predict-btn" onClick={handlePredictQuali} disabled={qualiLoading || !selectedRound}>
            {qualiLoading ? "Predicting…" : "Predict Qualifying Order"}
          </button>
          {qualiError && <span className="pred-predict-error">{qualiError}</span>}
        </div>

        {qualiOrder && drivers.length > 0 && (
          <div style={{ marginTop: "1.2rem" }}>
            <StartingGrid
              drivers={drivers}
              order={qualiOrder}
              onReorder={() => {}}
              editable={false}
              size="lg"
            />
            <button className="pred-train-btn" style={{ marginTop: "0.7rem" }} onClick={useQualiAsGrid}>
              Use This Grid For Race Predictor ↓
            </button>
          </div>
        )}
      </div>

      {/* ── Race predictor grid editor ──────────────────────────────── */}
      <div className="pred-results" style={{ marginTop: "4rem" }}>
        <div className="pred-results-label">Race Predictor</div>
        <h3 className="pred-results-heading">Set the grid, predict the podium.</h3>

        <StatusBanner label="Race" status={raceStatus} error={raceStatusErr} training={raceTraining} onTrain={handleTrainRace} />

        {drivers.length > 0 && raceOrder.length > 0 && (
          <div style={{ marginTop: "1rem" }}>
            <StartingGrid drivers={drivers} order={raceOrder} onReorder={setRaceOrder} editable size="lg" />
          </div>
        )}

        <div className="pred-submit-row">
          <button className="pred-predict-btn" onClick={handlePredictRace} disabled={predicting || !selectedRound || raceOrder.length === 0}>
            {predicting ? "Predicting…" : "Predict Podium"}
          </button>
          {predictError && <span className="pred-predict-error">{predictError}</span>}
        </div>

        {/* ── Podium output, same visual language as homepage results ── */}
        {podiumTop3 && (
          <div className="pred-podium-wrap">
            <div className="pred-podium-label">// Predicted Podium</div>
            <PredictorPodium top3={podiumTop3} />
          </div>
        )}

        {/* ── SHAP explainability ─────────────────────────────────── */}
        {result && (
          <div className="pred-shap">
            <div className="pred-shap-label">Why Did The Model Predict This?</div>
            <h3 className="pred-shap-heading">The reasoning, laid bare.</h3>
            <p className="pred-shap-desc">
              SHAP values show which features pushed a driver's podium probability up
              (lime) or down (red).
              <InfoTooltip>
                <p>
                  Each bar is one input feature the model used for this prediction. The
                  number next to it is that feature's SHAP value — how many percentage
                  points of podium probability it added (lime, right side) or subtracted
                  (red, left side) for this specific driver, in this specific race.
                </p>
                <p>
                  Common features you'll see: <b>grid_position</b> (qualifying slot — lower
                  is better), <b>driver_5race_avg_points</b> / <b>driver_5race_avg_pos</b>
                  (recent form), <b>constructor_pace_score</b> (how quick the team has been),
                  <b> circuit_type_code</b> (street / power / technical / high-downforce —
                  some drivers and cars suit some circuit types better), and
                  <b> constructor_dnf_rate</b> (how often the team retires).
                </p>
                <p>
                  A larger bar means that feature mattered more to this particular
                  prediction — it doesn't mean the feature is "good" or "bad" in general,
                  only that it moved this driver's number up or down for this race.
                </p>
              </InfoTooltip>
            </p>

            <select className="pred-driver-select" value={explainDriver || ""} onChange={(e) => setExplainDriver(e.target.value)}>
              {result.predictions.map((p) => (
                <option key={p.driver} value={p.driver}>{driverByCode[p.driver]?.name || p.driver}</option>
              ))}
            </select>

            {shapLoading && <div className="pred-loading">Computing SHAP values…</div>}
            {shapError && <div className="pred-predict-error">SHAP unavailable — {shapError}</div>}

            {shapFactors && !shapLoading && (
              <>
                <div className="pred-shap-chart">
                  {shapFactors.map((f) => {
                    const widthPct = (Math.abs(f.shap_value) / maxShapAbs) * 50;
                    const isPos = f.shap_value > 0;
                    return (
                      <div className="pred-shap-row" key={f.feature}>
                        <span className="pred-shap-feature">{f.feature}</span>
                        <span className="pred-shap-bar-wrap">
                          <span className="pred-shap-zero" />
                          <span className={`pred-shap-bar ${isPos ? "pos" : "neg"}`} style={{ width: `${widthPct}%` }} />
                          <span className="pred-shap-val" style={isPos ? { left: `calc(50% + ${widthPct}% + 0.5rem)` } : { right: `calc(50% + ${widthPct}% + 0.5rem)` }}>
                            {isPos ? "+" : ""}{f.shap_value.toFixed(3)}
                          </span>
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="pred-reasoning">
                  <div className="pred-reasoning-title">{driverByCode[explainDriver]?.name || explainDriver} — model reasoning</div>
                  {positiveFactors.length > 0 && (
                    <div className="pred-reasoning-line help">
                      <b>Helped by:</b> <span>{positiveFactors.slice(0, 3).map((f) => `${f.feature} (+${f.shap_value.toFixed(3)})`).join(", ")}</span>
                    </div>
                  )}
                  {negativeFactors.length > 0 && (
                    <div className="pred-reasoning-line hurt">
                      <b>Hurt by:</b> <span>{negativeFactors.slice(0, 2).map((f) => `${f.feature} (${f.shap_value.toFixed(3)})`).join(", ")}</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
