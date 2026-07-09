import { useRef, useState, useEffect, useMemo } from "react";
import { motion, useScroll, useTransform, useMotionValue, useSpring } from "framer-motion";
import { getPredictorStatus, getDriverStandings, getSchedule } from "../api";
import CircuitMap from "./CircuitMap";
import "./Hero.css";

export default function Hero() {
  const heroRef = useRef(null);
  const circuitRef = useRef(null);

  // ── Scroll-shrink left panel ─────────────────────────────────────────
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const scale  = useTransform(scrollYProgress, [0, 1], [1, 0.72]);
  const opacity = useTransform(scrollYProgress, [0, 1], [1, 0]);

  // ── Mouse parallax on circuit panel ─────────────────────────────────
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const spx = useSpring(mx, { stiffness: 80, damping: 18 });
  const spy = useSpring(my, { stiffness: 80, damping: 18 });

  const handleMouseMove = (e) => {
    if (!circuitRef.current) return;
    const r = circuitRef.current.getBoundingClientRect();
    const nx = (e.clientX - r.left - r.width / 2) / (r.width / 2);
    const ny = (e.clientY - r.top - r.height / 2) / (r.height / 2);
    mx.set(nx * 18);
    my.set(ny * 14);
  };
  const handleMouseLeave = () => { mx.set(0); my.set(0); };

  // ── Live model status ────────────────────────────────────────────────
  const [status, setStatus] = useState(null);
  const [standings, setStandings] = useState([]);
  const [schedule, setSchedule] = useState([]);

  useEffect(() => {
    getPredictorStatus().then(setStatus).catch(() => {});
    getDriverStandings(2026).then(d => setStandings(d.slice(0, 3))).catch(() => {});
    getSchedule(2026)
      .then((rows) => setSchedule([...rows].sort((a, b) => a.round - b.round)))
      .catch(() => {});
  }, []);

  // ── Next race — the actual fix: this used to be a hardcoded Silverstone
  // path/labels that never updated once that race passed. Now derived live
  // from the schedule: the first race whose date hasn't happened yet, or
  // the final round if the whole season's already run.
  const nextRace = useMemo(() => {
    if (!schedule.length) return null;
    const now = new Date();
    const upcoming = schedule.find((r) => r.date && new Date(r.date) >= now);
    return upcoming || schedule[schedule.length - 1];
  }, [schedule]);

  const accuracy = status?.metadata?.accuracy;
  const isLive   = status?.model_exists && !status?.is_stale;

  const eyebrowText = nextRace
    ? `// ${nextRace.gp_name} · ${nextRace.circuit} · Rd ${nextRace.round} · ${
        nextRace.date
          ? new Date(nextRace.date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })
          : ""
      }`
    : "// Loading next race…";

  return (
    <section ref={heroRef} className="hero" id="hero">
      {/* ── LEFT: headline ── */}
      <motion.div
        className="hero-left"
        style={{ scale, opacity, transformOrigin: "left bottom" }}
      >
        <div className="hero-eyebrow">{eyebrowText}</div>

        <h1 className="hero-h1">
          <span className="ln-slide" style={{ animationDelay: "0.1s" }}>Race</span>
          <span className="ln-slide outline" style={{ animationDelay: "0.24s" }}>Mind</span>
          <span className="ln-slide red"    style={{ animationDelay: "0.38s" }}>AI</span>
        </h1>

        <div className="hero-rule" />

        <div className="hero-stats">
          <div>
            <div className="hstat-label">Model</div>
            <div className="hstat-val">XGBoost</div>
          </div>
          <div>
            <div className="hstat-label">Status</div>
            <div className={`hstat-val ${isLive ? "pulse" : ""}`}>
              {status === null ? "—" : isLive ? "Live" : "Stale"}
            </div>
          </div>
          <div>
            <div className="hstat-label">Accuracy</div>
            <div className="hstat-val lime">
              {accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : "…"}
            </div>
          </div>
        </div>

        {/* Live top-3 from API */}
        {standings.length > 0 && (
          <div className="hero-podium">
            {standings.map((d, i) => (
              <div key={d.driver} className="podium-row" style={{ animationDelay: `${1.4 + i * 0.1}s` }}>
                <span className="podium-pos">P{i + 1}</span>
                <span className="podium-name">{d.driver}</span>
                <span className="podium-pts">{d.points} pts</span>
              </div>
            ))}
          </div>
        )}
      </motion.div>

      {/* ── RIGHT: next race's circuit ── */}
      <div
        className="hero-right"
        ref={circuitRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Ambient glow behind circuit */}
        <div className="circuit-glow" />

        <motion.div
          className="circuit-wrap"
          style={{ x: spx, y: spy }}
        >
          {nextRace ? (
            <div className="circuit-frame">
              <CircuitMap circuitName={nextRace.circuit} />
            </div>
          ) : (
            <div className="circuit-loading">Loading circuit…</div>
          )}

          {/* ── Circuit info overlay ── */}
          <div className="circuit-info">
            <div className="ci-label">// Circuit</div>
            <div className="ci-name">{nextRace?.circuit || "—"}</div>
            <div className="ci-details">
              <span>{nextRace?.country || ""}</span>
            </div>
          </div>
        </motion.div>

        {/* ── Next race badge top-right ── */}
        <div className="hero-next-race">
          Next Race
          <strong>{nextRace?.gp_name || "—"}</strong>
          <div className="race-round">{nextRace ? `${nextRace.circuit} · Rd ${nextRace.round}` : ""}</div>
        </div>

        {/* ── Ghost text ── */}
        <div className="hero-right-text">
          <div className="big">CIRCUIT</div>
        </div>
      </div>
    </section>
  );
}
