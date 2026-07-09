import { useRef, useEffect, useState, useLayoutEffect } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { getSchedule, getRaceResults } from "../api";
import CircuitMap from "./CircuitMap";
import "./RaceRail.css";

const TEAM_COLORS = {
  "McLaren": "#FF8000", "Ferrari": "#E8002D", "Red Bull": "#3671C6",
  "Mercedes": "#27F4D2", "Williams": "#64C4FF", "Aston Martin": "#006F62",
  "Alpine": "#0090FF", "Racing Bulls": "#6692FF", "Haas": "#B6BABD",
  "Audi": "#C0C0C0", "Cadillac": "#9C8E55",
};

// Fallback mock data while API loads / if it fails entirely
const MOCK_RACES = [
  { round: 13, raceName: "British Grand Prix",   circuit: "Silverstone Circuit",          date: "2026-07-06" },
  { round: 14, raceName: "Hungarian Grand Prix", circuit: "Hungaroring",                  date: "2026-08-03" },
  { round: 15, raceName: "Belgian Grand Prix",   circuit: "Circuit de Spa-Francorchamps", date: "2026-08-31" },
  { round: 16, raceName: "Dutch Grand Prix",     circuit: "Circuit Zandvoort",            date: "2026-09-07" },
  { round: 17, raceName: "Italian Grand Prix",   circuit: "Autodromo Nazionale di Monza", date: "2026-09-14" },
  { round: 18, raceName: "Singapore Grand Prix", circuit: "Marina Bay Street Circuit",    date: "2026-09-21" },
];

export default function RaceRail({ year = 2026 }) {
  const sectionRef   = useRef(null);
  const trackRef     = useRef(null);
  const overflowRef  = useRef(null);
  const [races, setRaces] = useState(MOCK_RACES);
  const [usingFallback, setUsingFallback] = useState(true);
  const [maxShift, setMaxShift] = useState(0); // px, negative or 0

  useEffect(() => {
    getSchedule(year)
      .then(data => {
        if (!data?.length) return;
        // Full season, in round order — not trimmed. Podium vs. upcoming
        // circuit art is decided per-card based on each race's own date.
        setRaces([...data].sort((a, b) => a.round - b.round));
        setUsingFallback(false);
      })
      .catch(err => {
        console.error("RaceRail: failed to load /api/schedule, showing fallback data:", err);
      });
  }, [year]);

  // Measure actual track/viewport width so the horizontal scroll distance
  // always matches however many cards are currently rendered — a fixed
  // percentage (e.g. -62%) only worked for the old 6-8 card mock list and
  // breaks (stops early, or starts mid-way) once the real 24-race season
  // loads.
  useLayoutEffect(() => {
    const measure = () => {
      if (!trackRef.current || !overflowRef.current) return;
      const trackWidth = trackRef.current.scrollWidth;
      const viewWidth = overflowRef.current.clientWidth;
      setMaxShift(Math.min(0, -(trackWidth - viewWidth)));
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [races]);

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });

  const x = useTransform(scrollYProgress, [0, 1], [0, maxShift]);

  // Scroll distance scales with card count so scroll speed feels consistent
  // whether there are 6 cards (mock) or 24 (full season).
  const sectionHeightVh = Math.max(250, Math.min(races.length * 42, 750));

  return (
    <section ref={sectionRef} className="rail-section" id="calendar" style={{ height: `${sectionHeightVh}vh` }}>
      <div className="rail-sticky">
        {/* Fixed left label */}
        <div className="rail-label-col">
          <div className="rail-label">// 2026 Calendar</div>
          <h2 className="rail-heading">
            Race<br />
            <span className="outline">Schedule</span>
          </h2>
          <p className="rail-sub">
            Full 2026 season. Scroll to browse past results and upcoming rounds.
          </p>
          {usingFallback && (
            <p className="rail-sub" style={{ color: "var(--red)", opacity: 0.8 }}>
              ⚠ Live schedule unavailable — showing placeholder data.
            </p>
          )}
          <div className="rail-progress-wrap">
            <motion.div className="rail-progress-fill" style={{ scaleX: scrollYProgress, transformOrigin: "left" }} />
          </div>
        </div>

        {/* Horizontal track */}
        <div className="rail-overflow" ref={overflowRef}>
          <motion.div className="rail-track" ref={trackRef} style={{ x }}>
            {races.map((race, i) => (
              <RaceCard key={race.round ?? i} race={race} index={i} year={year} />
            ))}
            <div className="rail-spacer" />
          </motion.div>
        </div>
      </div>
    </section>
  );
}

// ── Podium graphic (P2 / P1 / P3 blocks, classic podium arrangement) ──────
const PODIUM_ORDER = [2, 1, 3];
const PODIUM_HEIGHT = { 1: "78%", 2: "56%", 3: "38%" };

function PodiumGraphic({ results }) {
  const byPos = Object.fromEntries(results.map(r => [Number(r.Position), r]));
  return (
    <div className="podium-graphic">
      {PODIUM_ORDER.map((pos) => {
        const r = byPos[pos];
        const color = r ? (TEAM_COLORS[r.TeamName] ?? "var(--gray)") : "#222";
        return (
          <div key={pos} className="podium-slot">
            {r && (
              <>
                <div className="podium-avatar" style={{ borderColor: color }}>
                  <span className="podium-avatar-num" style={{ color }}>{r.DriverNumber ?? "—"}</span>
                </div>
                <div className="podium-driver">{r.Abbreviation ?? r.FullName}</div>
              </>
            )}
            <div
              className="podium-block"
              style={{ height: PODIUM_HEIGHT[pos], background: `${color}22`, borderTop: `2px solid ${color}` }}
            >
              <span className="podium-num">{pos}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function RaceCard({ race, index, year }) {
  // API schedule rows use "gp_name" (Ergast raceName); mock fallback data
  // uses "raceName" — support both so this works before/after the API loads.
  const gpName = race.gp_name ?? race.raceName ?? `Round ${race.round}`;
  const short = gpName.replace(" Grand Prix", " GP");
  const country = race.circuit?.split(" ").slice(-2).join(" ") ?? "";
  const dateStr = race.date
    ? new Date(race.date).toLocaleDateString("en-GB", { day: "numeric", month: "short" })
    : "";
  const color = Object.values(TEAM_COLORS)[index % Object.keys(TEAM_COLORS).length];

  const isPast = race.date ? new Date(race.date) < new Date() : false;

  // Only start fetching once a card has actually scrolled near the
  // viewport — for past races this avoids firing 10+ uncached FastF1
  // session loads at once; for upcoming races it avoids firing 10+ ML
  // predictions (each of which rebuilds features from historical results)
  // at once. Either way, staggering real demand instead of a thundering
  // herd on mount.
  const cardRef = useRef(null);
  const [isNearView, setIsNearView] = useState(false);
  useEffect(() => {
    if (isNearView || !cardRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setIsNearView(true);
          observer.disconnect();
        }
      },
      { root: null, rootMargin: "600px", threshold: 0.01 }
    );
    observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, [isNearView]);

  // null = not fetched yet / loading, [] = fetch completed but no podium
  // data (a genuine failure — never silently swapped back to circuit art,
  // since that looks identical to an upcoming race and hides the problem).
  const [podium, setPodium] = useState(null);
  useEffect(() => {
    if (!isPast || !isNearView || !gpName) return;
    let cancelled = false;
    setPodium(null);
    getRaceResults(year, gpName)
      .then((results) => {
        if (cancelled || !Array.isArray(results)) return;
        const top3 = results
          .filter((r) => [1, 2, 3].includes(Number(r.Position)))
          .sort((a, b) => Number(a.Position) - Number(b.Position));
        setPodium(top3);
      })
      .catch((err) => {
        console.error(`RaceRail: failed to load results for "${gpName}" (${year}):`, err);
        if (!cancelled) setPodium([]);
      });
    return () => { cancelled = true; };
  }, [isPast, isNearView, gpName, year]);

  return (
    <div className="race-card" ref={cardRef}>
      <div className="rc-stripe" style={{ background: color }} />
      <div className="rc-num">Rd {race.round ?? index + 1}</div>

      <div className="rc-visual">
        {!isPast ? (
          <CircuitMap circuitName={race.circuit} mini />
        ) : podium === null ? (
          <div className="rc-podium-loading">Loading results…</div>
        ) : podium.length === 0 ? (
          <div className="rc-results-empty">
            <CircuitMap circuitName={race.circuit} mini />
            <span className="rc-results-empty-label">Results unavailable</span>
          </div>
        ) : (
          <>
            <div className="rc-podium-label">// Podium</div>
            <PodiumGraphic results={podium} />
          </>
        )}
      </div>

      <div className="rc-body">
        <div className="rc-tag">{dateStr}</div>
        <div className="rc-name">{short}</div>
        <div className="rc-circuit">{race.circuit ?? country}</div>
      </div>
    </div>
  );
}
