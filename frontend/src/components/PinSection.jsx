import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import windTunnelBg from "../assets/wind-tunnel-bg.png";
import "./PinSection.css";

const BEATS = [
  {
    tag: "// Race Predictor",
    lines: [
      { text: "Set the grid.", size: "md" },
      { text: "Predict the podium.", size: "xl", accent: true },
      { text: "See why.", size: "sm" },
    ],
    sub: "Set a hypothetical qualifying grid and get live podium probabilities from our XGBoost model — with a SHAP breakdown showing exactly which features pushed each driver's odds up or down.",
    accent: "var(--red)",
    cta: { label: "Open Predictor", href: "#predictor" },
    in: [0, 0.09],
    out: [0.30, 0.40],
  },
  {
    tag: "// Standings",
    lines: [
      { text: "Driver points.", size: "md" },
      { text: "Championship race.", size: "xl", accent: true },
      { text: "Constructor battle.", size: "sm" },
    ],
    sub: "Live 2026 driver and constructor standings, updated after every session — see exactly who's leading the title fight and by how much.",
    accent: "var(--lime)",
    cta: { label: "View Standings", href: "#standings" },
    in: [0.28, 0.37],
    out: [0.60, 0.70],
  },
  {
    tag: "// Driver Dynamics",
    lines: [
      { text: "Race craft.", size: "sm" },
      { text: "Driver DNA.", size: "xl", accent: true },
      { text: "Circuit strengths.", size: "md" },
    ],
    sub: "A per-driver breakdown of street, power, and technical circuit strengths, consistency, and race-craft — built from four seasons of lap-level data.",
    accent: "var(--white)",
    cta: { label: "Explore Driver Dynamics", href: "#dynamics" },
    in: [0.58, 0.67],
    out: [0.90, 1.0],
  },
];

// ── Digital wind tunnel background ──────────────────────────────────────
// Static image (src/assets/wind-tunnel-bg.png), not an interactive canvas —
// a dark radial/linear gradient overlay sits on top to keep the scroll text
// readable against it.
export default function PinSection() {
  const sectionRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });

  const beatMotions = BEATS.map((beat) => {
    const [inStart, inEnd] = beat.in;
    const [outStart, outEnd] = beat.out;
    const range = [inStart, inEnd, outStart, outEnd];
    return {
      opacity: useTransform(scrollYProgress, range, [0, 1, 1, 0]),
      y: useTransform(scrollYProgress, range, [28, 0, 0, -28]),
      scale: useTransform(scrollYProgress, range, [0.96, 1, 1, 1.04]),
    };
  });

  return (
    <section ref={sectionRef} className="pin-section" id="how">
      <div className="pin-sticky">
        <div
          className="pin-bg"
          style={{ backgroundImage: `url(${windTunnelBg})` }}
        />
        <div className="pin-bg-overlay" />

        <div className="pin-center">
          <div className="pin-label">// How RaceMind Thinks</div>

          <div className="pin-beat-wrap">
            {BEATS.map((beat, i) => (
              <div key={i} className="pin-beat-center">
                <motion.div
                  className="pin-beat"
                  style={beatMotions[i]}
                >
                  <div className="pin-beat-tag">{beat.tag}</div>
                  <h2 className="pin-beat-title">
                    {beat.lines.map((line, j) => (
                      <span
                        key={j}
                        className={`pin-line pin-line-${line.size}`}
                        style={line.accent ? { color: beat.accent } : undefined}
                      >
                        {line.text}
                      </span>
                    ))}
                  </h2>
                  <p className="pin-beat-sub">{beat.sub}</p>
                  {beat.cta && (
                    <a
                      href={beat.cta.href}
                      className="pin-beat-cta"
                      style={{ "--cta-accent": beat.accent }}
                    >
                      {beat.cta.label} <span className="pin-beat-cta-arrow">→</span>
                    </a>
                  )}
                </motion.div>
              </div>
            ))}
          </div>

          <div className="pin-dots">
            {BEATS.map((_, i) => (
              <motion.div
                key={i}
                className="pin-dot"
                style={{ opacity: beatMotions[i].opacity }}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
