import { motion } from "framer-motion";

/**
 * Pure-SVG line chart for lap time evolution — no charting library
 * dependency. laps: [{LapNumber, Driver, LapTimeSeconds}], drivers: array
 * of driver codes to plot, colors: {driverCode: hexColor}.
 */
export default function LapTimeChart({ laps, drivers, colors, width = 900, height = 380 }) {
  const padding = { top: 20, right: 24, bottom: 40, left: 56 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const relevant = laps.filter((l) => drivers.includes(l.Driver));
  if (relevant.length === 0) return null;

  const maxLap = Math.max(...relevant.map((l) => l.LapNumber));
  const minLap = Math.min(...relevant.map((l) => l.LapNumber));
  const times = relevant.map((l) => l.LapTimeSeconds).filter((t) => t > 0);
  const minT = Math.min(...times) * 0.98;
  const maxT = Math.max(...times) * 1.02;

  const x = (lap) => padding.left + ((lap - minLap) / (maxLap - minLap || 1)) * innerW;
  const y = (t) => padding.top + innerH - ((t - minT) / (maxT - minT || 1)) * innerH;

  const fmtTime = (s) => {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(2).padStart(5, "0");
    return `${m}:${sec}`;
  };

  const yTicks = 5;
  const xTicks = Math.min(8, maxLap - minLap);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="lap-chart-svg" width="100%" height="auto" preserveAspectRatio="xMidYMid meet">
      {/* Y grid + labels */}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const t = minT + (i / yTicks) * (maxT - minT);
        return (
          <g key={i}>
            <line x1={padding.left} x2={width - padding.right} y1={y(t)} y2={y(t)} className="lap-chart-grid" />
            <text x={padding.left - 10} y={y(t)} textAnchor="end" dominantBaseline="middle" className="lap-chart-tick">
              {fmtTime(t)}
            </text>
          </g>
        );
      })}

      {/* X labels */}
      {Array.from({ length: xTicks + 1 }, (_, i) => {
        const lap = Math.round(minLap + (i / xTicks) * (maxLap - minLap));
        return (
          <text key={i} x={x(lap)} y={height - padding.bottom + 22} textAnchor="middle" className="lap-chart-tick">
            {lap}
          </text>
        );
      })}
      <text x={width / 2} y={height - 4} textAnchor="middle" className="lap-chart-axis-title">Lap</text>

      {/* Driver lines */}
      {drivers.map((code) => {
        const rows = relevant.filter((l) => l.Driver === code && l.LapTimeSeconds > 0).sort((a, b) => a.LapNumber - b.LapNumber);
        if (rows.length === 0) return null;
        const d = rows.map((r, i) => `${i === 0 ? "M" : "L"} ${x(r.LapNumber)} ${y(r.LapTimeSeconds)}`).join(" ");
        return (
          <motion.path
            key={code}
            d={d}
            fill="none"
            stroke={colors[code] || "#888"}
            strokeWidth={2}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        );
      })}
    </svg>
  );
}
