import { motion } from "framer-motion";

/**
 * Pure-SVG radar chart — no charting library dependency. Renders N axes
 * around a circle, concentric grid rings at 25/50/75/100, and one filled
 * polygon per series (each series: { label, color, values: {axisKey: 0-100} }).
 */
export default function RadarChart({ axes, series, size = 420 }) {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.34;
  const labelRadius = radius * 1.24;
  const n = axes.length;

  const angleFor = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;

  const pointFor = (i, value) => {
    const angle = angleFor(i);
    const r = radius * (Math.max(0, Math.min(100, value)) / 100);
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  };

  const ringPoints = (pct) =>
    axes
      .map((_, i) => {
        const angle = angleFor(i);
        const r = radius * pct;
        return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
      })
      .join(" ");

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="radar-svg" width="100%" height="auto">
      {/* Grid rings */}
      {[0.25, 0.5, 0.75, 1].map((pct) => (
        <polygon key={pct} points={ringPoints(pct)} className="radar-ring" />
      ))}

      {/* Axis spokes + labels */}
      {axes.map((axis, i) => {
        const angle = angleFor(i);
        const [x2, y2] = [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
        const [lx, ly] = [cx + labelRadius * Math.cos(angle), cy + labelRadius * Math.sin(angle)];
        return (
          <g key={axis.key}>
            <line x1={cx} y1={cy} x2={x2} y2={y2} className="radar-spoke" />
            <text
              x={lx} y={ly}
              textAnchor={Math.abs(Math.cos(angle)) < 0.2 ? "middle" : x2 > cx ? "start" : "end"}
              dominantBaseline={Math.abs(Math.sin(angle)) < 0.2 ? "middle" : y2 > cy ? "hanging" : "auto"}
              className="radar-axis-label"
            >
              {axis.label}
            </text>
          </g>
        );
      })}

      {/* Series polygons */}
      {series.map((s) => {
        const pts = axes.map((axis, i) => pointFor(i, s.values[axis.key] ?? 0));
        const pointsStr = pts.map((p) => p.join(",")).join(" ");
        return (
          <motion.polygon
            key={s.label}
            points={pointsStr}
            fill={s.color}
            fillOpacity={0.28}
            stroke={s.color}
            strokeWidth={2}
            initial={{ opacity: 0, scale: 0.7 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            style={{ transformOrigin: `${cx}px ${cy}px` }}
          />
        );
      })}

      {/* Series points */}
      {series.map((s) =>
        axes.map((axis, i) => {
          const [x, y] = pointFor(i, s.values[axis.key] ?? 0);
          return <circle key={`${s.label}-${axis.key}`} cx={x} cy={y} r={3} fill={s.color} />;
        })
      )}
    </svg>
  );
}
