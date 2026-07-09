const COMPOUND_COLORS = {
  SOFT: "#E10600", MEDIUM: "#FFC800", HARD: "#E0E0E0",
  INTERMEDIATE: "#00C853", WET: "#2979FF",
};

/**
 * Compound-per-lap strip chart — one row per driver, one colored dot per
 * lap showing which tire compound they were on. Mirrors the original
 * Plotly scatter (x=LapNumber, y=Driver, color=Compound) as plain SVG.
 */
export default function TireStrategyChart({ laps, drivers, width = 900 }) {
  const rowH = 32;
  const padding = { top: 10, right: 24, bottom: 34, left: 70 };
  const height = padding.top + padding.bottom + drivers.length * rowH;
  const innerW = width - padding.left - padding.right;

  const relevant = laps.filter((l) => drivers.includes(l.Driver));
  const maxLap = Math.max(...relevant.map((l) => l.LapNumber), 1);

  const x = (lap) => padding.left + ((lap - 1) / (maxLap - 1 || 1)) * innerW;
  const yFor = (i) => padding.top + i * rowH + rowH / 2;

  const xTicks = Math.min(8, maxLap);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="tire-chart-svg" width="100%" height="auto" preserveAspectRatio="xMidYMid meet">
      {drivers.map((code, i) => (
        <text key={code} x={padding.left - 12} y={yFor(i)} textAnchor="end" dominantBaseline="middle" className="tire-chart-driver-label">
          {code}
        </text>
      ))}

      {Array.from({ length: xTicks + 1 }, (_, i) => {
        const lap = Math.round(1 + (i / xTicks) * (maxLap - 1));
        return (
          <text key={i} x={x(lap)} y={height - padding.bottom + 18} textAnchor="middle" className="tire-chart-tick">
            {lap}
          </text>
        );
      })}

      {drivers.map((code, i) =>
        relevant
          .filter((l) => l.Driver === code)
          .map((l) => (
            <circle
              key={`${code}-${l.LapNumber}`}
              cx={x(l.LapNumber)}
              cy={yFor(i)}
              r={4}
              fill={COMPOUND_COLORS[l.Compound] || "#555"}
              opacity={0.9}
            />
          ))
      )}
    </svg>
  );
}

export { COMPOUND_COLORS };
