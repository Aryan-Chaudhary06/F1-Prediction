import { useEffect, useState, useRef } from "react";
import { getDriverStandings } from "../api";
import "./Ticker.css";

export default function Ticker({ year = 2026 }) {
  const [drivers, setDrivers] = useState([]);

  useEffect(() => {
    getDriverStandings(year).then(setDrivers).catch(() => {});
  }, [year]);

  if (!drivers.length) return null;

  // Duplicate for seamless loop
  const items = [...drivers, ...drivers];

  return (
    <div className="ticker-wrap" aria-label="Live driver standings">
      <div className="ticker-inner">
        {items.map((d, i) => (
          <div className="ticker-item" key={`${d.driver}-${i}`}>
            <span className="t-pos">P{d.position ?? i % drivers.length + 1}</span>
            <span className="t-dot" style={{ color: d.team_color ?? "#E8002D" }}>●</span>
            <span className="t-name">{d.driver}</span>
            <span className="t-pts">{d.points} PTS</span>
            <span className="t-team">{d.constructor}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
