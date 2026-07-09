import CarWidget from "./CarWidget";
import "./FinishLine.css";

const PODIUM_LABEL = ["P1", "P2", "P3"];

/**
 * Shows the top-3 predicted finishers crossing a checkered finish line,
 * one after another in finishing order (P1 first through the line).
 */
export default function FinishLine({ top3 }) {
  if (!top3 || top3.length === 0) return null;

  return (
    <div className="finish-track">
      <div className="finish-line">
        <div className="finish-checkers" />
        <span className="finish-label">FINISH</span>
      </div>
      <div className="finish-cars">
        {top3.map((entry, i) => (
          <div className={`finish-car-slot finish-car-slot--${i + 1}`} key={entry.driver.code}>
            <span className="finish-pos">{PODIUM_LABEL[i]}</span>
            <CarWidget driver={entry.driver} size="lg" />
            <span className="finish-prob">{(entry.podium_probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
