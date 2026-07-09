import { useState } from "react";
import CarWidget from "./CarWidget";
import "./StartingGrid.css";

// ── Real F1 grid geometry (horizontal) ───────────────────────────────────
// Track runs left (front, pole) to right (back of field), cars nose-left.
// Two rows, staggered: P1 front in the top row, P2 one UNIT further right
// in the bottom row, P3 back in the top row exactly 2×UNIT right of P1
// (i.e. 1×UNIT right of P2), and so on. Same-row neighbors (P1/P3, P2/P4,
// ...) sit exactly 2×UNIT apart; cross-row neighbors (P1/P2, P3/P4, ...)
// sit exactly 1×UNIT apart.
//
// Two size presets: "sm" for the tighter qualifying-predictor panel, "lg"
// for the roomier race-predictor panel. Same-row neighbors are 2×UNIT
// apart center-to-center; to get a HALF-car-width gap between them
// (rather than a full car length), UNIT = 0.75 × CAR_W.
const SIZE_PRESETS = {
  sm: { UNIT: 36, CAR_W: 48, ROW_BOTTOM_Y: 40, LABEL_W: 40, SLOT_W: 40, SLOT_H: 24 },
  lg: { UNIT: 48, CAR_W: 64, ROW_BOTTOM_Y: 54, LABEL_W: 50, SLOT_W: 52, SLOT_H: 32 },
};

function positionXY(index, UNIT, ROW_BOTTOM_Y) {
  // index is 0-based, position number = index + 1
  const isTop = index % 2 === 0; // odd positions (1,3,5,...) -> top row
  const row = Math.floor(index / 2);
  const x = isTop ? row * 2 * UNIT : row * 2 * UNIT + UNIT;
  const y = isTop ? 0 : ROW_BOTTOM_Y;
  return { x, y, isTop };
}

/**
 * Real staggered F1 starting grid, horizontal: P1 at the left end, P22 at
 * the right end, two rows staggered per positionXY above. Drag a car onto
 * another box to swap the two grid positions (native HTML5 DnD). The
 * driver's code+number tag (e.g. "HAM44") sits on the far side of the
 * track from the centerline stagger — above the car for the top row,
 * below it for the bottom row. `size` ("sm" | "lg") picks a spacing/car
 * preset for the panel it's used in. Always renders the full field (no
 * truncation/expand toggle).
 */
export default function StartingGrid({ drivers, order, onReorder, editable = false, size = "lg" }) {
  const [dragIndex, setDragIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const { UNIT, CAR_W, ROW_BOTTOM_Y, LABEL_W, SLOT_W, SLOT_H } = SIZE_PRESETS[size];
  const carSize = size === "sm" ? "row-sm" : "row-lg";

  const driverByCode = {};
  drivers.forEach((d) => { driverByCode[d.code] = d; });

  const handleDrop = (targetIndex) => {
    if (dragIndex === null || dragIndex === targetIndex) {
      setDragIndex(null); setDragOverIndex(null);
      return;
    }
    const next = [...order];
    [next[dragIndex], next[targetIndex]] = [next[targetIndex], next[dragIndex]];
    onReorder(next);
    setDragIndex(null); setDragOverIndex(null);
  };

  const maxX = Math.max(...order.map((_, i) => positionXY(i, UNIT, ROW_BOTTOM_Y).x));
  const containerWidth = maxX + CAR_W + LABEL_W;
  const containerHeight = ROW_BOTTOM_Y + (size === "sm" ? 100 : 130);

  return (
    <div className="grid-track" style={{ width: containerWidth, height: containerHeight }}>
      <div className="grid-startline" />
      {order.map((code, i) => {
        const driver = driverByCode[code];
        const { x, y, isTop } = positionXY(i, UNIT, ROW_BOTTOM_Y);
        const label = driver ? `${driver.code}${driver.number ?? ""}` : code;
        return (
          <div
            key={code}
            className={`grid-box ${dragOverIndex === i ? "grid-box--over" : ""} ${editable ? "grid-box--editable" : ""}`}
            style={{ left: x, top: y, width: CAR_W }}
            draggable={editable}
            onDragStart={() => setDragIndex(i)}
            onDragEnd={() => { setDragIndex(null); setDragOverIndex(null); }}
            onDragOver={(e) => { e.preventDefault(); setDragOverIndex(i); }}
            onDragLeave={() => setDragOverIndex((cur) => (cur === i ? null : cur))}
            onDrop={(e) => { e.preventDefault(); handleDrop(i); }}
          >
            {isTop && <span className="grid-box-label grid-box-label--above">{label}</span>}
            <span className="grid-car-stack">
              <span className="grid-slot-bracket" style={{ width: SLOT_W, height: SLOT_H }} />
              <CarWidget driver={driver} size={carSize} dragging={dragIndex === i} />
            </span>
            {!isTop && <span className="grid-box-label grid-box-label--below">{label}</span>}
          </div>
        );
      })}
    </div>
  );
}
