import { useEffect, useRef, useState } from "react";
import { getCircuitSvgUrl } from "../data/circuits";
import "./CircuitMap.css";

// Module-level cache so RaceRail (many mini cards) and Hero don't each
// re-fetch the same circuit SVG over the network.
const pathCache = new Map(); // layoutId/url -> Promise<string[]> (path "d" values)

async function fetchCircuitPaths(url) {
  if (pathCache.has(url)) return pathCache.get(url);

  const promise = fetch(url)
    .then((res) => {
      if (!res.ok) throw new Error(`Failed to load circuit SVG (${res.status})`);
      return res.text();
    })
    .then((svgText) => {
      const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
      if (doc.querySelector("parsererror")) throw new Error("Bad SVG markup");
      const paths = Array.from(doc.querySelectorAll("path"))
        .map((p) => p.getAttribute("d"))
        .filter(Boolean);
      if (!paths.length) throw new Error("No path data in circuit SVG");
      return paths;
    });

  pathCache.set(url, promise);
  // Don't cache a rejected promise — let a future render retry.
  promise.catch(() => pathCache.delete(url));
  return promise;
}

/**
 * Renders a real F1 circuit layout, sourced from julesr0y/f1-circuits-svg.
 *
 * Props:
 *  - circuitName: string — matched against src/data/circuits.js to find the SVG
 *  - mini: bool — compact styling for use inside cards (default false)
 *  - showCar: bool — animate a car dot lapping the circuit (default: !mini)
 *  - label: string — optional small caption shown under the map
 */
export default function CircuitMap({ circuitName, mini = false, showCar = !mini, label }) {
  const [pathD, setPathD] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | ready | error | unknown

  const pathRef = useRef(null);
  const carRef = useRef(null);
  const rafRef = useRef(null);
  const drawRafRef = useRef(null);

  const [pathLength, setPathLength] = useState(0);
  const [drawn, setDrawn] = useState(0);
  const [carPos, setCarPos] = useState(null);

  const url = getCircuitSvgUrl(circuitName);

  // ── Fetch + parse the circuit path ──────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setPathD(null);
    setDrawn(0);

    if (!url) {
      setStatus("unknown");
      return;
    }

    fetchCircuitPaths(url)
      .then((paths) => {
        if (cancelled) return;
        // Use the longest path (outline) — some layouts include small
        // extra marks (start/finish tick) as separate short paths.
        const longest = paths.reduce((a, b) => (b.length > a.length ? b : a));
        setPathD(longest);
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });

    return () => { cancelled = true; };
  }, [url]);

  // ── Draw-in animation once the path is mounted ──────────────────────
  useEffect(() => {
    if (status !== "ready" || !pathRef.current) return;
    const total = pathRef.current.getTotalLength();
    setPathLength(total);
    setDrawn(0);

    let start = null;
    const DURATION = mini ? 1200 : 2200;
    const draw = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / DURATION, 1);
      const eased = 1 - Math.pow(1 - p, 3); // ease-out cubic
      setDrawn(eased * total);
      if (p < 1) drawRafRef.current = requestAnimationFrame(draw);
    };
    drawRafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(drawRafRef.current);
  }, [status, pathD, mini]);

  // ── Car dot lapping the circuit ──────────────────────────────────────
  useEffect(() => {
    if (!showCar || status !== "ready" || !pathRef.current) return;
    const total = pathRef.current.getTotalLength();
    const LAP_DURATION = mini ? 6000 : 8000;
    let start = null;

    const animate = (ts) => {
      if (!start) start = ts;
      const elapsed = ts - start;
      const progress = (elapsed % LAP_DURATION) / LAP_DURATION;
      try {
        const point = pathRef.current.getPointAtLength(progress * total);
        setCarPos({ x: point.x, y: point.y });
      } catch {
        // path not ready — skip frame
      }
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [showCar, status, mini]);

  const wrapClass = `circuit-map ${mini ? "circuit-map-mini" : ""}`;

  if (status === "unknown" || status === "error") {
    return (
      <div className={`${wrapClass} circuit-map-empty`}>
        <div className="circuit-map-empty-mark" />
        {!mini && <span className="circuit-map-empty-label">Layout unavailable</span>}
      </div>
    );
  }

  return (
    <div className={wrapClass}>
      <svg
        className="circuit-map-svg"
        viewBox="0 0 500 500"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {pathD && (
          <>
            {/* Base track (dim, always fully visible) */}
            <path
              d={pathD}
              stroke={mini ? "#232323" : "#1a1a1a"}
              strokeWidth={mini ? 10 : 14}
              strokeLinejoin="round"
              strokeLinecap="round"
              fill="none"
            />
            {/* Measurement path, invisible, gives us getTotalLength/getPointAtLength */}
            <path
              ref={pathRef}
              d={pathD}
              stroke="transparent"
              strokeWidth="1"
              fill="none"
            />
            {/* Draw-in racing line */}
            {pathLength > 0 && (
              <path
                d={pathD}
                stroke={mini ? "rgba(242,242,242,0.55)" : "rgba(242,242,242,0.75)"}
                strokeWidth={mini ? 2 : 2.5}
                strokeLinejoin="round"
                strokeLinecap="round"
                fill="none"
                strokeDasharray={pathLength}
                strokeDashoffset={pathLength - drawn}
              />
            )}
            {/* Car dot */}
            {showCar && carPos && (
              <g transform={`translate(${carPos.x}, ${carPos.y})`}>
                <circle r={mini ? 6 : 9} fill="var(--red)" opacity="0.18" />
                <circle r={mini ? 3.2 : 4.5} fill="var(--red)" />
                <circle r={mini ? 1.4 : 2} fill="var(--white)" opacity="0.9" />
              </g>
            )}
          </>
        )}
      </svg>
      {label && !mini && <div className="circuit-map-label">{label}</div>}
    </div>
  );
}
