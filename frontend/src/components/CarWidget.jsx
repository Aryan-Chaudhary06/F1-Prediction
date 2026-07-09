import "./CarWidget.css";
import mclarenImg from "../assets/cars/mclaren.png";
import ferrariImg from "../assets/cars/ferrari.png";
import mercedesImg from "../assets/cars/mercedes.png";
import redbullImg from "../assets/cars/redbull.png";
import astonmartinImg from "../assets/cars/astonmartin.png";
import alpineImg from "../assets/cars/alpine.png";
import williamsImg from "../assets/cars/williams.png";
import haasImg from "../assets/cars/haas.png";
import racingbullsImg from "../assets/cars/racingbulls.png";
import audiImg from "../assets/cars/audi.png";
import cadillacImg from "../assets/cars/cadillac.png";

// Team -> cropped top-down car image (flat team-color artwork, no
// sponsor/manufacturer logos — safe to ship). All 11 teams covered.
const TEAM_IMAGES = {
  "McLaren": mclarenImg,
  "Ferrari": ferrariImg,
  "Mercedes": mercedesImg,
  "Red Bull": redbullImg,
  "Aston Martin": astonmartinImg,
  "Alpine": alpineImg,
  "Williams": williamsImg,
  "Haas": haasImg,
  "Racing Bulls": racingbullsImg,
  "Audi": audiImg,
  "Cadillac": cadillacImg,
};

function CarGlyph() {
  return (
    <svg viewBox="0 0 100 220" className="car-svg" xmlns="http://www.w3.org/2000/svg">
      <rect x="1"  y="150" width="14" height="30" rx="3" className="car-wheel" />
      <rect x="85" y="150" width="14" height="30" rx="3" className="car-wheel" />
      <rect x="4"  y="26" width="11" height="26" rx="3" className="car-wheel" />
      <rect x="85" y="26" width="11" height="26" rx="3" className="car-wheel" />
      <rect x="10" y="10" width="80" height="6" rx="1.5" className="car-wing" />
      <rect x="10" y="10" width="6"  height="16" rx="1" className="car-wing" />
      <rect x="84" y="10" width="6"  height="16" rx="1" className="car-wing" />
      <path
        d="M50 16 C54 16 56 24 56 34 C56 42 54 48 51 52 L51 62 C60 64 64 70 64 78
           L64 88 C64 94 60 98 50 100 C40 98 36 94 36 88 L36 78 C36 70 40 64 49 62
           L49 52 C46 48 44 42 44 34 C44 24 46 16 50 16 Z"
        className="car-body"
      />
      <path
        d="M30 98 C22 100 18 108 18 118 L18 138 C18 148 24 154 34 156 L34 190
           C34 198 40 204 50 205 C60 204 66 198 66 190 L66 156 C76 154 82 148 82 138
           L82 118 C82 108 78 100 70 98 C64 96 58 96 50 96 C42 96 36 96 30 98 Z"
        className="car-body car-body--rear"
      />
      <path d="M40 92 C40 84 44 79 50 79 C56 79 60 84 60 92" className="car-halo" />
      <line x1="50" y1="104" x2="50" y2="186" className="car-ridge" />
      <rect x="8"  y="192" width="84" height="6" rx="1.5" className="car-wing" />
      <rect x="8"  y="192" width="6"  height="18" rx="1" className="car-wing" />
      <rect x="86" y="192" width="6"  height="18" rx="1" className="car-wing" />
    </svg>
  );
}

/**
 * Pure car glyph — no number or code baked in. Labeling (driver code +
 * number, e.g. "HAM44") is handled by the grid layout so it can be placed
 * on whichever side of the car makes sense for that grid position.
 */
export default function CarWidget({ driver, size = "md", dragging = false, ghost = false }) {
  if (!driver) return null;
  const { team, team_color } = driver;
  const teamImg = TEAM_IMAGES[team];

  return (
    <div
      className={`car-widget car-widget--${size} ${dragging ? "is-dragging" : ""} ${ghost ? "is-ghost" : ""}`}
      style={{ "--car-color": team_color || "#555" }}
    >
      {teamImg ? (
        <span className="car-photo-frame">
          <img src={teamImg} alt={`${team} car`} className="car-photo" draggable={false} />
        </span>
      ) : (
        <CarGlyph />
      )}
    </div>
  );
}
