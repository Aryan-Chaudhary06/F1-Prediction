// Maps circuit names (as returned by our schedule API / Ergast-Jolpica
// "circuitName" field) to layout IDs in the julesr0y/f1-circuits-svg repo
// (CC-BY-4.0). Repo structure: circuits/minimal/white-outline/{layoutId}.svg
//
// Source of truth for available layouts:
// https://raw.githubusercontent.com/julesr0y/f1-circuits-svg/main/circuits.json
//
// Keys are normalized (lowercased, punctuation stripped) so lookups are
// resilient to minor naming differences between data sources.

export const CIRCUIT_SVG_BASE =
  "https://raw.githubusercontent.com/julesr0y/f1-circuits-svg/main/circuits/minimal/white-outline";

// circuitName (various sources) -> layoutId
const RAW_MAP = {
  "Bahrain International Circuit": "bahrain-1",
  "Jeddah Corniche Circuit": "jeddah-1",
  "Albert Park Grand Prix Circuit": "melbourne-2",
  "Melbourne Grand Prix Circuit": "melbourne-2",
  "Suzuka Circuit": "suzuka-2",
  "Shanghai International Circuit": "shanghai-1",
  "Miami International Autodrome": "miami-1",
  "Circuit de Monaco": "monaco-6",
  "Circuit de Barcelona-Catalunya": "catalunya-6",
  "Circuit Gilles Villeneuve": "montreal-6",
  "Red Bull Ring": "spielberg-3",
  "Silverstone Circuit": "silverstone-8",
  "Circuit de Spa-Francorchamps": "spa-francorchamps-4",
  "Hungaroring": "hungaroring-3",
  "Circuit Park Zandvoort": "zandvoort-5",
  "Circuit Zandvoort": "zandvoort-5",
  "Autodromo Nazionale Monza": "monza-7",
  "Autodromo Nazionale di Monza": "monza-7",
  "Baku City Circuit": "baku-1",
  "Marina Bay Street Circuit": "marina-bay-4",
  "Circuit of the Americas": "austin-1",
  "Autódromo Hermanos Rodríguez": "mexico-city-3",
  "Autodromo Hermanos Rodriguez": "mexico-city-3",
  "Autódromo José Carlos Pace": "interlagos-2",
  "Autodromo Jose Carlos Pace": "interlagos-2",
  "Las Vegas Street Circuit": "las-vegas-1",
  "Las Vegas Strip Street Circuit": "las-vegas-1",
  "Losail International Circuit": "lusail-1",
  "Lusail International Circuit": "lusail-1",
  "Yas Marina Circuit": "yas-marina-2",
  "Circuito de Madring": "madring-1",
  "Madring": "madring-1",
};

function normalize(str = "") {
  return str
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // strip accents
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

const NORMALIZED_MAP = Object.fromEntries(
  Object.entries(RAW_MAP).map(([name, layoutId]) => [normalize(name), layoutId])
);

/**
 * Resolve a circuit name to its julesr0y layoutId, or null if unknown.
 * Falls back to a loose substring match so slightly different naming
 * (e.g. missing diacritics, "Autodromo" vs "Autódromo") still resolves.
 */
export function getLayoutId(circuitName) {
  if (!circuitName) return null;
  const key = normalize(circuitName);
  if (NORMALIZED_MAP[key]) return NORMALIZED_MAP[key];

  const match = Object.keys(NORMALIZED_MAP).find(
    (k) => key.includes(k) || k.includes(key)
  );
  return match ? NORMALIZED_MAP[match] : null;
}

/** Full SVG URL for a given circuit name, or null if no layout is known. */
export function getCircuitSvgUrl(circuitName) {
  const layoutId = getLayoutId(circuitName);
  return layoutId ? `${CIRCUIT_SVG_BASE}/${layoutId}.svg` : null;
}
