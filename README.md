---
title: RaceMindAI F1 Intelligence Platform
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
# 🏎️ RaceMindAI - F1 Intelligence Platform

> A full-stack AI-powered Formula 1 intelligence platform with live 2026 season data, ML race predictions, and Monte Carlo championship simulations — built as a React + FastAPI application with an editorial, motorsport-inspired UI.

**Live app:** https://race-mind-ai-f1-intelligence-platfo.vercel.app
**Backend API:** https://huggingface.co/spaces/Megmez/RaceMindAI

## What is RaceMindAI?

RaceMindAI is a personal F1 data science project that goes beyond typical portfolio dashboards. It ingests **live 2026 F1 season data**, runs real ML models, and gives you a full analytics platform for the current championship — race predictions, qualifying predictions, championship simulation, driver comparison, and race-weekend breakdowns.

The app updates automatically as the 2026 season progresses: standings, race results, and championship predictions all reflect real data pulled live from FastF1 and Jolpica.

---

## Pages

### Dashboard (Home)
Editorial hero with live model status, a scroll-driven "how RaceMind thinks" walkthrough, and the full 2026 race schedule as a horizontally-scrolling rail — past rounds show real podium results, upcoming rounds show the circuit outline.

### Predictor
Two connected models in one page:
- **Qualifying Predictor** — XGBRanker model predicting single-lap qualifying order from rolling qualifying form, constructor pace, and circuit type.
- **Race Predictor** — XGBoost classifier predicting podium probability from grid position, rolling form, constructor pace, circuit type, and DNF rate. Includes SHAP explainability (with an in-app explainer for what the values mean) and a finish-line podium visual.
- A real, hand-built **F1 starting grid** — two staggered rows, drag-and-drop to reorder, team-colored car glyphs, correct grid-pairing geometry. The predicted qualifying grid can be sent straight into the race predictor with one click.

### Standings
Card-based championship page (not tables): leader card, podium cards, driver/constructor standings, season timeline with hover-to-see-winner, live stat strip (biggest gainer/loser, largest gap), and a configurable **Championship Forecast** — Monte Carlo simulation with adjustable simulation count, upset factor, safety car frequency, and DNF modeling, each with an inline explainer.

### Driver Dynamics
Compare up to 6 current-grid drivers on a 6-axis radar (street, power, technical, high-downforce, consistency, race craft), with per-driver scorecards (strengths/weaknesses, bar or numeric view) and a full current-grid comparison table.

### Race Analysis
Session-by-session breakdown: race/session selector, lap-time evolution (multi-driver line chart), tire strategy (compound-per-lap chart), and a premium race-results table. Built strictly from the columns the backend actually returns — no invented telemetry, weather, or sector data.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | FastF1, OpenF1 API, Jolpica (Ergast replacement) |
| ML models | XGBoost (race podium classifier), XGBRanker (qualifying order) |
| Simulation | NumPy Monte Carlo season simulator |
| Backend | Python, FastAPI, pandas |
| Frontend | React 18, Vite, React Router |
| Animation | Framer Motion |
| Charts | Custom SVG (radar chart, lap-time chart, tire-strategy chart — no charting library dependency) |
| Backend deployment | Docker on Hugging Face Spaces |
| Frontend deployment | Vercel |

---

## ML Model Details

**Race Podium Predictor (XGBoost)**
- Target: binary podium classification (top 3 finish)
- Training data: 2022–2026 seasons, weighted toward the current regulation era (2026 races weighted 3× over older seasons)
- Test accuracy: **~90–94%** depending on retrain
- Key features: grid position, 5-race rolling points average, 10-race win/podium rate, circuit type encoding, constructor avg points, constructor DNF rate
- Explainability: SHAP values per prediction, surfaced in the UI with plain-language reasoning

**Qualifying Predictor (XGBRanker)**
- Target: predicted qualifying order (pairwise ranking, not classification)
- Features: rolling qualifying-position form, qualifying-position consistency, constructor qualifying pace, circuit type, weather flag
- New-constructor and rookie fallbacks so 2026-only teams/drivers (Audi, Cadillac, rookies) don't get zero-filled

**Season Championship Simulator**
- Monte Carlo approach: user-configurable simulation count (1,000–10,000)
- Driver strength derived from current points standings
- Configurable upset factor (race-to-race variance) and safety car frequency multiplier
- Optional DNF/retirement modeling based on constructor historical DNF rate
- Updates after every round with fresh standings

---

## Project Structure

This is a monorepo — one GitHub repo, two deploy targets.

```
racemind-ai/
├── backend/                        → deploys to Hugging Face Spaces (Docker)
│   ├── app/
│   │   ├── api/
│   │   │   └── main.py             # FastAPI routes
│   │   ├── data/
│   │   │   ├── fastf1_client.py    # FastF1 telemetry & lap data
│   │   │   └── ergast_client.py    # Standings & schedule (Jolpica API)
│   │   └── models/
│   │       ├── feature_engineering.py          # Race model feature pipeline
│   │       ├── race_predictor.py                # XGBoost podium model
│   │       ├── qualifying_feature_engineering.py # Qualifying model feature pipeline
│   │       ├── qualifying_predictor.py           # XGBRanker qualifying model
│   │       ├── driver_dna.py                     # 6-dimension driver profile builder
│   │       └── season_simulator.py               # Monte Carlo simulator
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                       → deploys to Vercel
│   └── src/
│       ├── pages/                  # One page component per route
│       ├── components/             # Predictor, Standings, DriverDynamics, RaceAnalysis, StartingGrid, CarWidget, charts, etc.
│       ├── assets/cars/            # Team-colored top-down car artwork
│       └── api.js                  # All backend API calls
├── .github/workflows/
│   ├── deploy-hf.yml               # Syncs backend/ to the Hugging Face Space on push
│   └── hf-keep-alive.yml           # Pings the Space every 36h so it doesn't sleep
└── vercel.json                     # SPA rewrite rules for React Router
```

---

## Deployment

- **Backend**: Docker container on Hugging Face Spaces (free CPU tier). A GitHub Action (`deploy-hf.yml`) splits and force-pushes `backend/` to the Space's own git repo on every push to `main`. A second Action (`hf-keep-alive.yml`) pings the Space every 36 hours so it doesn't hit Hugging Face's 48-hour free-tier sleep timeout.
- **Frontend**: Vercel, root directory set to `frontend/`, `VITE_API_URL` pointing at the live Hugging Face Space URL.
- **CORS**: the backend explicitly allows the production Vercel domain plus `*.vercel.app` preview deployments.

---

## Data Sources

- **[FastF1](https://github.com/theOehrly/Fast-F1)** — Lap times, tire compounds, race results
- **[OpenF1 API](https://openf1.org)** — Live session data
- **[Jolpica API](https://jolpi.ca)** — Standings, schedules, historical qualifying results (Ergast replacement)

---