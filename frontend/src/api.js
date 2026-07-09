import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL;

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

// ── Standings ────────────────────────────────────────────────────────────
export const getDriverStandings = (year, round = null) =>
  client.get(`/api/standings/drivers/${year}`, { params: { round } }).then(r => r.data);

export const getConstructorStandings = (year, round = null) =>
  client.get(`/api/standings/constructors/${year}`, { params: { round } }).then(r => r.data);

export const getSchedule = (year) =>
  client.get(`/api/schedule/${year}`).then(r => r.data);

export const getQualifying = (year, round) =>
  client.get(`/api/qualifying/${year}/${round}`).then(r => r.data);

// ── Race Analysis ────────────────────────────────────────────────────────
export const getRaceLaps = (year, gp, sessionType = "R") =>
  client.get(`/api/race-analysis/laps`, { params: { year, gp, session_type: sessionType } }).then(r => r.data);

export const getRaceResults = (year, gp) =>
  client.get(`/api/race-analysis/results`, { params: { year, gp } }).then(r => r.data);

export const getTelemetry = (year, gp, driver, lapNumber = null) =>
  client.get(`/api/race-analysis/telemetry`, { params: { year, gp, driver, lap_number: lapNumber } }).then(r => r.data);

export const getSessionDrivers = (year, gp, sessionType = "R") =>
  client.get(`/api/race-analysis/drivers`, { params: { year, gp, session_type: sessionType } }).then(r => r.data);

// ── Predictor ────────────────────────────────────────────────────────────
export const getPredictorStatus = () =>
  client.get(`/api/predictor/status`).then(r => r.data);

export const trainModel = (forceRefresh = false) =>
  client.post(`/api/predictor/train`, null, { params: { force_refresh: forceRefresh } }).then(r => r.data);

export const getGridDefaults = (year) =>
  client.get(`/api/predictor/grid-defaults/${year}`).then(r => r.data);

export const predictRace = (year, round, grid) =>
  client.post(`/api/predictor/predict`, { year, round, grid }).then(r => r.data);

export const explainPrediction = (driver, rowsUsed, topN = 6) =>
  client.post(`/api/predictor/explain`, { driver, rows_used: rowsUsed }, { params: { top_n: topN } }).then(r => r.data);

export const getFeatureImportance = () =>
  client.get(`/api/predictor/feature-importance`).then(r => r.data);

// ── Qualifying Predictor ─────────────────────────────────────────────────
export const getQualifyingStatus = () =>
  client.get(`/api/qualifying/status`).then(r => r.data);

export const trainQualifyingModel = (forceRefresh = false) =>
  client.post(`/api/qualifying/train`, null, { params: { force_refresh: forceRefresh } }).then(r => r.data);

export const predictQualifying = (year, round, weather = "Dry") =>
  client.post(`/api/qualifying/predict`, { year, round, weather }).then(r => r.data);

// ── Season Simulator ─────────────────────────────────────────────────────
export const runSimulation = (params) =>
  client.post(`/api/simulate`, params).then(r => r.data);

// ── Driver Dynamics ──────────────────────────────────────────────────────
export const getDriverDynamics = () =>
  client.get(`/api/driver-dynamics`).then(r => r.data);

export default client;
