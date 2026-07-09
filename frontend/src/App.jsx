import { Routes, Route } from "react-router-dom";
import Nav from "./components/Nav";
import Home from "./pages/Home";
import PredictorPage from "./pages/PredictorPage";
import StandingsPage from "./pages/StandingsPage";
import DriverDynamicsPage from "./pages/DriverDynamicsPage";
import RaceAnalysisPage from "./pages/RaceAnalysisPage";

export default function App() {
  return (
    <>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predictor" element={<PredictorPage />} />
        <Route path="/standings" element={<StandingsPage />} />
        <Route path="/dynamics" element={<DriverDynamicsPage />} />
        <Route path="/race-analysis" element={<RaceAnalysisPage />} />
      </Routes>
    </>
  );
}
