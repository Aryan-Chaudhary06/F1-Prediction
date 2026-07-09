import { Link, useLocation } from "react-router-dom";

export default function Nav() {
  const { pathname } = useLocation();
  const isActive = (path) => (pathname === path ? "active" : "");

  return (
    <nav>
      <div className="nav-logo"><span>Race</span>Mind AI</div>
      <ul className="nav-links">
        <li><Link to="/" className={isActive("/")}>Dashboard</Link></li>
        <li><Link to="/predictor" className={isActive("/predictor")}>Predictor</Link></li>
        <li><Link to="/standings" className={isActive("/standings")}>Standings</Link></li>
        <li><Link to="/dynamics" className={isActive("/dynamics")}>Driver Dynamics</Link></li>
        <li><Link to="/race-analysis" className={isActive("/race-analysis")}>Race Analysis</Link></li>
      </ul>
    </nav>
  );
}
