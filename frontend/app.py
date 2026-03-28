import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.ergast_client import (
    get_driver_standings, get_constructor_standings,
    get_season_schedule, get_historical_results
)
from app.data.fastf1_client import get_lap_times, get_race_results
from app.models.race_predictor import train_model, load_model
from app.models.season_simulator import simulate_season, build_driver_strengths
from app.models.driver_dna import build_driver_dna
from app.models.explainability import get_shap_explanation, get_top_factors

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RaceMind AI — F1 Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapse sidebar — nav is in topbar now
)

# ── Global CSS: top navbar + premium dark theme ────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'DM Sans', sans-serif !important; }

/* ── App background ── */
.stApp { background: #080808 !important; }

/* ── Hide default sidebar & hamburger ── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }

/* ── Remove default top padding so our navbar sits flush ── */
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }

/* ── TOP NAVBAR ── */
.pit-navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(8,8,8,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: 60px;
    margin: 0 -2rem 2rem;           /* bleed edge-to-edge */
}
.pit-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #e10600;
    white-space: nowrap;
    user-select: none;
}
.pit-logo span { color: #fff; }
.pit-nav-links {
    display: flex;
    align-items: center;
    gap: 4px;
}
.pit-nav-btn {
    background: transparent;
    border: none;
    color: #666;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    transition: color 0.2s, background 0.2s;
    white-space: nowrap;
}
.pit-nav-btn:hover { color: #fff; background: rgba(255,255,255,0.05); }
.pit-nav-btn.active {
    color: #e10600 !important;
    background: rgba(225,6,0,0.1) !important;
}
.pit-season-select {
    display: flex;
    align-items: center;
    gap: 8px;
}
.pit-season-label {
    color: #555;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Page fade-in animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.page-enter {
    animation: fadeUp 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* ── Typography ── */
.main-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #e10600 0%, #ff5c58 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 2px;
}
.sub-header {
    font-size: 0.72rem;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

/* ── Hero card ── */
.hero-card {
    background: linear-gradient(135deg, #120000 0%, #180808 60%, #111 100%);
    border: 1px solid #2a1010;
    border-left: 4px solid #e10600;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin: 1rem 0 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: 'F1';
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 9rem; font-weight: 900;
    color: rgba(225,6,0,0.04);
    letter-spacing: -5px; pointer-events: none;
}
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(225,6,0,0.12);
    border: 1px solid rgba(225,6,0,0.25);
    color: #e10600; font-size: 0.62rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;
}
.live-dot {
    width: 6px; height: 6px; background: #e10600;
    border-radius: 50%; display: inline-block;
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.15} }

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1rem 0 1.5rem;
}
.stat-card {
    background: #0f0f0f;
    border: 1px solid #1a1a1a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    transition: transform 0.2s, border-color 0.2s;
    position: relative; overflow: hidden;
}
.stat-card:hover { transform: translateY(-2px); border-color: #2a2a2a; }
.stat-card::after {
    content: ''; position: absolute; top: 0; left: 0;
    width: 100%; height: 3px; border-radius: 14px 14px 0 0;
}
.stat-card.red::after   { background: #e10600; }
.stat-card.orange::after{ background: #ff8c00; }
.stat-card.teal::after  { background: #00d4aa; }
.stat-label { color: #444; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-weight: 600; }
.stat-value { color: #fff; font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1; }
.stat-sub   { color: #555; font-size: 0.75rem; margin-top: 6px; }

/* ── Section headers ── */
.section-header { display: flex; align-items: center; gap: 10px; margin: 1.8rem 0 1rem; }
.section-stripe { width: 4px; height: 20px; border-radius: 2px; flex-shrink: 0; }
.section-title  { font-size: 0.78rem; font-weight: 700; color: #ccc; text-transform: uppercase; letter-spacing: 1px; }

/* ── Info banner ── */
.info-banner {
    background: #0f0f0f;
    border: 1px solid #1a1a1a;
    border-left: 3px solid #7c4dff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.info-banner-label { color: #7c4dff; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.info-banner-text  { color: #888; font-size: 0.83rem; line-height: 1.6; }

/* ── Dashboard hero ── */
.dashboard-hero {
    background: linear-gradient(135deg, #e10600 0%, #8c0000 40%, #1a1a1a 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.dashboard-hero::after {
    content: '🏎️';
    position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem; opacity: 0.12;
    pointer-events: none;
}
.dashboard-hero h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.5rem; font-weight: 900;
    letter-spacing: -1px; text-transform: uppercase;
    color: #fff !important; margin: 0 !important;
    line-height: 1;
}
.dashboard-hero p { color: rgba(255,255,255,0.65); font-size: 0.9rem; margin-top: 8px; margin-bottom: 0; }

/* ── Quick action buttons (dashboard) ── */
.qa-btn {
    display: block;
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    color: #fff;
}
.qa-btn:hover { border-color: #e10600; background: rgba(225,6,0,0.06); transform: translateY(-2px); }
.qa-btn .qa-icon { font-size: 1.8rem; margin-bottom: 8px; display: block; }
.qa-btn .qa-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #ccc; }

/* ── Streamlit component overrides ── */
[data-testid="stDataFrame"] { border: 1px solid #1a1a1a !important; border-radius: 12px !important; overflow: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #e10600, #b80500) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    letter-spacing: 1px !important; padding: 0.55rem 1.4rem !important;
    text-transform: uppercase !important; font-size: 0.73rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(225,6,0,0.35) !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; border: none !important; }
[data-testid="metric-container"] {
    background: #111 !important; border: 1px solid #1a1a1a !important;
    border-radius: 12px !important; border-left: 3px solid #e10600 !important;
}
[data-testid="stMetricValue"] { color: #fff !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #555 !important; font-size: 0.65rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="stChatMessage"] { background: #111 !important; border: 1px solid #1a1a1a !important; border-radius: 12px !important; margin-bottom: 8px !important; }
[data-testid="stRadio"] label { color: #666 !important; font-size: 0.85rem !important; }
[data-testid="stRadio"] label:hover { color: #ddd !important; }
h2, h3 { color: #fff !important; font-weight: 700 !important; }
hr  { border-color: #1a1a1a !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080808; }
::-webkit-scrollbar-thumb { background: #e10600; border-radius: 2px; }

/* Select boxes, sliders, inputs */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stNumberInput"] label { color: #666 !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
</style>
""", unsafe_allow_html=True)

# ── GP Map (unchanged) ─────────────────────────────────────────────────────────
GP_MAP = {
    "Australian Grand Prix": "Australia",
    "Chinese Grand Prix": "China",
    "Japanese Grand Prix": "Japan",
    "Miami Grand Prix": "Miami",
    "Canadian Grand Prix": "Canada",
    "Monaco Grand Prix": "Monaco",
    "Barcelona Grand Prix": "Spain",
    "Austrian Grand Prix": "Austria",
    "British Grand Prix": "Great Britain",
    "Belgian Grand Prix": "Belgium",
    "Hungarian Grand Prix": "Hungary",
    "Dutch Grand Prix": "Netherlands",
    "Italian Grand Prix": "Monza",
    "Spanish Grand Prix": "Spain",
    "Azerbaijan Grand Prix": "Azerbaijan",
    "Singapore Grand Prix": "Singapore",
    "United States Grand Prix": "United States",
    "Mexico City Grand Prix": "Mexico",
    "São Paulo Grand Prix": "São Paulo",
    "Brazilian Grand Prix": "São Paulo",
    "Las Vegas Grand Prix": "Las Vegas",
    "Qatar Grand Prix": "Qatar",
    "Abu Dhabi Grand Prix": "Abu Dhabi",
    "Bahrain Grand Prix": "Bahrain",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "Emilia Romagna Grand Prix": "Emilia Romagna",
    "Emilia-Romagna Grand Prix": "Emilia Romagna",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def races_completed(schedule_df):
    today = str(pd.Timestamp.now().date())
    return len([r for r in schedule_df["date"] if r <= today])

def section_header(title: str, color: str = "#e10600"):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-stripe" style="background:{color};"></div>
        <div class="section-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Navigation state ───────────────────────────────────────────────────────────
NAV_PAGES = ["Dashboard", "Live Standings", "Race Analysis", "Race Predictor", "Season Championship", "Driver Dynamics"]
NAV_ICONS = ["🏁", "📊", "📈", "🤖", "🏆", "🧬"]
NAV_LABELS = ["Dashboard", "Standings", "Analysis", "Predictor", "Championship", "Driver Dynamics"]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"
if "season_year" not in st.session_state:
    st.session_state["season_year"] = 2026

page = st.session_state["active_page"]
season_year = st.session_state["season_year"]

# ── TOP NAVBAR ─────────────────────────────────────────────────────────────────
# Build navbar HTML — active page gets highlighted class
nav_buttons_html = "".join([
    f'<button class="pit-nav-btn{" active" if NAV_PAGES[i] == page else ""}" '
    f'onclick="window.location.href=\'?nav={NAV_PAGES[i].replace(" ","_")}\'">'
    f'{NAV_ICONS[i]} {NAV_LABELS[i]}</button>'
    for i in range(len(NAV_PAGES))
])

st.markdown(f"""
<div class="pit-navbar">
    <div class="pit-logo">RACE<span>MIND AI</span></div>
    <div class="pit-nav-links">
        {nav_buttons_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Handle URL-based nav (works in Streamlit via query params) ─────────────────
qp = st.query_params
if "nav" in qp:
    nav_val = qp["nav"].replace("_", " ")
    if nav_val in NAV_PAGES and nav_val != page:
        st.session_state["active_page"] = nav_val
        page = nav_val

# ── Streamlit-native nav buttons (hidden visually but functional) ──────────────
# We use a real button row for Streamlit interactivity as backup/primary mechanism
with st.container():
    cols = st.columns(len(NAV_PAGES) + 1)
    for i, (pg, lbl, icon) in enumerate(zip(NAV_PAGES, NAV_LABELS, NAV_ICONS)):
        with cols[i]:
            if st.button(f"{icon} {lbl}", key=f"nav_{pg}", use_container_width=True):
                st.session_state["active_page"] = pg
                page = pg
                st.rerun()
    with cols[-1]:
        # Season selector lives at the right end of the nav row
        new_year = st.selectbox("Season", [2026, 2025, 2024, 2023, 2022],
                                 index=[2026,2025,2024,2023,2022].index(season_year),
                                 key="season_selector", label_visibility="collapsed")
        if new_year != season_year:
            st.session_state["season_year"] = new_year
            season_year = new_year

# Style the nav row buttons to look like a topbar
st.markdown("""
<style>
/* Override the nav row buttons with pill-style appearance */
div[data-testid="stHorizontalBlock"] > div > div > div > .stButton > button {
    background: transparent !important;
    color: #555 !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 20px !important;
    font-size: 0.7rem !important;
    padding: 0.3rem 0.6rem !important;
    letter-spacing: 0.5px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"] > div > div > div > .stButton > button:hover {
    border-color: #e10600 !important;
    color: #e10600 !important;
    background: rgba(225,6,0,0.06) !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<hr style="border-color:#141414;margin:0 0 1.5rem;">', unsafe_allow_html=True)

# ── Wrap content in page-enter div for fade animation ─────────────────────────
st.markdown('<div class="page-enter">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="dashboard-hero">
        <h1>🏎️ RaceMind AI</h1>
        <p>Real-time Formula 1 Intelligence · Machine Learning Predictions · Championship Simulation</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Platform", "F1 Intelligence")
    col2.metric("Model", "XGBoost + Simulation")
    col3.metric("Status", "Live")

    st.markdown("### ⚡ Quick Actions")
    c1, c2, c3 = st.columns(3)

    if c1.button("📊 View Standings", use_container_width=True):
        st.session_state["active_page"] = "Live Standings"
        st.rerun()
    if c2.button("🤖 Run Prediction", use_container_width=True):
        st.session_state["active_page"] = "Race Predictor"
        st.rerun()
    if c3.button("🏆 Simulate Season", use_container_width=True):
        st.session_state["active_page"] = "Season Championship"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE STANDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Standings":
    st.markdown('<div class="main-header">Championship Standings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Formula 1 · Live Season Data</div>', unsafe_allow_html=True)

    with st.spinner("Loading live standings..."):
        drivers      = get_driver_standings(season_year)
        constructors = get_constructor_standings(season_year)
        schedule     = get_season_schedule(season_year)

    done  = races_completed(schedule)
    total = len(schedule)

    st.markdown(f"""
    <div class="hero-card">
        <div class="live-badge">
            <span class="live-dot"></span>
            {"LIVE · " + str(season_year) if season_year == 2026 else str(season_year) + " SEASON"}
        </div>
        <div style="color:#666;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
            Round {done} of {total}
        </div>
        <div style="display:flex;align-items:baseline;gap:24px;flex-wrap:wrap;">
            <div>
                <div style="color:#444;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;">Championship Leader</div>
                <div style="color:#fff;font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;font-weight:900;letter-spacing:-1px;line-height:1.1;margin-top:2px;">
                    {drivers.iloc[0]['full_name']}
                </div>
                <div style="color:#e10600;font-size:0.8rem;font-weight:600;margin-top:4px;">
                    {drivers.iloc[0]['constructor']} &nbsp;·&nbsp; {int(drivers.iloc[0]['points'])} pts
                </div>
            </div>
            <div style="width:1px;height:56px;background:#1e1e1e;margin:0 4px;"></div>
            <div>
                <div style="color:#444;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;">P2</div>
                <div style="color:#bbb;font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:800;margin-top:2px;">
                    {drivers.iloc[1]['full_name']}
                </div>
                <div style="color:#555;font-size:0.75rem;margin-top:3px;">
                    {int(drivers.iloc[1]['points'])} pts &nbsp;·&nbsp;
                    -{int(drivers.iloc[0]['points'] - drivers.iloc[1]['points'])} pts
                </div>
            </div>
            <div style="width:1px;height:56px;background:#1e1e1e;margin:0 4px;"></div>
            <div>
                <div style="color:#444;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;">P3</div>
                <div style="color:#bbb;font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:800;margin-top:2px;">
                    {drivers.iloc[2]['full_name']}
                </div>
                <div style="color:#555;font-size:0.75rem;margin-top:3px;">
                    {int(drivers.iloc[2]['points'])} pts &nbsp;·&nbsp;
                    -{int(drivers.iloc[0]['points'] - drivers.iloc[2]['points'])} pts
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card red">
            <div class="stat-label">Points Leader</div>
            <div class="stat-value">{int(drivers.iloc[0]['points'])}</div>
            <div class="stat-sub">{drivers.iloc[0]['full_name'].split()[-1]}</div>
        </div>
        <div class="stat-card orange">
            <div class="stat-label">Gap to P2</div>
            <div class="stat-value">+{int(drivers.iloc[0]['points'] - drivers.iloc[1]['points'])}</div>
            <div class="stat-sub">points ahead</div>
        </div>
        <div class="stat-card teal">
            <div class="stat-label">Season Progress</div>
            <div class="stat-value">{done}/{total}</div>
            <div class="stat-sub">{total - done} races left</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        section_header("Drivers Championship")
        fig = px.bar(
            drivers.head(10), x="points", y="full_name",
            orientation="h", color="constructor", text="points",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            yaxis={"categoryorder":"total ascending"},
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=400, xaxis_title="Points", yaxis_title="",
            font=dict(color="#ccc"), legend=dict(font=dict(color="#888")),
            xaxis=dict(gridcolor="#1a1a1a"),
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#aaa", size=11))
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Points: %{x}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Constructors Championship")
        fig2 = px.bar(
            constructors, x="points", y="constructor",
            orientation="h", color="constructor", text="points",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig2.update_layout(
            yaxis={"categoryorder":"total ascending"},
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=400, xaxis_title="Points", yaxis_title="",
            font=dict(color="#ccc"), xaxis=dict(gridcolor="#1a1a1a"),
        )
        fig2.update_traces(textposition="outside", textfont=dict(color="#aaa", size=11))
        fig2.update_traces(hovertemplate="<b>%{y}</b><br>Points: %{x}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

    section_header("Full Driver Standings")
    disp = drivers[["position","full_name","constructor","points","wins"]].copy()
    disp.columns = ["Pos","Driver","Constructor","Points","Wins"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    section_header("Season Schedule", color="#00d4aa")
    st.dataframe(schedule, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RACE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Race Analysis":
    st.markdown('<div class="main-header">Race Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Lap times · Tire strategy · Pace analysis</div>', unsafe_allow_html=True)

    schedule   = get_season_schedule(season_year)
    gp_options = schedule["gp_name"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        selected_gp = st.selectbox("Grand Prix", gp_options)
    with c2:
        session_type = st.selectbox(
            "Session", ["R","Q","FP1","FP2","FP3"],
            format_func=lambda x: {"R":"Race","Q":"Qualifying",
                                   "FP1":"Practice 1","FP2":"Practice 2","FP3":"Practice 3"}[x]
        )

    gp_key = GP_MAP.get(selected_gp, selected_gp.replace(" Grand Prix",""))

    if st.button("Load Session Data", type="primary"):
        with st.spinner(f"Loading {selected_gp} {season_year}..."):
            try:
                laps    = get_lap_times(season_year, gp_key, session_type)
                results = get_race_results(season_year, gp_key)
                st.session_state["laps"]    = laps
                st.session_state["results"] = results
                st.success(f"Loaded {len(laps)} laps from {selected_gp}")
            except Exception as e:
                st.error(f"Could not load data: {e}")

    if "laps" in st.session_state:
        laps    = st.session_state["laps"]
        results = st.session_state["results"]

        fastest = laps.loc[laps["LapTimeSeconds"].idxmin()]
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card red">
                <div class="stat-label">Total Laps</div>
                <div class="stat-value">{len(laps)}</div>
                <div class="stat-sub">across all drivers</div>
            </div>
            <div class="stat-card orange">
                <div class="stat-label">Fastest Lap</div>
                <div class="stat-value">{fastest['LapTimeSeconds']:.3f}s</div>
                <div class="stat-sub">{fastest['Driver']}</div>
            </div>
            <div class="stat-card teal">
                <div class="stat-label">Drivers</div>
                <div class="stat-value">{laps['Driver'].nunique()}</div>
                <div class="stat-sub">in session</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("Lap Time Evolution")
        avail = sorted(laps["Driver"].unique())
        sel   = st.multiselect("Select drivers", avail, default=avail[:5])
        if sel:
            fig = px.line(
                laps[laps["Driver"].isin(sel)],
                x="LapNumber", y="LapTimeSeconds", color="Driver",
                color_discrete_sequence=["#e10600","#ff8c00","#00d4aa","#7c4dff","#00b0ff",
                                          "#ff4081","#69f0ae","#ffeb3b","#40c4ff","#ff6d00"],
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=400, xaxis_title="Lap", yaxis_title="Lap Time (s)",
                font=dict(color="#ccc"), legend=dict(font=dict(color="#aaa")),
                xaxis=dict(gridcolor="#1a1a1a"), yaxis=dict(gridcolor="#1a1a1a"),
            )
            fig.update_traces(hovertemplate="<b>%{fullData.name}</b> Lap %{x}: %{y:.3f}s<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        section_header("Tire Strategy", color="#ff8c00")
        fig2 = px.scatter(
            laps, x="LapNumber", y="Driver", color="Compound",
            color_discrete_map={"SOFT":"#e10600","MEDIUM":"#ffc800",
                                 "HARD":"#e0e0e0","INTERMEDIATE":"#00c853","WET":"#2979ff"},
            opacity=0.85,
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=500, xaxis_title="Lap", yaxis_title="",
            font=dict(color="#ccc"), legend=dict(font=dict(color="#aaa")),
            xaxis=dict(gridcolor="#1a1a1a"),
        )
        fig2.update_traces(marker=dict(size=8))
        fig2.update_traces(hovertemplate="<b>%{y}</b> Lap %{x} — %{marker.color}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

        if session_type == "R":
            section_header("Race Results", color="#00d4aa")
            st.dataframe(
                results[["Abbreviation","TeamName","Position","GridPosition","Points","Status"]],
                use_container_width=True, hide_index=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RACE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Race Predictor":
    st.markdown('<div class="main-header">Race Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-powered podium probability · XGBoost model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <div class="info-banner-label">Model Info</div>
        <div class="info-banner-text">
            Trained on 2022–2024 season data · Predicts podium probability from
            grid position, recent form, circuit characteristics &amp; constructor pace
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Train / Refresh Model", type="primary"):
        with st.spinner("Fetching training data (2022-2024)..."):
            df = get_historical_results(2022, 2024)
        with st.spinner("Training XGBoost model..."):
            model = train_model(df)
            st.session_state["model"] = model
        st.success("Model trained successfully!")

    try:
        if "model" not in st.session_state:
            model = load_model()
            st.session_state["model"] = model
            st.info("Loaded existing model from disk.")
    except:
        st.warning("No model found — click 'Train / Refresh Model' first.")
        st.stop()

    model     = st.session_state["model"]
    standings = get_driver_standings(season_year)

    section_header("Configure Next Race")
    c1, c2 = st.columns(2)
    with c1:
        round_num = st.slider("Round number", 1, 24, 4)
    with c2:
        circuit_type = st.selectbox(
            "Circuit type", ["high_downforce","street","power","technical"]
        )

    circuit_code = {"high_downforce":0,"street":3,"power":2,"technical":4}.get(circuit_type,0)

    section_header("Set Grid Positions", color="#ff8c00")
    drivers_list = standings["driver"].tolist()[:10]
    cols  = st.columns(5)
    grids = {}
    for i, driver in enumerate(drivers_list):
        with cols[i % 5]:
            grids[driver] = st.number_input(
                driver, min_value=1, max_value=20, value=i+1, key=f"grid_{driver}"
            )

    if st.button("Predict Podium", type="primary"):
        from app.models.feature_engineering import build_training_features
        from app.models.race_predictor import FEATURES, predict_race

        with st.spinner("Building features..."):
            hist    = get_historical_results(2022, 2024)
            feat_df = build_training_features(hist)

        rows = []
        for driver in drivers_list:
            dh  = feat_df[feat_df["driver"] == driver]
            row = dh.iloc[-1][FEATURES].to_dict() if len(dh) > 0 else {f: 0.0 for f in FEATURES}
            row.update({
                "driver": driver,
                "grid": grids[driver],
                "grid_squared": grids[driver] ** 2,
                "circuit_type_code": circuit_code,
                "round": round_num,
                "year": season_year,
            })
            rows.append(row)

        predictions = predict_race(model, pd.DataFrame(rows))

        section_header("Podium Predictions", color="#00d4aa")
        fig = px.bar(
            predictions.head(10), x="driver", y="podium_probability",
            color="podium_probability",
            color_continuous_scale=["#1a1a1a","#e10600"],
            text=predictions.head(10)["podium_probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=400, xaxis_title="Driver", yaxis_title="Podium Probability",
            showlegend=False, coloraxis_showscale=False,
            font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#1a1a1a"), yaxis=dict(gridcolor="#1a1a1a"),
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#aaa"))
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Podium prob: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            predictions[["driver","podium_probability","predicted_position"]].head(10),
            use_container_width=True, hide_index=True
        )

        # SHAP Explainability
        section_header("Why did the model predict this?", color="#7c4dff")
        st.markdown("""
        <div class="info-banner">
            <div class="info-banner-text">
                SHAP values show which features pushed each driver's podium probability
                up (green ↑) or down (red ↓). Bigger bar = bigger impact on the prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            explain_driver = st.selectbox(
                "Explain prediction for driver:",
                predictions["driver"].tolist(),
                key="shap_driver"
            )
            shap_df = get_shap_explanation(model, pd.DataFrame(rows), FEATURES)
            factors = get_top_factors(shap_df, explain_driver, top_n=6)

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                x=factors["shap_value"],
                y=factors["feature"],
                orientation="h",
                marker_color=["#00d4aa" if v > 0 else "#e10600" for v in factors["shap_value"]],
                text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in factors["shap_value"]],
                textposition="outside",
                textfont=dict(color="#aaa", size=11),
            ))
            fig_shap.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=320,
                xaxis=dict(
                    gridcolor="#1a1a1a", zeroline=True,
                    zerolinecolor="#333", zerolinewidth=1,
                    title="Impact on podium probability",
                    tickfont=dict(color="#555"),
                ),
                yaxis=dict(tickfont=dict(color="#aaa")),
                font=dict(color="#ccc"), margin=dict(l=140, r=80),
            )
            fig_shap.update_traces(hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>")
            st.plotly_chart(fig_shap, use_container_width=True)

            pos = factors[factors["shap_value"] > 0]
            neg = factors[factors["shap_value"] < 0]
            helped  = ", ".join([f"{r['feature']} (+{r['shap_value']:.3f})" for _, r in pos.head(3).iterrows()])
            hurt    = ", ".join([f"{r['feature']} ({r['shap_value']:.3f})" for _, r in neg.head(2).iterrows()])
            reasoning = ""
            if helped:
                reasoning += f"<b style='color:#888'>Helped by:</b> <span style='color:#00d4aa'>{helped}</span>"
            if hurt:
                reasoning += f"<br><b style='color:#888'>Hurt by:</b> <span style='color:#e10600'>{hurt}</span>"
            st.markdown(
                f"<div style='background:#0f0f0f;border:1px solid #1a1a1a;border-radius:10px;"
                f"padding:1rem 1.2rem;margin-top:0.5rem;'>"
                f"<div style='color:#fff;font-size:0.85rem;font-weight:600;margin-bottom:6px;'>"
                f"{explain_driver} — model reasoning</div>"
                f"<div style='color:#888;font-size:0.82rem;line-height:1.7;'>{reasoning}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.info(f"SHAP analysis unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEASON CHAMPIONSHIP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Season Championship":
    st.markdown('<div class="main-header">Championship Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Monte Carlo simulation · 10,000 season scenarios</div>', unsafe_allow_html=True)

    with st.spinner("Loading standings..."):
        schedule     = get_season_schedule(season_year)
        constructors = get_constructor_standings(season_year)

    total_rounds = len(schedule)
    done         = races_completed(schedule)

    if season_year == 2026:
        st.markdown(f"""
        <div class="hero-card" style="padding:1.2rem 1.5rem;margin:1rem 0;">
            <div class="live-badge"><span class="live-dot"></span>Live Simulation</div>
            <div style="color:#aaa;font-size:0.88rem;">
                Simulating the <b style="color:#fff;">2026 season</b> —
                {done} races completed, <b style="color:#e10600;">{total_rounds - done} races remaining</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        current_round = st.slider(
            "Rounds completed", 1, total_rounds,
            min(done, total_rounds) if done > 0 else 1
        )
    with c2:
        remaining = total_rounds - current_round
        st.metric("Races remaining", remaining)
    with c3:
        n_sims = st.selectbox("Simulations", [1000, 5000, 10000], index=1)

    noise = st.slider("Upset factor (chaos level)", 0.05, 0.40, 0.20)

    if st.button("Run Championship Simulation", type="primary"):
        with st.spinner(f"Fetching round {current_round} standings..."):
            round_standings = get_driver_standings(season_year, round_num=current_round)

        if round_standings.empty:
            st.error("No standings data found. Try a lower round number.")
            st.stop()

        strengths = build_driver_strengths(round_standings)

        with st.spinner(f"Running {n_sims:,} season simulations..."):
            import numpy as np
            POINTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
            drivers_  = round_standings["driver"].tolist()
            base_pts  = dict(zip(round_standings["driver"], round_standings["points"]))
            win_counts    = {d: 0 for d in drivers_}
            final_pts_sum = {d: 0.0 for d in drivers_}

            for _ in range(n_sims):
                season_pts = base_pts.copy()
                for __ in range(remaining):
                    scores = {d: max(0, s + np.random.normal(0, noise)) for d, s in strengths.items()}
                    order  = sorted(scores, key=scores.get, reverse=True)
                    for pos, d in enumerate(order, 1):
                        if d in season_pts:
                            season_pts[d] = season_pts.get(d,0) + POINTS.get(pos,0)
                            if pos == 1: season_pts[d] += 1
                champ = max(season_pts, key=season_pts.get)
                win_counts[champ] += 1
                for d in drivers_: final_pts_sum[d] += season_pts.get(d,0)

            results = pd.DataFrame([{
                "driver": d,
                "wdc_probability": round(win_counts[d]/n_sims*100, 1),
                "avg_final_points": round(final_pts_sum[d]/n_sims, 1),
                "current_points": base_pts.get(d, 0),
            } for d in drivers_]).sort_values("wdc_probability", ascending=False).reset_index(drop=True)

        st.session_state["sim_results"]   = results
        st.session_state["sim_standings"] = round_standings

    if "sim_results" in st.session_state:
        results         = st.session_state["sim_results"]
        round_standings = st.session_state["sim_standings"]
        winner          = results.iloc[0]

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#120000,#180808);
                    border:1px solid #3a1010;border-left:4px solid #e10600;
                    border-radius:14px;padding:1.4rem 1.8rem;margin:1.2rem 0;">
            <div style="color:#666;font-size:0.65rem;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px;">Model Prediction</div>
            <div style="color:#fff;font-family:'Barlow Condensed',sans-serif;
                        font-size:2rem;font-weight:900;letter-spacing:-0.5px;">
                {winner['driver']} wins the {season_year} championship
            </div>
            <div style="color:#e10600;font-size:1rem;font-weight:700;margin-top:4px;">
                {winner['wdc_probability']}% probability
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("WDC Win Probability")
        top = results[results["wdc_probability"] > 0]
        fig = px.bar(
            top, x="driver", y="wdc_probability",
            color="wdc_probability", color_continuous_scale=["#1a1a1a","#e10600"],
            text=top["wdc_probability"].apply(lambda x: f"{x}%"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=420, xaxis_title="Driver",
            yaxis_title="Championship Win Probability (%)",
            coloraxis_showscale=False, font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#1a1a1a"), yaxis=dict(gridcolor="#1a1a1a"),
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#aaa"))
        fig.update_traces(hovertemplate="<b>%{x}</b><br>WDC prob: %{y}%<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            section_header("Current vs Projected Points", color="#ff8c00")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Current", x=results["driver"],
                                   y=results["current_points"], marker_color="#2a2a2a"))
            fig2.add_trace(go.Bar(name="Projected", x=results["driver"],
                                   y=results["avg_final_points"], marker_color="#e10600", opacity=0.85))
            fig2.update_layout(
                barmode="group", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=350,
                font=dict(color="#ccc"), legend=dict(font=dict(color="#aaa"), orientation="h"),
                xaxis=dict(gridcolor="#1a1a1a"), yaxis=dict(gridcolor="#1a1a1a"),
            )
            fig2.update_traces(hovertemplate="<b>%{x}</b><br>Points: %{y}<extra></extra>")
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            section_header("Points gap to leader", color="#00d4aa")
            leader_pts    = round_standings.iloc[0]["points"]
            gap_df        = round_standings.copy()
            gap_df["gap"] = leader_pts - gap_df["points"]
            fig3 = px.bar(
                gap_df.head(10), x="driver", y="gap",
                color="gap", color_continuous_scale=["#e10600","#2a2a2a"], text="gap",
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=350, coloraxis_showscale=False, font=dict(color="#ccc"),
                xaxis=dict(gridcolor="#1a1a1a"), yaxis=dict(gridcolor="#1a1a1a"),
                yaxis_title="Points behind leader",
            )
            fig3.update_traces(textposition="outside", textfont=dict(color="#aaa"))
            fig3.update_traces(hovertemplate="<b>%{x}</b><br>Gap: %{y} pts<extra></extra>")
            st.plotly_chart(fig3, use_container_width=True)

        section_header("Full simulation breakdown")
        st.dataframe(results, use_container_width=True, hide_index=True)
        section_header("Constructors Championship", color="#00d4aa")
        st.dataframe(constructors, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIVER DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Driver Dynamics":
    st.markdown('<div class="main-header">Driver Dynamics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Circuit fingerprints · Strengths & weaknesses per driver</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <div class="info-banner-label">How it works</div>
        <div class="info-banner-text">
            Each driver gets a fingerprint across 6 dimensions — performance at street circuits,
            power tracks, technical circuits, high-downforce venues, consistency, and race craft
            (positions gained from grid). Built from historical race data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Building driver dynamics profiles from historical data..."):
        hist = get_historical_results(2022, 2024)
        dna  = build_driver_dna(hist)

    DIMENSIONS  = ["street","power","technical","high_downforce","consistency","race_craft"]
    DIM_LABELS  = ["Street","Power","Technical","High Downforce","Consistency","Race Craft"]
    dim_icons   = {"street":"🏙️","power":"⚡","technical":"🔧",
                   "high_downforce":"🌀","consistency":"📊","race_craft":"🎯"}
    colors      = ["#e10600","#00d4aa","#7c4dff","#ff8c00","#00b0ff",
                   "#ff4081","#69f0ae","#ffeb3b","#40c4ff","#ff6d00"]

    section_header("Compare Drivers", color="#7c4dff")
    available = dna["driver"].tolist()
    defaults  = [d for d in ["VER","NOR","LEC","RUS","HAM"] if d in available][:5]
    selected  = st.multiselect("Select drivers to compare", available, default=defaults)

    if selected:
        fig = go.Figure()
        for i, driver in enumerate(selected):
            row = dna[dna["driver"] == driver]
            if row.empty:
                continue
            values = [float(row[d].iloc[0]) for d in DIMENSIONS]
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(
                r=values, theta=DIM_LABELS + [DIM_LABELS[0]],
                fill="toself", fillcolor=colors[i % len(colors)],
                opacity=0.15, line=dict(color=colors[i % len(colors)], width=2),
                name=driver,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#1a1a1a",
                                linecolor="#1a1a1a", tickfont=dict(color="#444", size=9),
                                tickvals=[25,50,75,100]),
                angularaxis=dict(gridcolor="#1a1a1a", linecolor="#222",
                                 tickfont=dict(color="#aaa", size=11)),
            ),
            showlegend=True,
            legend=dict(font=dict(color="#aaa"), orientation="h", y=-0.15, x=0.5, xanchor="center"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=520, margin=dict(t=40, b=60),
            font=dict(color="#ccc"),
        )
        fig.update_traces(hovertemplate="<b>%{theta}</b>: %{r}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        section_header("Driver Scorecards", color="#7c4dff")
        cols = st.columns(len(selected))
        for i, driver in enumerate(selected):
            row = dna[dna["driver"] == driver]
            if row.empty:
                continue
            with cols[i]:
                color     = colors[i % len(colors)]
                best_dim  = max(DIMENSIONS, key=lambda d: float(row[d].iloc[0]))
                worst_dim = min(DIMENSIONS, key=lambda d: float(row[d].iloc[0]))
                card = f'<div style="background:#0f0f0f;border:1px solid #1a1a1a;border-top:3px solid {color};border-radius:10px;padding:12px 14px;">'
                card += f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.2rem;font-weight:800;color:#fff;margin-bottom:10px;">{driver}</div>'
                for d in DIMENSIONS:
                    val   = float(row[d].iloc[0])
                    icon  = dim_icons.get(d, "")
                    label = d.replace("_", " ").title()
                    card += f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;"><span style="color:#555;font-size:0.68rem;">{icon} {label}</span><div style="display:flex;align-items:center;gap:6px;"><div style="width:60px;height:4px;background:#1a1a1a;border-radius:2px;"><div style="width:{val}%;height:100%;background:{color};border-radius:2px;opacity:0.8;"></div></div><span style="color:#888;font-size:0.68rem;min-width:28px;text-align:right;">{val:.0f}</span></div></div>'
                card += f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #1a1a1a;"><div style="color:#444;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Profile</div><div style="color:#00d4aa;font-size:0.75rem;font-weight:600;">↑ {best_dim.replace("_"," ").title()}</div><div style="color:#e10600;font-size:0.75rem;font-weight:600;margin-top:2px;">↓ {worst_dim.replace("_"," ").title()}</div></div></div>'
                st.markdown(card, unsafe_allow_html=True)

    section_header("All Drivers", color="#444")
    display_dna = dna[["driver"] + DIMENSIONS].copy()
    display_dna.columns = ["Driver","Street","Power","Technical","High Downforce","Consistency","Race Craft"]
    st.dataframe(display_dna, use_container_width=True, hide_index=True)

# Close page-enter animation wrapper
st.markdown('</div>', unsafe_allow_html=True)