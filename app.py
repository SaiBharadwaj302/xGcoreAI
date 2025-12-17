import streamlit as st
import os
import sys

# --- 1. SYSTEM PATH CONFIGURATION ---
# This block ensures 'utils' and 'app' modules are discoverable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- 2. MODULE IMPORTS ---
from utils.config import CSS_STYLE
from utils.state import load_global_data, load_league_data, get_league_file_map
from app.tabs import TabContext, render_simulation_tab, render_squad_genome_tab
from app.radar import render_radar_tab
from app.sniper import render_sniper_tab

# --- 3. APP CONFIGURATION ---
st.set_page_config(page_title="xGcoreAI", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 4. INITIALIZATION & DATA LOADING ---
shots_global, stats_global, models_map = load_global_data()
league_map = get_league_file_map(ROOT_DIR)

# --- 5. SIDEBAR / HEADER ---
c1, c2 = st.columns([3, 1])
with c1: 
    st.title("xG CoreAI")
    st.caption("Advanced Expected Goals Modeling & Squad Optimization")

with c2: 
    if not league_map: 
        st.error("No data found. Please run preprocessing first.")
        st.stop()
    active_league = st.selectbox("LEAGUE", sorted(league_map.keys()))

# Load Active League Data
f_shots, f_stats = load_league_data(ROOT_DIR, active_league, league_map)

# --- 6. MODEL INFERENCE ---
def sanitize_key(k: str) -> str:
    return str(k).lower().strip().replace(' ', '_')

model_key = sanitize_key(active_league)
# Attempt strict match, then file_key match
model = models_map.get(model_key) or models_map.get(sanitize_key(league_map.get(active_league, "")))

if model and not f_shots.empty:
    # Ensure correct feature order for XGBoost
    feats = model.feature_names 
    X = f_shots.reindex(columns=feats, fill_value=0)
    try:
        f_shots['model_xg'] = model.predict(X)
    except Exception as e:
        # Fallback to 0.0 silently, or log warning in debug mode
        f_shots['model_xg'] = 0.0

# --- 7. TAB RENDERING ---
# Context object passes state to tabs cleanly
ctx = TabContext(
    f_shots=f_shots, 
    normalized_stats=f_stats, 
    normalized_shots=f_shots,
    shots_df=shots_global, 
    league_file_map=league_map, 
    active_league=active_league,
    file_key=league_map.get(active_league), 
    model=model, 
    calibrator=None
)

t1, t2, t3, t4 = st.tabs(["🎯 SIMULATION", "🧬 BEST XI", "🧠 RADAR", "🥅 SNIPER MAP"])

render_simulation_tab(t1, ctx)
render_squad_genome_tab(t2, ctx, ["4-3-3", "4-4-2", "3-5-2", "4-2-3-1"])

with t3:
    render_radar_tab(f_shots)

with t4:
    render_sniper_tab(active_league, f_shots, league_map, ROOT_DIR)