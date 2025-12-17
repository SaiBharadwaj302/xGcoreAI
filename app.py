import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import re
from typing import Optional

# --- 1. PATH FIX (ROOT LEVEL) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- 2. ROBUST IMPORTS ---
try:
    if not os.path.exists(os.path.join(ROOT_DIR, 'utils')):
        st.error(f"🚨 **CRITICAL ERROR:** The `utils` folder is missing from `{ROOT_DIR}`.")
        st.stop()

    from utils.config import CSS_STYLE
    from utils.data_loader import load_resources
    from utils.visualisations import draw_cyber_pitch
    from app.tabs import TabContext, render_simulation_tab, render_squad_genome_tab
    from matplotlib.patches import ConnectionPatch

except ImportError as e:
    st.error(f"❌ **Import Error:** {e}")
    st.stop()

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="VantagePoint Ultra", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 4. LOAD DATA ---
try:
    shots_df, stats_df, models_map = load_resources()
except Exception as e:
    print(f"Global load skipped: {e}")
    shots_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    models_map = {}

# Prefer using per-league CSVs in Data/processed/leagues
LEAGUES_DIR = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues')
league_file_map = {} 

if os.path.exists(LEAGUES_DIR):
    for fn in os.listdir(LEAGUES_DIR):
        if fn.startswith('shots_') and fn.endswith('.csv'):
            key = fn[len('shots_'):-len('.csv')]
            display = key.replace('_', ' ')
            league_file_map[display] = key

# --- 5. UI HEADER ---
c1, c2 = st.columns([3, 1])
with c1: 
    st.title("xG CoreAI")
    st.caption("Where xG meets Intelligence")
with c2:
    if not league_file_map:
        st.error("❌ No per-league CSVs found. Please run preprocessing first.")
        st.stop()

    league_options = sorted(list(league_file_map.keys()))
    active_league = st.selectbox("SOURCE (Select League)", league_options)

# Load per-league data
file_key = league_file_map.get(active_league)
shots_path = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"shots_{file_key}.csv")
stats_path = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"stats_{file_key}.csv")

if not os.path.exists(shots_path):
    st.error(f"❌ Missing per-league shots file: {shots_path}")
    st.stop()

f_shots = pd.read_csv(shots_path)
f_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else pd.DataFrame()

# Establish unified active dataframes
active_shots = f_shots.copy()
active_stats = f_stats if not f_stats.empty else pd.DataFrame()

# --- HELPER FUNCTIONS ---
def _ensure_season_norm(df):
    if df is None: return pd.DataFrame(columns=['season_norm'])
    df = df.copy()
    # Normalize simply to the 'season_name' or 'season' column string
    if 'season_name' in df.columns:
        df['season_norm'] = df['season_name'].astype(str).str.strip()
    elif 'season' in df.columns:
        df['season_norm'] = df['season'].astype(str).str.strip()
    else:
        df['season_norm'] = 'Unknown'
    return df

normalized_stats = _ensure_season_norm(active_stats)
normalized_shots = _ensure_season_norm(f_shots)

# --- MODEL LOOKUP ---
def sanitize_key(name):
    return str(name).lower().strip().replace(' ', '_')

smart_model_map = {sanitize_key(k): v for k, v in models_map.items()}
target_key = sanitize_key(active_league)
model = smart_model_map.get(target_key) or smart_model_map.get(sanitize_key(file_key))

if model is None:
    st.error(f"❌ No trained model found for league '{active_league}'.")
    st.stop()

# Determine expected feature order
expected_features = [
    'start_x', 'start_y', 'distance', 'visible_angle', 'body_part_code', 
    'technique_code', 'angle_sin', 'angle_cos', 'dist_to_goal_center', 
    'is_header', 'start_x_norm', 'start_y_norm', 'player_last5_conv'
]

# --- PREDICTION LOGIC ---
if not f_shots.empty:
    for col in ['body_part_code', 'technique_code', 'visible_angle', 'distance', 'start_x', 'start_y']:
        if col not in f_shots.columns:
            f_shots[col] = 0
            
    from utils.helpers import _add_engineered_features
    try:
        f_shots = _add_engineered_features(f_shots)
    except Exception:
        pass

    X_local = pd.DataFrame(index=f_shots.index)
    for feat in expected_features:
        X_local[feat] = f_shots[feat] if feat in f_shots.columns else 0
    X_local = X_local.fillna(0)

    try:
        raw_preds = model.predict(X_local)
        if isinstance(raw_preds, (list, np.ndarray)):
            if len(np.array(raw_preds).shape) > 1:
                f_shots['model_xg'] = np.array(raw_preds).flatten()
            else:
                f_shots['model_xg'] = raw_preds
        else:
            f_shots['model_xg'] = raw_preds
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        f_shots['model_xg'] = 0.0

# --- TABS ---
tab_ctx = TabContext(
    f_shots=f_shots,
    normalized_stats=normalized_stats,
    normalized_shots=normalized_shots,
    shots_df=shots_df,
    league_file_map=league_file_map,
    active_league=active_league,
    file_key=file_key,
    model=model,
    calibrator=None, 
)

t1, t2, t3, t4 = st.tabs(["🎯 SIMULATION", "🧬 BEST XI", "🧠 RADAR", "🥅 SNIPER MAP"])

render_simulation_tab(t1, tab_ctx)
FORMATIONS = ["4-3-3", "4-4-2", "3-5-2", "4-5-1", "4-2-3-1", "4-1-4-1", "3-4-3", "5-3-2"]
render_squad_genome_tab(t2, tab_ctx, FORMATIONS)

# --- TAB 3: RADAR ---
with t3:
    st.markdown("### 🧠 MODEL CONFIDENCE RADAR")
    if 'model_xg' not in f_shots.columns:
        st.info('Model confidence data is missing.')
    else:
        player_metrics = (
            f_shots.groupby('player_name')
            .agg(actual_goals=('is_goal', 'sum'), expected_goals=('model_xg', 'sum'), attempts=('start_x', 'count'))
            .reset_index()
        )
        player_metrics = player_metrics[player_metrics['attempts'] >= 5]
        
        if player_metrics.empty:
            st.info('Need more shot data.')
        else:
            player_metrics['confidence_gap'] = (player_metrics['actual_goals'] - player_metrics['expected_goals']).abs()
            highlight = player_metrics.sort_values('confidence_gap', ascending=False).head(6)
            
            if highlight.empty:
                st.info('No standout players yet.')
            else:
                col_ctrl, col_viz = st.columns([1, 2])
                with col_ctrl:
                    focus_options = ['Top 6 Players'] + highlight['player_name'].tolist()
                    focus_player = st.selectbox('Focus Player', focus_options)
                    
                    st.markdown("#### Data")
                    display_df = highlight[['player_name', 'actual_goals', 'expected_goals']].rename(columns={'player_name':'Player', 'actual_goals':'Goals', 'expected_goals':'xG'})
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

                with col_viz:
                    if focus_player == 'Top 6 Players':
                        categories = highlight['player_name'].tolist()
                        actual = highlight['actual_goals'].tolist()
                        expected = highlight['expected_goals'].tolist()
                        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                        angles += angles[:1]
                        actual += actual[:1]
                        expected += expected[:1]
                        
                        max_val = max(max(actual), max(expected), 1)
                        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'polar': True})
                        ax.set_theta_offset(np.pi / 2)
                        ax.set_theta_direction(-1)
                        ax.set_ylim(0, max_val * 1.1)
                        ax.plot(angles, actual, linewidth=2, label='Actual Goals', color='#ff0055')
                        ax.fill(angles, actual, color='#ff0055', alpha=0.25)
                        ax.plot(angles, expected, linewidth=2, label='Expected xG', color='#00f3ff')
                        ax.fill(angles, expected, color='#00f3ff', alpha=0.25)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories, fontsize=9)
                        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                        st.pyplot(fig, use_container_width=False)
                    else:
                        row = highlight[highlight['player_name'] == focus_player].iloc[0]
                        metrics = {'Actual Goals': row['actual_goals'], 'Expected xG': row['expected_goals'], 'Gap': row['confidence_gap']}
                        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                        values = list(metrics.values())
                        values += values[:1]
                        angles += angles[:1]
                        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'polar': True})
                        ax.plot(angles, values, color='#00f3ff', linewidth=2)
                        ax.fill(angles, values, color='#00f3ff', alpha=0.3)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(list(metrics.keys()))
                        ax.set_title(f"{focus_player}", color='white')
                        st.pyplot(fig, use_container_width=False)

# --- TAB 4: SNIPER MAP ---
with t4:
    st.markdown("### 🥅 SNIPER MAP")
    
    # 1. FILTERS
    f1, f2, f3, f4 = st.columns(4)
    
    # A. League
    with f1:
        leagues = sorted(list(league_file_map.keys()))
        sel_league = st.selectbox("LEAGUE", leagues, index=leagues.index(active_league) if active_league in leagues else 0)

    # B. Team
    with f2:
        if sel_league == active_league:
            current_data = f_shots
        else:
            try:
                other_key = league_file_map.get(sel_league)
                other_path = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"shots_{other_key}.csv")
                current_data = pd.read_csv(other_path)
            except Exception:
                current_data = f_shots 

        if 'team_name' in current_data.columns:
            teams = sorted(current_data['team_name'].dropna().astype(str).unique())
        elif 'team' in current_data.columns:
            teams = sorted(current_data['team'].dropna().astype(str).unique())
        else:
            teams = []
        sel_team = st.selectbox("TEAM", ["All"] + teams)

    # C. Player
    with f3:
        if sel_team != "All":
            if 'team_name' in current_data.columns:
                players_df = current_data[current_data['team_name'] == sel_team]
            elif 'team' in current_data.columns:
                players_df = current_data[current_data['team'] == sel_team]
            else:
                players_df = current_data
        else:
            players_df = current_data
            
        players = sorted(players_df['player_name'].dropna().unique())
        default_idx = players.index("Lionel Messi") if "Lionel Messi" in players else 0
        sel_player = st.selectbox("PLAYER", players, index=default_idx if players else 0)

    # D. Season (Using simple string matching on season_name)
    with f4:
        # Determine which column holds the season string
        # Priority: season_name -> season
        season_col = 'season_name' if 'season_name' in players_df.columns else 'season'
        
        # Filter for the selected player
        p_data = players_df[players_df['player_name'] == sel_player]
        
        if season_col in p_data.columns:
            # Get unique seasons (strings like "2015/2016") and sort descending
            available_seasons = sorted(p_data[season_col].dropna().unique().astype(str), reverse=True)
            # Filter out any weird "nan" strings just in case
            available_seasons = [s for s in available_seasons if s.lower() != 'nan']
        else:
            available_seasons = []
        
        if available_seasons:
            sel_season = st.selectbox("SEASON", available_seasons)
        else:
            sel_season = None
            st.warning("No season info.")

    # 2. VISUALIZATION
    if sel_season:
        # Strict string filtering
        p_shots = p_data[p_data[season_col].astype(str) == sel_season]

        if not p_shots.empty:
            goals = p_shots[p_shots['is_goal'] == 1]
            misses = p_shots[p_shots['is_goal'] == 0]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("TOTAL ATTEMPTS", len(p_shots))
            m2.metric("GOALS SCORED", len(goals))
            conversion = (len(goals) / len(p_shots)) * 100 if len(p_shots) > 0 else 0
            m3.metric("CONVERSION RATE", f"{conversion:.1f}%")

            fig, ax = draw_cyber_pitch()
            for _, row in goals.iterrows():
                con = ConnectionPatch(xyA=(row['start_x'], row['start_y']), xyB=(120, 40), 
                                      coordsA="data", coordsB="data", axesA=ax, axesB=ax, 
                                      color='#00f3ff', alpha=0.6, lw=2)
                ax.add_artist(con)
                
            ax.scatter(goals.start_x, goals.start_y, c='#00f3ff', edgecolors='white', s=150, marker='h', zorder=10, label='Goal')
            ax.scatter(misses.start_x, misses.start_y, c='#f43f5e', alpha=0.3, s=50, zorder=5, label='Miss')
            ax.legend(facecolor='#050505', labelcolor='white', loc='lower center', ncol=2)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info(f"No shot data found for {sel_player} in {sel_season}.")
    else:
        st.info("Please select a valid season.")