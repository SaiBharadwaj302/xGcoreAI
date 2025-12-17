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
# Get the absolute path of the folder containing this app.py file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Force this root directory to the top of the system path
# This ensures Python can find the 'utils' folder sitting right next to app.py
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- 2. ROBUST IMPORTS ---
try:
    # Check if utils exists before importing to give a better error message
    if not os.path.exists(os.path.join(ROOT_DIR, 'utils')):
        st.error(f"🚨 **CRITICAL ERROR:** The `utils` folder is missing from `{ROOT_DIR}`.")
        st.info("Based on your screenshot, `utils` should be in the same folder as `app.py`.")
        st.stop()

    # Imports (Matching your specific filenames from the screenshot)
    from utils.config import CSS_STYLE, TACTICS
    from utils.data_loader import load_resources
    from utils.visualisations import draw_cyber_pitch

    from app.tabs import TabContext, render_simulation_tab, render_squad_genome_tab
    from matplotlib.patches import ConnectionPatch

except ImportError as e:
    st.error(f"❌ **Import Error:** {e}")
    st.write("Debug Information:")
    st.write(f"Script Location: `{ROOT_DIR}`")
    st.write(f"Python Path: `{sys.path[0]}`")
    st.write(f"Files in root: {os.listdir(ROOT_DIR)}")
    if os.path.exists(os.path.join(ROOT_DIR, 'utils')):
        st.write(f"Files in utils: {os.listdir(os.path.join(ROOT_DIR, 'utils'))}")
    st.stop()

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="VantagePoint Ultra", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 4. LOAD DATA ---
# load_resources: shots, stats, models_map, calibrators_map
# We use a try-except block to allow "soft loading" in case global files are missing
try:
    shots_df, stats_df, models_map, calibrators_map = load_resources()
except Exception as e:
    print(f"Global load skipped: {e}")
    shots_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    models_map = {}
    calibrators_map = {}

# Prefer using per-league CSVs in Data/processed/leagues when available
# NOTE: Casing changed to 'Data' to match server structure
LEAGUES_DIR = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues')
league_file_map = {}  # display_name -> file_key (sanitized)

if os.path.exists(LEAGUES_DIR):
    for fn in os.listdir(LEAGUES_DIR):
        if fn.startswith('shots_') and fn.endswith('.csv'):
            key = fn[len('shots_'):-len('.csv')]
            display = key.replace('_', ' ')
            league_file_map[display] = key

    # Helper to load per-league stats consistently across the app
    def get_stats_for_league(display_name):
        # try per-league CSV first
        fk = league_file_map.get(display_name)
        if fk:
            # NOTE: Casing changed to 'Data'
            sp = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"stats_{fk}.csv")
            if os.path.exists(sp):
                try:
                    base_stats = pd.read_csv(sp)
                    # If there is a corresponding shots file, ensure any high-impact
                    # players present in shots but missing from stats are included.
                    try:
                        shots_p = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"shots_{fk}.csv")
                        if os.path.exists(shots_p):
                            sdet = pd.read_csv(shots_p)
                            # aggregate basic tallies from shots for players not in base_stats
                            missing_players = set(sdet['player_name'].dropna().unique()) - set(base_stats['player_name'].astype(str).dropna().unique())
                            if missing_players:
                                agg = sdet[sdet['player_name'].isin(missing_players)].groupby('player_name').agg({'is_goal':'sum', 'start_x':'count'}).rename(columns={'start_x':'shots'}).reset_index()
                                rows = []
                                for _, r in agg.iterrows():
                                    pname = r['player_name']
                                    pr = sdet[sdet['player_name'] == pname]
                                    primary_pos = pr['primary_pos'].mode().iloc[0] if 'primary_pos' in pr.columns and not pr['primary_pos'].mode().empty else ''
                                    goals = int(r['is_goal']) if 'is_goal' in r else 0
                                    shots_count = int(r['shots'])
                                    att_score = int(shots_count * 5 + goals * 50)
                                    rows.append({'player_name':pname, 'league': display_name, 'team_name': pr['team_name'].mode().iloc[0] if 'team_name' in pr.columns and not pr['team_name'].mode().empty else '', 'primary_pos': primary_pos, 'goals': goals, 'shots': shots_count, 'passes':0, 'tackles':0, 'interceptions':0, 'clearances':0, 'blocks':0, 'def_score':0, 'mid_score':0, 'att_score':att_score, 'extracted_year':0, 'season': 'All Time'})
                                if rows:
                                    try:
                                        base_stats = pd.concat([base_stats, pd.DataFrame(rows)], ignore_index=True, sort=False)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    return base_stats
                except Exception:
                    pass
        # next, try global stats_df filtered by league
        try:
            if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                cand = stats_df[(stats_df['league'].astype(str).str.lower() == display_name.lower()) | (stats_df['league'].astype(str).str.lower() == (fk or '').replace('_',' ').lower())]
                if not cand.empty:
                    return cand.copy()
        except Exception:
            pass
        # last resort: build minimal stats from global shots_df filtered by league
        try:
            sd = shots_df[shots_df['league'].astype(str).str.lower() == display_name.lower()]
            if sd.empty:
                return pd.DataFrame()
            agg = sd.groupby('player_name').agg({'is_goal':'sum', 'start_x':'count'}).rename(columns={'start_x':'shots'}).reset_index()
            rows = []
            for _, r in agg.iterrows():
                pname = r['player_name']
                pr = sd[sd['player_name'] == pname]
                primary_pos = pr['primary_pos'].mode().iloc[0] if 'primary_pos' in pr.columns and not pr['primary_pos'].mode().empty else ''
                goals = int(r['is_goal']) if 'is_goal' in r else 0
                shots_count = int(r['shots'])
                att_score = int(shots_count * 5 + goals * 50)
                rows.append({'player_name':pname, 'league': display_name, 'team_name': pr['team_name'].mode().iloc[0] if 'team_name' in pr.columns and not pr['team_name'].mode().empty else '', 'primary_pos': primary_pos, 'goals': goals, 'shots': shots_count, 'passes':0, 'tackles':0, 'interceptions':0, 'clearances':0, 'blocks':0, 'def_score':0, 'mid_score':0, 'att_score':att_score, 'extracted_year':0, 'season': 'All Time'})
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()


# --- 5. UI HEADER ---
c1, c2 = st.columns([3, 1])
with c1: 
    st.title("xG CoreAI")
    st.caption("Where xG meets Intelligence")
with c2:
    # Require per-league files to be present. Do not offer a global option anymore.
    if not league_file_map:
        st.error("❌ No per-league CSVs found in `Data/processed/leagues/`. Place `shots_{league}.csv` files there and restart.")
        st.stop()

    league_options = sorted(list(league_file_map.keys()))
    active_league = st.selectbox("SOURCE (Select League)", league_options)
    # Display model metrics for the selected league if available
    try:
        MODEL_DIR = os.path.join(ROOT_DIR, 'models')
        manifest_path = os.path.join(MODEL_DIR, 'manifest.json')
        model_metrics = None
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as mf:
                manifest = json.load(mf)
            key = league_file_map.get(active_league)
            if key in manifest:
                model_metrics = manifest[key]
        # also try training_report_{key}.json for brier/CV
        if model_metrics is None:
            key = league_file_map.get(active_league)
            tr = os.path.join(MODEL_DIR, f'training_report_{key}.json')
            if os.path.exists(tr):
                with open(tr, 'r', encoding='utf-8') as fh:
                    model_metrics = json.load(fh)
        if model_metrics:
            mm = model_metrics
            auc = mm.get('auc')
            acc = mm.get('accuracy') or mm.get('acc')
            brier = mm.get('brier')
            cv_auc = mm.get('cv_auc')
            with st.expander('Model metrics', expanded=True):
                if auc is not None: st.write(f"- AUC: {auc:.3f}")
                if acc is not None: st.write(f"- Accuracy: {acc:.3f}")
                if brier is not None: st.write(f"- Brier: {brier:.4f}")
                if cv_auc is not None: st.write(f"- CV AUC: {cv_auc:.3f}")
    except Exception:
        pass

from utils.helpers import sanitize, _add_engineered_features

# Supported formations (keeps UI in sync with `utils.simulations` additions)
FORMATIONS = [
    "4-3-3", "4-4-2", "3-5-2", "4-5-1", "4-2-3-1", "4-1-4-1", "3-4-3", "5-3-2", "4-4-1-1"
]

def _ensure_season_norm(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=['season_norm'])
    df = df.copy()
    if 'season_norm' in df.columns:
        return df
    if 'season' in df.columns:
        df['season_norm'] = df['season'].fillna('All Time').astype(str).str.strip()
    elif 'season_name' in df.columns:
        df['season_norm'] = df['season_name'].fillna('All Time').astype(str).str.strip()
    else:
        df['season_norm'] = 'All Time'
    return df

def _filter_by_season(df: pd.DataFrame, season_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = _ensure_season_norm(df)
    target = str(season_choice or 'All Time').strip() or 'All Time'
    season_series = normalized['season_norm'].fillna('All Time').astype(str)
    mask = season_series.str.lower() == target.lower()
    if not mask.any():
        mask = season_series.str.lower().str.contains(target.lower())
    return normalized[mask].copy()


def _season_sort_key(value: Optional[str]) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    try:
        if '/' in text:
            return int(text.split('/')[0])
        if text.isdigit():
            return int(text)
    except Exception:
        pass
    if text == 'All Time':
        return -1
    return 0


def _extract_year_from_season_text(text: Optional[str]) -> int:
    if not text:
        return 0
    match = re.search(r"(\d{4})", str(text))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _ensure_year_column(df: pd.DataFrame, target_col: str = 'season_year') -> pd.DataFrame:
    if df is None or df.empty:
        return df
    result = df.copy()
    year_series = pd.Series(0, index=result.index, dtype=int)
    if 'extracted_year' in result.columns:
        year_series = result['extracted_year'].fillna(0).astype(int)

    def _fill_from_column(col: str) -> None:
        nonlocal year_series
        if col not in result.columns:
            return
        mask = year_series == 0
        if not mask.any():
            return
        derived = result.loc[mask, col].astype(str).apply(_extract_year_from_season_text).fillna(0).astype(int)
        year_series.loc[mask] = derived

    for candidate in ('season', 'season_name', 'season_norm'):
        _fill_from_column(candidate)

    result[target_col] = year_series
    return result


# Load per-league data from leagues folder when selected; otherwise fall back to global DataFrames
# active_league is a display name; map to file key and require the files
file_key = league_file_map.get(active_league)
# NOTE: Casing changed to 'Data'
shots_path = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"shots_{file_key}.csv")
stats_path = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"stats_{file_key}.csv")

if not os.path.exists(shots_path):
    st.error(f"❌ Missing per-league shots file: {shots_path}")
    st.stop()

f_shots = pd.read_csv(shots_path)

# If per-league shots CSVs don't contain team columns, try to enrich from global `stats_df`
try:
    if 'team_name' not in f_shots.columns and isinstance(stats_df, pd.DataFrame) and 'team_name' in stats_df.columns:
        team_mapping = stats_df[['player_name', 'team_name']].drop_duplicates()
        try:
            f_shots = f_shots.merge(team_mapping, on='player_name', how='left', suffixes=('', '_stats'))
        except Exception:
            pass
except Exception:
    pass

if os.path.exists(stats_path):
    f_stats = pd.read_csv(stats_path)
else:
    st.warning(f"⚠️ No per-league stats file found for {active_league} ({stats_path}). Some features may be unavailable.")
    f_stats = pd.DataFrame()

# Establish unified active dataframes so all tabs use the same per-league source
active_shots = f_shots.copy()
# Prefer per-league stats when present; otherwise try to filter the global stats_df
if not f_stats.empty:
    active_stats = f_stats.copy()
else:
    # try to derive per-league stats from global stats_df loaded by load_resources()
    active_stats = pd.DataFrame()
    try:
        if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
            # Match by league display name or sanitized file_key
            candidates = stats_df.copy()
            if 'league' in candidates.columns:
                matched = candidates[(candidates['league'].astype(str).str.lower() == active_league.lower()) | (candidates['league'].astype(str).str.lower() == file_key.replace('_',' ').lower())]
                if not matched.empty:
                    active_stats = matched.copy()
            # As a fallback, filter by player presence in shots for this league
            if active_stats.empty and 'player_name' in active_shots.columns:
                players = active_shots['player_name'].dropna().unique().tolist()
                active_stats = candidates[candidates['player_name'].isin(players)].copy()
    except Exception:
        active_stats = pd.DataFrame()

    # Last resort: build minimal per-player stats from the shots file (lightweight)
    if active_stats.empty:
        try:
            ss = active_shots.copy()
            if 'season' in ss.columns:
                ss['season_norm'] = ss['season'].fillna('All Time').astype(str).str.strip()
            elif 'season_name' in ss.columns:
                ss['season_norm'] = ss['season_name'].fillna('All Time').astype(str).str.strip()
            else:
                ss['season_norm'] = 'All Time'
            agg = ss.groupby('player_name').agg({'is_goal':'sum', 'start_x':'count'}).rename(columns={'start_x':'shots'}).reset_index()
            def mode_col(df, col):
                if col in df.columns and not df[col].dropna().empty:
                    return df[col].mode().iloc[0]
                return ''
            rows = []
            for _, r in agg.iterrows():
                pname = r['player_name']
                player_rows = ss[ss['player_name'] == pname]
                primary_pos = mode_col(player_rows, 'primary_pos')
                team = mode_col(player_rows, 'team') if 'team' in player_rows.columns else mode_col(player_rows, 'team_name') if 'team_name' in player_rows.columns else ''
                goals = int(r['is_goal']) if 'is_goal' in r else int(player_rows['is_goal'].sum()) if 'is_goal' in player_rows.columns else 0
                shots_count = int(r['shots'])
                att_score = int(shots_count * 5 + goals * 50)
                rows.append({'player_name':pname, 'league': active_league, 'team_name': team, 'primary_pos': primary_pos, 'goals': goals, 'shots': shots_count, 'passes':0, 'tackles':0, 'interceptions':0, 'clearances':0, 'blocks':0, 'def_score':0, 'mid_score':0, 'att_score':att_score, 'extracted_year':0, 'season': 'All Time'})
            active_stats = pd.DataFrame(rows)
        except Exception:
            active_stats = pd.DataFrame()

stats_source = f_stats if not f_stats.empty else active_stats
normalized_stats = _ensure_season_norm(stats_source)
normalized_shots = _ensure_season_norm(f_shots)

# --- IMPROVED MODEL LOOKUP LOGIC ---
# 1. Helper function to normalize names (e.g. "1 Bundesliga" -> "1_bundesliga")
def sanitize_key(name):
    return str(name).lower().strip().replace(' ', '_')

# 2. Build a "smart map" where keys are normalized
# This maps "1_bundesliga" -> model_object AND "1 bundesliga" -> model_object
smart_model_map = {}
if isinstance(models_map, dict):
    for k, v in models_map.items():
        smart_model_map[sanitize_key(k)] = v
        smart_model_map[k] = v # keep original keys too

# 3. Lookup using the sanitized version of the active league
model = None
calibrator = None
target_key = sanitize_key(active_league)

# Try finding the model using the sanitized key
if target_key in smart_model_map:
    model = smart_model_map[target_key]
    # Try finding the matching calibrator
    if isinstance(calibrators_map, dict):
        smart_calib_map = {sanitize_key(k): v for k, v in calibrators_map.items()}
        if target_key in smart_calib_map:
            calibrator = smart_calib_map[target_key]

# 4. Fallback: If still no model, try the file_key directly (if it exists)
if model is None and file_key:
    sanitized_file_key = sanitize_key(file_key)
    if sanitized_file_key in smart_model_map:
        model = smart_model_map[sanitized_file_key]

# 5. Final Check
if model is None:
    st.error(f"❌ No trained model found for league '{active_league}'.")
    st.write(f"🔍 Looked for key: `{target_key}`")
    st.write(f"📂 Available keys in models_map: {list(smart_model_map.keys())}")
    st.stop()


# Determine expected feature order from the loaded XGBoost model (fallback to known list)
expected_features = None
try:
    booster = model.get_booster()
    expected_features = list(booster.feature_names) if booster.feature_names is not None else None
except Exception:
    expected_features = None
if expected_features is None:
    expected_features = ['start_x', 'start_y', 'distance', 'visible_angle', 'body_part_code', 'technique_code',
                         'angle_sin', 'angle_cos', 'dist_to_goal_center', 'is_header', 'start_x_norm', 'start_y_norm', 'player_last5_conv']

# Compute per-league model_xg for f_shots using the selected league model
if not f_shots.empty:
    # Ensure base categorical/angle/distance columns exist
    for col in ['body_part_code', 'technique_code', 'visible_angle', 'distance', 'start_x', 'start_y']:
        if col not in f_shots.columns:
            f_shots[col] = 0
    # Add engineered features used during training
    try:
        f_shots = _add_engineered_features(f_shots)
    except Exception:
        pass

    # Create aligned input matrix matching the model's feature names (fill missing cols with 0)
    X_local = pd.DataFrame(index=f_shots.index)
    for feat in expected_features:
        if feat in f_shots.columns:
            X_local[feat] = f_shots[feat]
        else:
            X_local[feat] = 0
    X_local = X_local.fillna(0)
    try:
        if calibrator is not None:
            f_shots['model_xg'] = calibrator.predict_proba(X_local)[:, 1]
        else:
            f_shots['model_xg'] = model.predict_proba(X_local)[:, 1]
    except Exception:
        f_shots['model_xg'] = 0.0

# Show precomputed suggestions (counterfactuals) for this league if present
try:
    suggestions_path = os.path.join(ROOT_DIR, 'models', f'suggestions_{file_key}.csv')
    if os.path.exists(suggestions_path):
        with st.expander('Counterfactual Suggestions (precomputed)', expanded=False):
            try:
                s_df = pd.read_csv(suggestions_path)
                st.write(f"Loaded suggestions: {os.path.relpath(suggestions_path, ROOT_DIR)}")
                st.dataframe(s_df.head(10), use_container_width=True)
                csv_data = s_df.to_csv(index=False)
                st.download_button('Download suggestions CSV', data=csv_data, file_name=os.path.basename(suggestions_path))
            except Exception as e:
                st.write('Could not load suggestions file:', e)
except Exception:
    pass

tab_ctx = TabContext(
    f_shots=f_shots,
    normalized_stats=normalized_stats,
    normalized_shots=normalized_shots,
    shots_df=shots_df,
    league_file_map=league_file_map,
    active_league=active_league,
    file_key=file_key,
    model=model,
    calibrator=calibrator,
)

# --- 6. MAIN TABS ---
t1, t2, t3, t4 = st.tabs([
    "🎯 SIMULATION", 
    "🧬 BEST XI", 
     "🧠 MODEL CONFIDENCE RADAR", 
    "🥅 SNIPER MAP"
])

render_simulation_tab(t1, tab_ctx)
render_squad_genome_tab(t2, tab_ctx, FORMATIONS)

# --- TAB 3: MODEL CONFIDENCE RADAR ---
with t3:
    st.markdown("### 🧠 MODEL CONFIDENCE RADAR")
    st.caption("Compare actual goals vs the model's predicted xG to highlight where the trainer is most confident (or surprised).")

    if 'model_xg' not in f_shots.columns:
        st.info('Model confidence data is missing. Run the league model or rerun preprocessing to restore model_xg values.')
    else:
        player_metrics = (
            f_shots.groupby('player_name')
            .agg(actual_goals=('is_goal', 'sum'), expected_goals=('model_xg', 'sum'), attempts=('start_x', 'count'))
            .reset_index()
        )
        player_metrics = player_metrics[player_metrics['attempts'] >= 5]
        if player_metrics.empty:
            st.info('Need more shot data to build the radar. Try a different league or rerun preprocessing.')
        else:
            player_metrics['confidence_gap'] = (player_metrics['actual_goals'] - player_metrics['expected_goals']).abs()
            highlight = player_metrics.sort_values('confidence_gap', ascending=False).head(6)
            if highlight.empty:
                st.info('No standout players yet. Check back after running more matches.')
            else:
                focus_options = ['Top 6 Players'] + highlight['player_name'].tolist()
                col_ctrl, col_viz = st.columns([1, 2])
                with col_ctrl:
                    focus_player = st.selectbox('Focus Player', focus_options)
                    st.caption('Changes here will update the radar on the right. Use Top 6 to compare the leaders or pick a single player to zoom in.')
                with col_viz:
                    if focus_player == 'Top 6 Players':
                        categories = highlight['player_name'].tolist()
                        actual = highlight['actual_goals'].tolist()
                        expected = highlight['expected_goals'].tolist()
                        max_radius = max(max(actual), max(expected), 1)
                        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                        angles += angles[:1]
                        actual_plot = actual + actual[:1]
                        expected_plot = expected + expected[:1]

                        fig, ax = plt.subplots(figsize=(4.8, 4), subplot_kw={'polar': True})
                        ax.set_theta_offset(np.pi / 2)
                        ax.set_theta_direction(-1)
                        ax.set_ylim(0, max_radius * 1.1)
                        ax.plot(angles, actual_plot, linewidth=2, label='Actual Goals', color='#ff0055')
                        ax.fill(angles, actual_plot, color='#ff0055', alpha=0.25)
                        ax.plot(angles, expected_plot, linewidth=2, label='Expected xG', color='#00f3ff')
                        ax.fill(angles, expected_plot, color='#00f3ff', alpha=0.25)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories, fontsize=10)
                        ax.set_rlabel_position(0)
                        ax.yaxis.set_tick_params(labelsize=8)
                        ax.grid(color='#88888833')
                        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
                        fig.tight_layout(pad=1)
                        st.pyplot(fig, use_container_width=False)
                    else:
                        row = highlight[highlight['player_name'] == focus_player].iloc[0]
                        metrics = {
                            'Actual Goals': row['actual_goals'],
                            'Expected xG': row['expected_goals'],
                            'Confidence Gap': row['confidence_gap']
                        }
                        fig, ax = plt.subplots(figsize=(4.2, 3.5), subplot_kw={'polar': True})
                        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                        values = list(metrics.values())
                        values += values[:1]
                        angles = list(angles) + [angles[0]]
                        max_radius = max(values) if values else 1
                        ax.set_ylim(0, max_radius * 1.1)
                        ax.plot(angles, values, color='#10b981', linewidth=2)
                        ax.fill(angles, values, color='#10b981', alpha=0.35)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(list(metrics.keys()), fontsize=10)
                        ax.yaxis.set_tick_params(labelsize=8)
                        ax.set_title(f"{focus_player} vs Model", pad=20)
                        fig.tight_layout(pad=1)
                        st.pyplot(fig, use_container_width=False)

                summary = highlight[['player_name', 'attempts', 'actual_goals', 'expected_goals', 'confidence_gap']]
                summary = summary.rename(columns={
                    'player_name': 'Player',
                    'attempts': 'Shots',
                    'actual_goals': 'Actual Goals',
                    'expected_goals': 'Predicted xG',
                    'confidence_gap': 'Gap'
                })
                st.dataframe(summary.reset_index(drop=True), use_container_width=True)

# --- TAB 4: SNIPER MAP ---
with t4:
    st.markdown("### 🥅 SNIPER MAP")
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        # Use per-league files for league list
        leagues = sorted(list(league_file_map.keys()))
        sel_league = st.selectbox("LEAGUE", ["All"] + leagues)
        include_all_comp = st.checkbox("Include player data from all competitions (ignore league filter)", value=False)
    
    # Build a working dataframe depending on whether we should respect the league filter
    # Default to the active per-league shots (selected at top of app)
    df_filtered = f_shots.copy()
    if sel_league != "All" and not include_all_comp:
        # If user picked a different league than the app's active league, load that league's shots
        if sel_league != active_league:
            lk = league_file_map.get(sel_league)
            # NOTE: Casing changed to 'Data'
            p = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"shots_{lk}.csv")
            if os.path.exists(p):
                try:
                    df_filtered = pd.read_csv(p)
                except Exception:
                    df_filtered = f_shots.copy()
        else:
            df_filtered = df_filtered[df_filtered['league'] == sel_league]

    with f2:
        # For the team selector, derive the team list from the LEAGUE the user selected.
        # Prefer the per-league file for `sel_league` (if available). If not available,
        # fall back to filtering the global `shots_df` by league. `include_all_comp` only
        # affects whether the final shot filtering ignores the league selection.
        team_col_candidates = ['team_name', 'team', 'squad']
        team_col = None
        teams = []

        try:
            if sel_league != "All":
                # try to load per-league CSV for the selected league
                lk = league_file_map.get(sel_league)
                if lk:
                    # NOTE: Casing changed to 'Data'
                    per_league_path = os.path.join(ROOT_DIR, 'Data', 'processed', 'leagues', f"shots_{lk}.csv")
                    if os.path.exists(per_league_path):
                        tmp = pd.read_csv(per_league_path)
                        team_col = next((c for c in team_col_candidates if c in tmp.columns), None)
                        if team_col:
                            teams = sorted(tmp[team_col].dropna().astype(str).unique())
                # fallback: use global shots_df filtered by league
                if not teams and isinstance(shots_df, pd.DataFrame):
                    # find team col in shots_df
                    team_col = next((c for c in team_col_candidates if c in shots_df.columns), None)
                    if team_col and 'league' in shots_df.columns:
                        teams = sorted(shots_df[shots_df['league'].astype(str).str.lower() == sel_league.lower()][team_col].dropna().astype(str).unique())
            else:
                # sel_league == All: if include_all_comp, list all teams from global shots_df
                if include_all_comp and isinstance(shots_df, pd.DataFrame):
                    team_col = next((c for c in team_col_candidates if c in shots_df.columns), None)
                    if team_col:
                        teams = sorted(shots_df[team_col].dropna().astype(str).unique())
                else:
                    # use current filtered dataframe (which represents the active league)
                    team_col = next((c for c in team_col_candidates if c in df_filtered.columns), None)
                    if team_col:
                        teams = sorted(df_filtered[team_col].dropna().astype(str).unique())
        except Exception:
            team_col = None
            teams = []

        if teams:
            sel_team = st.selectbox("TEAM", ["All"] + teams)
        else:
            st.markdown("*(Team data unavailable)*"); sel_team = "All"

        # Apply team filtering to the working dataframe (respecting include_all_comp)
        if sel_team != "All":
            try:
                # If include_all_comp is true, filter using the global shots_df mapping
                if include_all_comp and isinstance(shots_df, pd.DataFrame):
                    mapping = shots_df[['player_name', team_col]].drop_duplicates().set_index('player_name')[team_col].to_dict() if team_col and team_col in shots_df.columns else {}
                    df_filtered = df_filtered[df_filtered['player_name'].map(mapping) == sel_team]
                else:
                    # Prefer filtering directly on df_filtered if it contains the team column
                    if team_col and team_col in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered[team_col] == sel_team]
                    else:
                        # Last resort: use mapping from global shots_df
                        mapping = shots_df[['player_name', team_col]].drop_duplicates().set_index('player_name')[team_col].to_dict() if team_col and isinstance(shots_df, pd.DataFrame) and team_col in shots_df.columns else {}
                        df_filtered = df_filtered[df_filtered['player_name'].map(mapping) == sel_team]
            except Exception:
                pass

    with f3:
        # Player list: allow showing all players across competitions if requested
        players_source = f_shots if include_all_comp else df_filtered
        # Build a short display name (first two tokens) so long canonical names
        # like 'Cristiano Ronaldo dos Santos Aveiro' become 'Cristiano Ronaldo'
        try:
            ps = players_source.copy()
            ps['player_display'] = ps['player_name'].astype(str).apply(lambda n: ' '.join(str(n).split()[:2]).strip())
            players = sorted(ps['player_display'].dropna().unique())
        except Exception:
            players = sorted(players_source['player_name'].dropna().unique())
        idx = players.index("Lionel Messi") if "Lionel Messi" in players else 0
        sel_player = st.selectbox("PLAYER", players, index=idx if players else 0)

    with f4:
        filter_mode = st.radio("FILTER TYPE", ["Season", "Calendar Year"], horizontal=True, label_visibility="collapsed")
        # When including all competitions, start from the full shots_df, otherwise start from df_filtered
        p_source = f_shots if include_all_comp else df_filtered
        # Use the same short display mapping when filtering shots for the selected player
        try:
            p_src = p_source.copy()
            p_src['player_display'] = p_src['player_name'].astype(str).apply(lambda n: ' '.join(str(n).split()[:2]).strip())
        except Exception:
            p_src = p_source.copy()
            p_src['player_display'] = p_src['player_name'].astype(str)

        season_data = _ensure_year_column(p_src, 'season_year')
        year_candidates = sorted({int(y) for y in season_data['season_year'].dropna().astype(int) if y > 0}, reverse=True)
        year_labels = ["All Time"] + [str(y) for y in year_candidates]
        column_label = "YEAR" if filter_mode == "Calendar Year" else "SEASON YEAR"
        sel_label = st.selectbox(column_label, year_labels, index=0)
        if sel_label == "All Time":
            p_shots = season_data[season_data['player_display'] == sel_player]
        else:
            sel_year = int(sel_label)
            p_shots = season_data[(season_data['player_display'] == sel_player) & (season_data['season_year'] == sel_year)]

    if not p_shots.empty:
        goals = p_shots[p_shots['is_goal'] == 1]
        c1, c2, c3 = st.columns(3)
        c1.metric("TOTAL ATTEMPTS", len(p_shots))
        c2.metric("GOALS SCORED", len(goals))
        c3.metric("CONVERSION", f"{(len(goals)/len(p_shots)*100):.1f}%")

        fig, ax = draw_cyber_pitch()
        for _, row in goals.iterrows():
            con = ConnectionPatch(xyA=(row['start_x'], row['start_y']), xyB=(120, 40), 
                                  coordsA="data", coordsB="data", axesA=ax, axesB=ax, 
                                  color='#00f3ff', alpha=0.6, lw=2)
            ax.add_artist(con)
            
        ax.scatter(goals.start_x, goals.start_y, c='#00f3ff', edgecolors='white', s=150, marker='h', zorder=10, label='Goal')
        misses = p_shots[p_shots['is_goal'] == 0]
        ax.scatter(misses.start_x, misses.start_y, c='#f43f5e', alpha=0.3, s=50, zorder=5, label='Miss')
        ax.legend(facecolor='#050505', labelcolor='white')
        st.pyplot(fig, use_container_width=True)
    else:
        st.info(f"No shot data found for {sel_player} in this period.")