# utils/data_loader.py
import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st
# Ensure this import works relative to your structure
try:
    from utils.physics import calculate_visible_angle
except ImportError:
    # Fallback if running from a different root
    from physics import calculate_visible_angle

# Paths need to be calculated relative to the utils folder, or root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LEAGUES_DIR = os.path.join(ROOT_DIR, "Data", "processed", "leagues")
PROCESSED_DIRS = [
    Path(ROOT_DIR) / "Data" / "processed",
    Path("/mount/src/xgcoreai/Data/processed"),
]

def resolve_processed_file(name: str) -> str:
    """Return the first matching processed file from the supported directories."""
    for base in PROCESSED_DIRS:
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    # Return None instead of raising Error to allow soft-fail
    return None

@st.cache_resource
def load_resources():
    # 1. ATTEMPT TO LOAD GLOBAL SHOTS
    shots_path = resolve_processed_file("shots_final.csv")
    if shots_path:
        try:
            shots_df = pd.read_csv(shots_path)
        except Exception:
            shots_df = pd.DataFrame()
    else:
        # Initialize empty if missing - DO NOT STOP APP
        shots_df = pd.DataFrame()

    # 2. ATTEMPT TO LOAD GLOBAL STATS
    stats_path = resolve_processed_file("player_stats_final.csv")
    if stats_path:
        try:
            stats_df = pd.read_csv(stats_path)
        except Exception:
            stats_df = pd.DataFrame()
    else:
        stats_df = pd.DataFrame()

    # 3. LOAD MODELS (Always try to load models)
    models_map = {}
    calibrators_map = {}
    
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith('goal_predictor_') and fname.endswith('.json'):
                league_key = fname[len('goal_predictor_'):-len('.json')]
                path = os.path.join(MODELS_DIR, fname)
                try:
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    # store under both sanitized key and display name
                    display = league_key.replace('_', ' ')
                    models_map[league_key] = m
                    models_map[display] = m
                except Exception:
                    pass
        
        # Load calibrators
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith('goal_predictor_') and fname.endswith('_calibrator.joblib'):
                league_key = fname[len('goal_predictor_'):-len('_calibrator.joblib')]
                path = os.path.join(MODELS_DIR, fname)
                try:
                    import joblib
                    c = joblib.load(path)
                    display = league_key.replace('_', ' ')
                    calibrators_map[league_key] = c
                    calibrators_map[display] = c
                except Exception:
                    pass

        # Load legacy global model if present
        global_model_path = os.path.join(MODELS_DIR, 'goal_predictor.json')
        if os.path.exists(global_model_path):
            try:
                gm = xgb.XGBClassifier()
                gm.load_model(global_model_path)
                models_map['Global Database'] = gm
            except Exception:
                pass

    # --- DEFENSIVE PROCESSING (Only if shots_df has data) ---
    if not shots_df.empty:
        # Body Part
        if 'shot_body_part' in shots_df.columns:
            try:
                shots_df['body_part_code'] = shots_df['shot_body_part'].astype('category').cat.codes
                shots_df['is_header'] = shots_df['shot_body_part'].astype(str).str.lower().str.contains('head').astype(int)
            except Exception:
                shots_df['body_part_code'] = 0
                shots_df['is_header'] = 0
        else:
            shots_df['body_part_code'] = 0
            shots_df['is_header'] = 0

        # Technique
        if 'shot_technique' in shots_df.columns:
            try:
                shots_df['technique_code'] = shots_df['shot_technique'].astype('category').cat.codes
            except Exception:
                shots_df['technique_code'] = 0
        else:
            shots_df['technique_code'] = 0

        # Visible Angle
        if 'visible_angle' not in shots_df.columns:
            if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)
            else:
                shots_df['visible_angle'] = 0.0

        # Extract Year
        if 'date' in shots_df.columns:
            shots_df['extracted_year'] = pd.to_datetime(shots_df['date'], errors='coerce').dt.year
        elif 'match_date' in shots_df.columns:
            shots_df['extracted_year'] = pd.to_datetime(shots_df['match_date'], errors='coerce').dt.year
        elif 'season' in shots_df.columns:
            shots_df['extracted_year'] = pd.to_numeric(shots_df['season'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
        
        # Ensure League
        if 'league' not in shots_df.columns:
            if 'competition_name' in shots_df.columns:
                shots_df['league'] = shots_df['competition_name']
            else:
                shots_df['league'] = ''
        
        shots_df['model_xg'] = 0.0

        # --- MERGE TEAM DATA ---
        if not stats_df.empty and 'team_name' in stats_df.columns and 'team_name' not in shots_df.columns:
            try:
                # Logic to merge team names...
                has_season = 'season' in stats_df.columns
                if has_season and 'season' in shots_df.columns and 'league' in shots_df.columns:
                    tm = stats_df[['player_name', 'league', 'season', 'team_name']].drop_duplicates()
                    shots_df = shots_df.merge(tm, on=['player_name', 'league', 'season'], how='left')
                elif 'league' in shots_df.columns:
                    tm = stats_df[['player_name', 'league', 'team_name']].drop_duplicates()
                    shots_df = shots_df.merge(tm, on=['player_name', 'league'], how='left', suffixes=('', '_stats'))
                else:
                    tm = stats_df[['player_name', 'team_name']].drop_duplicates()
                    shots_df = shots_df.merge(tm, on='player_name', how='left', suffixes=('', '_stats'))
            except Exception:
                pass

    return shots_df, stats_df, models_map, calibrators_map