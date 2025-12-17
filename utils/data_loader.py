# utils/data_loader.py
import os
import pandas as pd
import xgboost as xgb
import streamlit as st
from utils.physics import calculate_visible_angle

# Paths need to be calculated relative to the utils folder, or root
# This moves up one level from 'utils' to root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SHOTS = os.path.join(ROOT_DIR, "data", "processed", "shots_final.csv")
DATA_STATS = os.path.join(ROOT_DIR, "data", "processed", "player_stats_final.csv")
DATA_DNA = os.path.join(ROOT_DIR, "data", "processed", "player_dna_clustered.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LEAGUES_DIR = os.path.join(ROOT_DIR, "data", "processed", "leagues")

@st.cache_resource
def load_resources():
    missing_files = []
    if not os.path.exists(DATA_SHOTS): missing_files.append(f"Shots Data ({DATA_SHOTS})")
    if not os.path.exists(DATA_STATS): missing_files.append(f"Stats Data ({DATA_STATS})")
    if missing_files:
        for f in missing_files: st.error(f"❌ Missing File: {f}")
        return None, None, None, None
    try:
        shots_df = pd.read_csv(DATA_SHOTS)
        stats_df = pd.read_csv(DATA_STATS)
        # `player_dna_clustered.csv` (player DNA) has been removed from this build.
        # We no longer load or return a `dna_df` to keep the codebase free of clustering artifacts.
        # Load models: global model + per-league models
        models_map = {}
        calibrators_map = {}
        # Load per-league models and create friendly display keys.
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
            # try to load calibrators saved as joblib alongside models
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
        # Optionally load a global model if present (deprecated)
        global_model_path = os.path.join(MODELS_DIR, 'goal_predictor.json')
        if os.path.exists(global_model_path):
            try:
                gm = xgb.XGBClassifier()
                gm.load_model(global_model_path)
                models_map['Global Database'] = gm
            except Exception:
                pass
        # Optionally load a global calibrator
        global_cal_path = os.path.join(MODELS_DIR, 'goal_predictor_calibrator.joblib')
        if os.path.exists(global_cal_path):
            try:
                import joblib
                gc = joblib.load(global_cal_path)
                calibrators_map['Global Database'] = gc
            except Exception:
                pass
        # Defensive handling for body part / technique columns which may not exist
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

        if 'shot_technique' in shots_df.columns:
            try:
                shots_df['technique_code'] = shots_df['shot_technique'].astype('category').cat.codes
            except Exception:
                shots_df['technique_code'] = 0
        else:
            shots_df['technique_code'] = 0

        # Compute visible angle only if start coordinates exist
        if 'visible_angle' not in shots_df.columns:
            if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)
            else:
                shots_df['visible_angle'] = 0.0

        # --- EXTRACTOR LOGIC FOR YEAR/SEASON ---
        if 'date' in shots_df.columns:
            shots_df['extracted_year'] = pd.to_datetime(shots_df['date'], errors='coerce').dt.year
        elif 'match_date' in shots_df.columns:
            shots_df['extracted_year'] = pd.to_datetime(shots_df['match_date'], errors='coerce').dt.year
        elif 'season' in shots_df.columns:
            shots_df['extracted_year'] = pd.to_numeric(shots_df['season'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
        # Ensure a `league` column exists for downstream code that expects it.
        # Our ETL produces `competition_name`; create `league` as a copy when missing.
        if 'league' not in shots_df.columns:
            if 'competition_name' in shots_df.columns:
                shots_df['league'] = shots_df['competition_name']
            else:
                shots_df['league'] = ''
        
        # Do NOT compute a global model_xg: model predictions will be performed per-league
        # by the app when a league is selected. Fill column with zeros as placeholder.
        shots_df['model_xg'] = 0.0
        
        # --- MERGE TEAM DATA FROM PLAYER_STATS_FINAL ---
        # Add team_name to shots_df by merging from stats_df
        if 'team_name' in stats_df.columns and 'team_name' not in shots_df.columns:
            # Check if stats_df has meaningful season data (not just "All Time")
            has_season_data = False
            if 'season' in stats_df.columns:
                unique_seasons = stats_df['season'].unique()
                has_season_data = len(unique_seasons) > 1 or (len(unique_seasons) == 1 and unique_seasons[0] != 'All Time')
            
            # Try merging with season if both dataframes have it and stats has real season data
            if has_season_data and 'season' in shots_df.columns and 'league' in shots_df.columns:
                team_mapping = stats_df[['player_name', 'league', 'season', 'team_name']].drop_duplicates()
                shots_df = shots_df.merge(
                    team_mapping, 
                    on=['player_name', 'league', 'season'], 
                    how='left'
                )
            # Otherwise merge by player_name and league only
            elif 'league' in shots_df.columns:
                team_mapping_simple = stats_df[['player_name', 'league', 'team_name']].drop_duplicates()
                shots_df = shots_df.merge(
                    team_mapping_simple,
                    on=['player_name', 'league'],
                    how='left',
                    suffixes=('', '_stats')
                )
            else:
                # Last resort: just use player_name
                team_mapping_basic = stats_df[['player_name', 'team_name']].drop_duplicates()
                shots_df = shots_df.merge(
                    team_mapping_basic,
                    on='player_name',
                    how='left',
                    suffixes=('', '_stats')
                )
        
        return shots_df, stats_df, models_map, calibrators_map
    except Exception as e:
        st.error(f"❌ Error loading: {str(e)}"); return None, None, None, None, None