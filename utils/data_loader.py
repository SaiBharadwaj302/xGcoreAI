import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np

# Safe import for physics
try:
    from utils.physics import calculate_visible_angle
except ImportError:
    from physics import calculate_visible_angle

# 1. Define Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

PROCESSED_DIRS = [
    Path(ROOT_DIR) / "Data" / "processed",
    Path(ROOT_DIR) / "data" / "processed",
]

def resolve_processed_file(name: str) -> str:
    for base in PROCESSED_DIRS:
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return None

@st.cache_resource
def load_resources():
    # --- 1. LOAD SHOTS ---
    shots_path = resolve_processed_file("shots_final.csv")
    shots_df = pd.read_csv(shots_path) if shots_path else pd.DataFrame()

    # --- 2. LOAD STATS ---
    stats_path = resolve_processed_file("player_stats_final.csv")
    stats_df = pd.read_csv(stats_path) if stats_path else pd.DataFrame()

    # --- 3. LOAD MODELS (With Force-Load Logic) ---
    models_map = {}
    calibrators_map = {}
    
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            # Load JSON Models
            if fname.startswith('goal_predictor_') and fname.endswith('.json'):
                league_key = fname[len('goal_predictor_'):-len('.json')]
                path = os.path.join(MODELS_DIR, fname)
                try:
                    # Attempt 1: Standard Load
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    
                    # Fix for missing estimator_type in strict environments
                    if not hasattr(m, '_estimator_type'):
                        m._estimator_type = "classifier"
                    
                    models_map[league_key] = m
                    models_map[league_key.replace('_', ' ')] = m

                except Exception:
                    try:
                        # Attempt 2: "Force Load" (Load as raw booster, then wrap)
                        # This bypasses the sklearn metadata check causing your error
                        booster = xgb.Booster()
                        booster.load_model(path)
                        
                        m = xgb.XGBClassifier()
                        m._Booster = booster
                        m._estimator_type = "classifier"
                        m.classes_ = np.array([0, 1])  # Assume binary classification
                        
                        models_map[league_key] = m
                        models_map[league_key.replace('_', ' ')] = m
                    except Exception as e:
                        st.error(f"❌ Could not force-load model `{fname}`: {e}")
        
            # Load Calibrators
            elif fname.endswith('_calibrator.joblib'):
                league_key = fname[len('goal_predictor_'):-len('_calibrator.joblib')]
                path = os.path.join(MODELS_DIR, fname)
                try:
                    import joblib
                    c = joblib.load(path)
                    calibrators_map[league_key] = c
                    calibrators_map[league_key.replace('_', ' ')] = c
                except Exception:
                    pass

    # --- 4. DATA PROCESSING ---
    if not shots_df.empty:
        # Fill missing columns
        for col in ['body_part_code', 'technique_code', 'is_header', 'visible_angle', 'model_xg']:
            if col not in shots_df.columns: shots_df[col] = 0.0
        
        # Ensure League column
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', '')

        # Recalculate angles if missing (safety check)
        if 'visible_angle' in shots_df.columns and shots_df['visible_angle'].sum() == 0:
             if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                 shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)

    return shots_df, stats_df, models_map, calibrators_map