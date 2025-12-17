import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st

# Safe import for physics
try:
    from utils.physics import calculate_visible_angle
except ImportError:
    from physics import calculate_visible_angle

# 1. Define Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Handle Case Sensitivity for Data/processed vs data/processed
PROCESSED_DIRS = [
    Path(ROOT_DIR) / "Data" / "processed",  # Capital D (GitHub Structure)
    Path(ROOT_DIR) / "data" / "processed",  # Lowercase d (Fallback)
]

def resolve_processed_file(name: str) -> str:
    """Return the first matching processed file from the supported directories."""
    for base in PROCESSED_DIRS:
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return None

@st.cache_resource
def load_resources():
    # --- 1. LOAD SHOTS ---
    shots_path = resolve_processed_file("shots_final.csv")
    if shots_path:
        try:
            shots_df = pd.read_csv(shots_path)
        except Exception:
            shots_df = pd.DataFrame()
    else:
        shots_df = pd.DataFrame()

    # --- 2. LOAD STATS ---
    stats_path = resolve_processed_file("player_stats_final.csv")
    if stats_path:
        try:
            stats_df = pd.read_csv(stats_path)
        except Exception:
            stats_df = pd.DataFrame()
    else:
        stats_df = pd.DataFrame()

    # --- 3. LOAD MODELS (VERBOSE DEBUGGING) ---
    models_map = {}
    calibrators_map = {}
    
    if not os.path.exists(MODELS_DIR):
        st.error(f"🚨 CRITICAL: The models folder was not found at: `{MODELS_DIR}`")
        st.write(f"Contents of Root `{ROOT_DIR}`: {os.listdir(ROOT_DIR)}")
    else:
        # Check if folder is empty
        files = os.listdir(MODELS_DIR)
        if not files:
            st.warning(f"⚠️ The models folder at `{MODELS_DIR}` is empty.")
        
        for fname in files:
            # Load JSON Models
            if fname.startswith('goal_predictor_') and fname.endswith('.json'):
                league_key = fname[len('goal_predictor_'):-len('.json')]
                path = os.path.join(MODELS_DIR, fname)
                try:
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    
                    # Store keys
                    display = league_key.replace('_', ' ')
                    models_map[league_key] = m
                    models_map[display] = m
                except Exception as e:
                    # <<< THIS IS THE FIX: PRINT THE ERROR >>>
                    st.error(f"❌ Failed to load model `{fname}`: {e}")
        
            # Load Calibrators
            elif fname.endswith('_calibrator.joblib'):
                league_key = fname[len('goal_predictor_'):-len('_calibrator.joblib')]
                path = os.path.join(MODELS_DIR, fname)
                try:
                    import joblib
                    c = joblib.load(path)
                    display = league_key.replace('_', ' ')
                    calibrators_map[league_key] = c
                    calibrators_map[display] = c
                except Exception as e:
                    st.warning(f"⚠️ Could not load calibrator `{fname}`: {e}")

    # --- 4. DATA PROCESSING (Feature Engineering) ---
    if not shots_df.empty:
        # (Keep your existing feature engineering logic minimal here to save space)
        # Ensure required columns exist
        for col in ['body_part_code', 'technique_code', 'is_header', 'visible_angle', 'model_xg']:
            if col not in shots_df.columns:
                shots_df[col] = 0.0
                
        # Basic League/Season extractions
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', '')

    return shots_df, stats_df, models_map, calibrators_map