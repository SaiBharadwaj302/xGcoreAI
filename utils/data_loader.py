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

# --- CUSTOM WRAPPER CLASS (UPDATED) ---
class SafeModel:
    def __init__(self, booster):
        self._booster = booster
        self.classes_ = np.array([0, 1])
        # Cache the expected feature names from the trained model
        try:
            self.feature_names = booster.feature_names
        except:
            self.feature_names = None

    def predict_proba(self, X):
        data = X
        
        # --- CRITICAL FIX: FEATURE ALIGNMENT ---
        # The simulation tab might send columns in a different order than training.
        # We must force the input 'data' to match the booster's expected features exactly.
        if self.feature_names is not None and isinstance(data, pd.DataFrame):
            data = data.copy() # Don't modify the original
            
            # 1. Add any missing columns (fill with 0)
            for feat in self.feature_names:
                if feat not in data.columns:
                    data[feat] = 0
            
            # 2. Reorder columns to match the trained model's order EXACTLY
            data = data[self.feature_names]

        # --- PREDICTION ---
        dmat = xgb.DMatrix(data, enable_categorical=True)
        preds = self._booster.predict(dmat)
        return np.column_stack((1 - preds, preds))

    def get_booster(self):
        return self._booster

@st.cache_resource
def load_resources():
    # --- 1. LOAD SHOTS ---
    shots_path = resolve_processed_file("shots_final.csv")
    shots_df = pd.read_csv(shots_path) if shots_path else pd.DataFrame()

    # --- 2. LOAD STATS ---
    stats_path = resolve_processed_file("player_stats_final.csv")
    stats_df = pd.read_csv(stats_path) if stats_path else pd.DataFrame()

    # --- 3. LOAD MODELS (Robust Mode) ---
    models_map = {}
    calibrators_map = {}
    
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith('goal_predictor_') and fname.endswith('.json'):
                league_key = fname[len('goal_predictor_'):-len('.json')]
                path = os.path.join(MODELS_DIR, fname)
                
                model_loaded = None
                try:
                    # Attempt 1: Standard Load
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    if not hasattr(m, '_estimator_type'):
                        raise ValueError("Missing metadata")
                    model_loaded = m
                except Exception:
                    # Attempt 2: Load as Raw Booster & Wrap
                    try:
                        booster = xgb.Booster()
                        booster.load_model(path)
                        # Wrap it in our smarter SafeModel class
                        model_loaded = SafeModel(booster)
                    except Exception as e:
                        st.error(f"❌ Failed to load `{fname}`: {e}")

                if model_loaded:
                    models_map[league_key] = model_loaded
                    models_map[league_key.replace('_', ' ')] = model_loaded
        
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
        for col in ['body_part_code', 'technique_code', 'is_header', 'visible_angle', 'model_xg']:
            if col not in shots_df.columns: shots_df[col] = 0.0
        
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', '')

        if 'visible_angle' in shots_df.columns and shots_df['visible_angle'].sum() == 0:
             if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                 shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)

    return shots_df, stats_df, models_map, calibrators_map