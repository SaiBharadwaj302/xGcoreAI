import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np
import json
import re

# Safe import for physics
try:
    from utils.physics import calculate_visible_angle
except ImportError:
    # Minimal fallback
    def calculate_visible_angle(x, y):
        if x >= 120: return 0.0
        dx = 120 - x
        dy1 = 36 - y
        dy2 = 44 - y
        a1 = np.arctan2(abs(dy1), dx)
        a2 = np.arctan2(abs(dy2), dx)
        if 36 < y < 44: return a1 + a2
        return abs(a1 - a2)

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

# --- CUSTOM WRAPPER CLASS ---
class SafeModel:
    def __init__(self, booster, model_path=None):
        self._booster = booster
        self.classes_ = np.array([0, 1])
        
        # --- 🚨 YOUR HARDCODED LIST 🚨 ---
        self.feature_names = [
            'start_x', 'start_y', 'distance', 'visible_angle', 
            'body_part_code', 'technique_code', 'angle_sin', 'angle_cos', 
            'dist_to_goal_center', 'is_header', 'start_x_norm', 
            'start_y_norm', 'player_last5_conv'
        ]

    def predict_proba(self, X):
        # 1. Ensure DataFrame
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # 2. STANDARDIZE INPUTS (Handle 'x' vs 'start_x')
        if 'x' in data.columns and 'start_x' not in data.columns:
            data['start_x'] = data['x']
        if 'y' in data.columns and 'start_y' not in data.columns:
            data['start_y'] = data['y']

        # 3. FEATURE ENGINEERING (Calculate Physics)
        if 'start_x' in data.columns and 'start_y' in data.columns:
            # Distance / Angle
            if 'distance' not in data.columns:
                data['distance'] = np.sqrt((120 - data['start_x'])**2 + (40 - data['start_y'])**2)
            if 'visible_angle' not in data.columns:
                data['visible_angle'] = data.apply(
                    lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1
                )
            if 'angle_sin' not in data.columns:
                data['angle_sin'] = np.sin(data['visible_angle'])
            if 'angle_cos' not in data.columns:
                data['angle_cos'] = np.cos(data['visible_angle'])
            if 'dist_to_goal_center' not in data.columns:
                data['dist_to_goal_center'] = np.abs(data['start_y'] - 40)
            
            # Normalization
            if 'start_x_norm' not in data.columns:
                data['start_x_norm'] = data['start_x'] / 120.0
            if 'start_y_norm' not in data.columns:
                data['start_y_norm'] = data['start_y'] / 80.0

        # 4. ALIGNMENT & TYPE FORCING
        # Force everything to be simple floats. No 'category' types allowed.
        for feat in self.feature_names:
            if feat not in data.columns:
                data[feat] = 0.0
        
        # Force Exact Order
        data = data[self.feature_names]
        
        # Force Float Type (Fixes XGBoost 2.0+ Categorical Crash)
        data = data.astype(float)

        # 5. DEBUGGING (Look at your sidebar!)
        # This will show you exactly what the model is seeing.
        # If 'distance' is 0 here, we know the calc failed.
        if len(data) == 1:
            st.sidebar.markdown("### 🛠️ Debug: Live Model Input")
            st.sidebar.json(data.iloc[0].to_dict())

        # 6. PREDICT
        # REMOVED 'enable_categorical=True' -> This was likely the cause of the 0% error
        dmat = xgb.DMatrix(data) 
        preds = self._booster.predict(dmat)
        return np.column_stack((1 - preds, preds))

    def get_booster(self):
        return self._booster

# Changed function name to 'load_resources_v4' to FORCE cache clear
@st.cache_resource
def load_resources_v4():
    # --- 1. LOAD SHOTS ---
    shots_path = resolve_processed_file("shots_final.csv")
    shots_df = pd.read_csv(shots_path) if shots_path else pd.DataFrame()

    # --- 2. LOAD STATS ---
    stats_path = resolve_processed_file("player_stats_final.csv")
    stats_df = pd.read_csv(stats_path) if stats_path else pd.DataFrame()

    # --- 3. LOAD MODELS ---
    models_map = {}
    calibrators_map = {}
    
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith('goal_predictor_') and fname.endswith('.json'):
                league_key = fname[len('goal_predictor_'):-len('.json')]
                path = os.path.join(MODELS_DIR, fname)
                
                model_loaded = None
                try:
                    # Try standard load
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    if not hasattr(m, '_estimator_type'):
                         # Force fallback if metadata missing
                        raise ValueError("Missing metadata")
                    model_loaded = m
                except Exception:
                    # Fallback to SafeModel
                    try:
                        booster = xgb.Booster()
                        booster.load_model(path)
                        model_loaded = SafeModel(booster, model_path=path)
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

    # --- 4. CLEANUP ---
    if not shots_df.empty:
        for col in ['body_part_code', 'technique_code', 'is_header', 'visible_angle', 'model_xg']:
            if col not in shots_df.columns: shots_df[col] = 0.0
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', '')
        if 'visible_angle' in shots_df.columns and shots_df['visible_angle'].sum() == 0:
             if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                 shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)

    return shots_df, stats_df, models_map, calibrators_map

# Map function so app.py finds it
load_resources = load_resources_v4