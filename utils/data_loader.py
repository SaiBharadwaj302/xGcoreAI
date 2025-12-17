import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np
import json

# Safe import for physics
try:
    from utils.physics import calculate_visible_angle
except ImportError:
    # Minimal fallback if physics utils are missing
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

def extract_feature_names_from_json(model_path):
    """
    Manually parses the XGBoost JSON file to find feature names
    when the Cloud version can't read metadata via API.
    """
    try:
        with open(model_path, 'r') as f:
            dump = json.load(f)
        
        # Check standard locations for feature names in XGBoost dumps
        if 'learner' in dump and 'feature_names' in dump['learner']:
            return dump['learner']['feature_names']
        if 'feature_names' in dump:
            return dump['feature_names']
    except Exception:
        pass
    return None

# --- CUSTOM WRAPPER CLASS (SMART VERSION) ---
class SafeModel:
    def __init__(self, booster, model_path=None):
        self._booster = booster
        self.classes_ = np.array([0, 1])
        self.feature_names = None

        # 1. Try getting features from the booster object
        try:
            self.feature_names = booster.feature_names
        except:
            pass

        # 2. If that failed, try parsing the JSON file directly
        if not self.feature_names and model_path:
            self.feature_names = extract_feature_names_from_json(model_path)

        # 3. Fallback: Use the standard list from your training code
        if not self.feature_names:
            self.feature_names = [
                'start_x', 'start_y', 'distance', 'visible_angle', 
                'body_part_code', 'technique_code', 'angle_sin', 'angle_cos', 
                'dist_to_goal_center', 'is_header', 'start_x_norm', 
                'start_y_norm', 'player_last5_conv'
            ]

    def predict_proba(self, X):
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # --- A. STANDARDIZE INPUTS ---
        # Simulation might send 'x'/'y' instead of 'start_x'/'start_y'
        if 'x' in data.columns and 'start_x' not in data.columns:
            data['start_x'] = data['x']
        if 'y' in data.columns and 'start_y' not in data.columns:
            data['start_y'] = data['y']

        # --- B. FEATURE ENGINEERING (Fill Missing Values) ---
        if 'start_x' in data.columns and 'start_y' in data.columns:
            # 1. Distance & Angles
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
            
            # 2. Normalization (CRITICAL FIX FOR 0% ISSUE)
            # This was missing before!
            if 'start_x_norm' not in data.columns:
                data['start_x_norm'] = data['start_x'] / 120.0
            if 'start_y_norm' not in data.columns:
                data['start_y_norm'] = data['start_y'] / 80.0

        # --- C. ALIGNMENT ---
        # Add any other missing columns as 0.0
        for feat in self.feature_names:
            if feat not in data.columns:
                data[feat] = 0.0
        
        # FORCE COLUMN ORDER
        data = data[self.feature_names]

        # --- D. PREDICT ---
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
                    m = xgb.XGBClassifier()
                    m.load_model(path)
                    # Check metadata
                    if not hasattr(m, '_estimator_type'):
                        raise ValueError("Missing metadata")
                    model_loaded = m
                except Exception:
                    try:
                        booster = xgb.Booster()
                        booster.load_model(path)
                        # Pass path so we can scan JSON for feature names
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

    # --- 4. GLOBAL CLEANUP ---
    if not shots_df.empty:
        for col in ['body_part_code', 'technique_code', 'is_header', 'visible_angle', 'model_xg']:
            if col not in shots_df.columns: shots_df[col] = 0.0
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', '')
        if 'visible_angle' in shots_df.columns and shots_df['visible_angle'].sum() == 0:
             if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                 shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)

    return shots_df, stats_df, models_map, calibrators_map