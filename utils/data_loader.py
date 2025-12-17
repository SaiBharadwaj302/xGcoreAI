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
    # Minimal fallback if physics utils are missing
    def calculate_visible_angle(x, y):
        # StatsBomb coords: Goal at (120, 40), posts at 36 and 44
        if x >= 120: return 0.0
        dx = 120 - x
        dy1 = 36 - y
        dy2 = 44 - y
        a1 = np.arctan2(abs(dy1), dx)
        a2 = np.arctan2(abs(dy2), dx)
        # If y is between posts
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

# --- CUSTOM WRAPPER CLASS (SMART VERSION) ---
class SafeModel:
    def __init__(self, booster):
        self._booster = booster
        self.classes_ = np.array([0, 1])
        try:
            self.feature_names = booster.feature_names
        except:
            self.feature_names = None

    def predict_proba(self, X):
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # 1. Coordinate Normalization (StatsBomb assumed: 120x80)
        # Ensure we have start_x/start_y. Simulation might send 'x'/'y'.
        if 'x' in data.columns and 'start_x' not in data.columns:
            data['start_x'] = data['x']
        if 'y' in data.columns and 'start_y' not in data.columns:
            data['start_y'] = data['y']

        # 2. On-the-fly Feature Engineering
        # If the model needs features that are missing, calculate them now.
        if 'start_x' in data.columns and 'start_y' in data.columns:
            # Distance to goal center (120, 40)
            if 'distance' not in data.columns:
                data['distance'] = np.sqrt((120 - data['start_x'])**2 + (40 - data['start_y'])**2)
            
            # Visible Angle
            if 'visible_angle' not in data.columns:
                data['visible_angle'] = data.apply(
                    lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1
                )
            
            # Angle Sine/Cosine
            if 'angle_sin' not in data.columns:
                data['angle_sin'] = np.sin(data['visible_angle'])
            if 'angle_cos' not in data.columns:
                data['angle_cos'] = np.cos(data['visible_angle'])
                
            # Distance to center axis (y=40)
            if 'dist_to_goal_center' not in data.columns:
                data['dist_to_goal_center'] = np.abs(data['start_y'] - 40)

        # 3. Align Columns with Model
        if self.feature_names is not None:
            # Add any other missing columns as 0 (e.g. is_header, body_part)
            for feat in self.feature_names:
                if feat not in data.columns:
                    data[feat] = 0.0
            
            # CRITICAL: Reorder columns to match training order exactly
            data = data[self.feature_names]

        # 4. Predict
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
                    if not hasattr(m, '_estimator_type'):
                        raise ValueError("Missing metadata")
                    model_loaded = m
                except Exception:
                    try:
                        booster = xgb.Booster()
                        booster.load_model(path)
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

    # --- 4. GLOBAL DATAFRAME CLEANUP ---
    if not shots_df.empty:
        for col in ['body_part_code', 'technique_code', 'is_header', 'visible_angle', 'model_xg']:
            if col not in shots_df.columns: shots_df[col] = 0.0
        
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', '')

        if 'visible_angle' in shots_df.columns and shots_df['visible_angle'].sum() == 0:
             if 'start_x' in shots_df.columns and 'start_y' in shots_df.columns:
                 shots_df['visible_angle'] = shots_df.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)

    return shots_df, stats_df, models_map, calibrators_map