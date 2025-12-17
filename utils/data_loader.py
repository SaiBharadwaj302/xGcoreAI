import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np

# --- 1. Define Paths ---
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

# --- 2. Physics Helper (Simplified) ---
def calculate_visible_angle(x, y):
    """Calculates angle of goal visibility from pitch coordinates."""
    if x >= 120: return 0.0
    dx = 120 - x
    dy1 = 36 - y
    dy2 = 44 - y
    a1 = np.arctan2(abs(dy1), dx)
    a2 = np.arctan2(abs(dy2), dx)
    if 36 < y < 44: return a1 + a2
    return abs(a1 - a2)

# --- 3. Custom Model Wrapper (No Probability) ---
class SafeModel:
    def __init__(self, booster):
        self._booster = booster
        # Define features strictly
        self.feature_names = [
            'start_x', 'start_y', 'distance', 'visible_angle', 
            'body_part_code', 'technique_code', 'angle_sin', 'angle_cos', 
            'dist_to_goal_center', 'is_header', 'start_x_norm', 
            'start_y_norm', 'player_last5_conv'
        ]

    def predict(self, X):
        """Returns raw prediction score (0.0 to 1.0)"""
        # 1. Ensure DataFrame
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # 2. Standardize Coordinates
        if 'x' in data.columns and 'start_x' not in data.columns:
            data['start_x'] = data['x']
        if 'y' in data.columns and 'start_y' not in data.columns:
            data['start_y'] = data['y']

        # 3. Calculate Physics Features (On the fly)
        if 'start_x' in data.columns and 'start_y' in data.columns:
            data['distance'] = np.sqrt((120 - data['start_x'])**2 + (40 - data['start_y'])**2)
            data['visible_angle'] = data.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)
            data['angle_sin'] = np.sin(data['visible_angle'])
            data['angle_cos'] = np.cos(data['visible_angle'])
            data['dist_to_goal_center'] = np.abs(data['start_y'] - 40)
            data['start_x_norm'] = data['start_x'] / 120.0
            data['start_y_norm'] = data['start_y'] / 80.0

        # 4. Fill Missing Features with 0.0
        for feat in self.feature_names:
            if feat not in data.columns:
                data[feat] = 0.0
        
        # 5. Reorder & Type Force
        data = data[self.feature_names].astype(float)

        # 6. Predict (Return simple 1D array)
        dmat = xgb.DMatrix(data) 
        preds = self._booster.predict(dmat)
        return preds

# --- 4. Loader Function ---
@st.cache_resource
def load_resources():
    # Load Data
    shots_path = resolve_processed_file("shots_final.csv")
    shots_df = pd.read_csv(shots_path) if shots_path else pd.DataFrame()
    
    stats_path = resolve_processed_file("player_stats_final.csv")
    stats_df = pd.read_csv(stats_path) if stats_path else pd.DataFrame()

    # Load Models
    models_map = {}
    
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith('goal_predictor_') and fname.endswith('.json'):
                league_key = fname.replace('goal_predictor_', '').replace('.json', '')
                path = os.path.join(MODELS_DIR, fname)
                
                try:
                    # Always use SafeModel wrapper for consistency
                    booster = xgb.Booster()
                    booster.load_model(path)
                    safe_model = SafeModel(booster)
                    
                    models_map[league_key] = safe_model
                    models_map[league_key.replace('_', ' ')] = safe_model # Handle spaces
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

    # Basic Cleanup on Shots Data
    if not shots_df.empty:
        if 'league' not in shots_df.columns:
            shots_df['league'] = shots_df.get('competition_name', 'Unknown')

    return shots_df, stats_df, models_map