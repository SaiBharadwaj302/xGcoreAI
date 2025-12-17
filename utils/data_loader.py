import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
import streamlit as st

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PROCESSED_DIRS = [Path(ROOT_DIR) / "Data" / "processed", Path(ROOT_DIR) / "data" / "processed"]

def resolve_processed_file(name: str):
    for base in PROCESSED_DIRS:
        cand = base / name
        if cand.exists(): return str(cand)
    return None

class SafeModel:
    """
    Wrapper for XGBoost models to ensure safe inference.
    Assumes X has already been engineered via utils.helpers._add_engineered_features
    """
    def __init__(self, booster):
        self._booster = booster
        self.feature_names = [
            'start_x', 'start_y', 'distance', 'visible_angle', 
            'body_part_code', 'technique_code', 'angle_sin', 'angle_cos', 
            'dist_to_goal_center', 'is_header', 'start_x_norm', 
            'start_y_norm', 'player_last5_conv'
        ]

    def predict(self, X):
        # 1. Ensure DataFrame
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # 2. Fill Missing Columns with 0 (Safe Fallback)
        for feat in self.feature_names:
            if feat not in data.columns:
                data[feat] = 0.0
        
        # 3. Order Columns strictly
        data = data[self.feature_names].astype(float)

        # 4. Predict
        dmat = xgb.DMatrix(data) 
        return self._booster.predict(dmat)

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
                key = fname.replace('goal_predictor_', '').replace('.json', '')
                try:
                    booster = xgb.Booster()
                    booster.load_model(os.path.join(MODELS_DIR, fname))
                    safe_model = SafeModel(booster)
                    models_map[key] = safe_model
                    models_map[key.replace('_', ' ')] = safe_model
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

    return shots_df, stats_df, models_map