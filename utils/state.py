import os
import sys
import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any, Optional

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.data_loader import load_resources
from utils.helpers import _add_engineered_features

# FIX: Use cache_resource for Models (Unserializable objects)
@st.cache_resource(show_spinner=False)
def load_global_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads and caches global resources (models, stats, shots).
    Uses cache_resource because XGBoost models are complex objects.
    """
    try:
        shots, stats, models = load_resources()
        return shots, stats, models
    except Exception as e:
        print(f"Error loading global resources: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

# Keep cache_data for League Data (It only returns DataFrames, which are serializable)
@st.cache_data(show_spinner=False)
def load_league_data(root_dir: str, active_league: str, file_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses specific league data on demand.
    
    Args:
        root_dir: Absolute path to project root.
        active_league: Display name of the league.
        file_map: Dictionary mapping display names to file keys.
    """
    key = file_map.get(active_league)
    if not key:
        return pd.DataFrame(), pd.DataFrame()
    
    # Construct paths using pathlib or os.path for OS agnostic handling
    s_path = os.path.join(root_dir, 'Data', 'processed', 'leagues', f"shots_{key}.csv")
    st_path = os.path.join(root_dir, 'Data', 'processed', 'leagues', f"stats_{key}.csv")
    
    if not os.path.exists(s_path):
        return pd.DataFrame(), pd.DataFrame()
        
    try:
        l_shots = pd.read_csv(s_path)
        l_stats = pd.read_csv(st_path) if os.path.exists(st_path) else pd.DataFrame()
        
        # Apply Feature Engineering (Cached Operation)
        if not l_shots.empty:
            l_shots = _add_engineered_features(l_shots)
            
        return l_shots, l_stats
    except Exception as e:
        print(f"Error processing league data for {active_league}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def get_league_file_map(root_dir: str) -> Dict[str, str]:
    """Scans the processed directory to build a dynamic map of available leagues."""
    leagues_dir = os.path.join(root_dir, 'Data', 'processed', 'leagues')
    file_map = {} 
    
    if os.path.exists(leagues_dir):
        for fn in os.listdir(leagues_dir):
            if fn.startswith('shots_') and fn.endswith('.csv'):
                # format: shots_{key}.csv
                key = fn[6:-4] 
                display = key.replace('_', ' ')
                file_map[display] = key
    return file_map