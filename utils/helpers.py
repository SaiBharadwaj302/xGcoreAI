import numpy as np
import pandas as pd
import re
from typing import Optional, Any

def sanitize(name: Any) -> str:
    """Sanitizes strings for file system safety."""
    return ''.join([c for c in str(name).strip().replace(' ', '_') if c.isalnum() or c in ['_','-']])

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Centralized feature engineering logic.
    Ensures consistency between training (offline) and inference (online).
    """
    df = df.copy()

    # --- 1. Chronological Sorting (Prevent Data Leakage) ---
    sort_cols = [c for c in ['season', 'match_date', 'date', 'match_id', 'minute'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=True)

    # --- 2. Geometric Features ---
    if 'visible_angle' not in df.columns:
        df['visible_angle'] = 0.0

    # Vectorized geometric calculations
    try:
        ang = pd.to_numeric(df['visible_angle'], errors='coerce').fillna(0.0)
        ang_rad = np.deg2rad(ang)
        df['angle_sin'] = np.sin(ang_rad)
        df['angle_cos'] = np.cos(ang_rad)
    except Exception:
        df['angle_sin'] = 0.0
        df['angle_cos'] = 0.0

    if 'start_x' in df.columns and 'start_y' in df.columns:
        sx = pd.to_numeric(df['start_x'], errors='coerce').fillna(0.0)
        sy = pd.to_numeric(df['start_y'], errors='coerce').fillna(0.0)
        # Distance to goal center (120, 40)
        df['dist_to_goal_center'] = np.sqrt((120 - sx) ** 2 + (40 - sy) ** 2)
        df['start_x_norm'] = sx / 120.0
        df['start_y_norm'] = sy / 80.0
    else:
        df['dist_to_goal_center'] = 0.0
        df['start_x_norm'] = 0.0
        df['start_y_norm'] = 0.0

    # Header identification
    if 'shot_body_part' in df.columns:
        df['is_header'] = df['shot_body_part'].astype(str).str.lower().str.contains('head').astype(int)
    else:
        df['is_header'] = 0

    # --- 3. Rolling Player Form (Leakage-Proof) ---
    if 'player_name' in df.columns and 'is_goal' in df.columns:
        def compute_roll(g):
            # Explicitly copy to avoid SettingWithCopy warnings
            g = g.copy() 
            # .shift() is critical: exclude current shot from history
            shifted_goals = g['is_goal'].shift()
            g['player_last5_goals'] = shifted_goals.rolling(window=5, min_periods=1).sum().fillna(0)
            g['player_last5_attempts'] = shifted_goals.rolling(window=5, min_periods=1).count().fillna(0)
            # Add epsilon to prevent ZeroDivisionError
            g['player_last5_conv'] = g['player_last5_goals'] / (g['player_last5_attempts'] + 1e-6)
            return g
            
        # Group without sorting to preserve chronological order
        df = df.groupby('player_name', sort=False, group_keys=False).apply(compute_roll)
    else:
        # Fallback for inference where history might be missing
        if 'player_last5_conv' not in df.columns:
            df['player_last5_conv'] = 0.0

    return df.fillna(0)

def _ensure_season_norm(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['season_norm'])
    
    result = df.copy()
    if 'season_norm' in result.columns:
        return result
        
    # Cascade check for season columns
    for col in ['season', 'season_name']:
        if col in result.columns:
            result['season_norm'] = result[col].fillna('All Time').astype(str).str.strip()
            return result
            
    result['season_norm'] = 'All Time'
    return result

def _filter_by_season(df: pd.DataFrame, season_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
        
    normalized = _ensure_season_norm(df)
    target = str(season_choice or 'All Time').strip()
    
    if target.lower() == 'all time':
        return normalized
        
    season_series = normalized['season_norm'].fillna('All Time').astype(str)
    # Exact match first
    mask = season_series.str.lower() == target.lower()
    
    # Fallback to substring match if exact fails
    if not mask.any():
        mask = season_series.str.lower().str.contains(target.lower())
        
    return normalized[mask].copy()

def _extract_year_from_season_text(text: Optional[str]) -> int:
    """Extracts 4-digit year from season string (e.g., '2019/20' -> 2019)."""
    if not text:
        return 0
    match = re.search(r"(\d{4})", str(text))
    return int(match.group(1)) if match else 0

def _season_sort_key(value: Optional[str]) -> int:
    """Sort helper to handle '2019/2020' strings vs 'All Time'."""
    if value is None:
        return 0
    text = str(value).strip()
    if text == 'All Time':
        return -1
    try:
        # Extract the first year found
        if '/' in text:
            return int(text.split('/')[0])
        if text.isdigit():
            return int(text)
    except Exception:
        pass
    return 0

def _ensure_year_column(df: pd.DataFrame, target_col: str = 'season_year') -> pd.DataFrame:
    """Ensures a numeric year column exists for sorting."""
    if df is None or df.empty:
        return df
    result = df.copy()
    
    # Initialize with zeros
    year_series = pd.Series(0, index=result.index, dtype=int)
    
    if 'extracted_year' in result.columns:
        year_series = result['extracted_year'].fillna(0).astype(int)

    # Fill logic
    for col in ['season', 'season_name', 'season_norm']:
        if col in result.columns:
            # Update only where we still have 0s
            mask = year_series == 0
            if not mask.any(): break
            
            derived = result.loc[mask, col].astype(str).apply(_extract_year_from_season_text).fillna(0).astype(int)
            year_series.loc[mask] = derived

    result[target_col] = year_series
    return result