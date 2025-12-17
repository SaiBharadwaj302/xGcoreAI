import numpy as np
import pandas as pd
import re
from typing import Optional


def sanitize(name):
    return ''.join([c for c in str(name).strip().replace(' ', '_') if c.isalnum() or c in ['_','-']])


def _add_engineered_features(df):
    df = df.copy()
    if 'visible_angle' not in df.columns:
        df['visible_angle'] = 0.0

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
        df['dist_to_goal_center'] = np.sqrt((120 - sx) ** 2 + (40 - sy) ** 2)
        df['start_x_norm'] = sx / 120.0
        df['start_y_norm'] = sy / 80.0
    else:
        df['dist_to_goal_center'] = 0.0
        df['start_x_norm'] = 0.0
        df['start_y_norm'] = 0.0

    if 'shot_body_part' in df.columns:
        df['is_header'] = df['shot_body_part'].astype(str).str.lower().str.contains('head').astype(int)
    elif 'body_part_code' in df.columns:
        df['is_header'] = 0
    else:
        df['is_header'] = 0

    if 'player_name' in df.columns and 'is_goal' in df.columns:
        def compute_roll(g):
            g = g.copy()
            g['player_last5_goals'] = g['is_goal'].shift().rolling(window=5, min_periods=1).sum().fillna(0)
            g['player_last5_attempts'] = g['is_goal'].shift().rolling(window=5, min_periods=1).count().fillna(0)
            g['player_last5_conv'] = g['player_last5_goals'] / (g['player_last5_attempts'] + 1e-6)
            return g
        df = df.groupby('player_name', sort=False).apply(compute_roll).reset_index(drop=True)
    else:
        df['player_last5_conv'] = 0.0

    df.fillna(0, inplace=True)
    return df


def _ensure_season_norm(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=['season_norm'])
    result = df.copy()
    if 'season_norm' in result.columns:
        return result
    if 'season' in result.columns:
        result['season_norm'] = result['season'].fillna('All Time').astype(str).str.strip()
    elif 'season_name' in result.columns:
        result['season_norm'] = result['season_name'].fillna('All Time').astype(str).str.strip()
    else:
        result['season_norm'] = 'All Time'
    return result


def _filter_by_season(df: pd.DataFrame, season_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = _ensure_season_norm(df)
    target = str(season_choice or 'All Time').strip() or 'All Time'
    season_series = normalized['season_norm'].fillna('All Time').astype(str)
    mask = season_series.str.lower() == target.lower()
    if not mask.any():
        mask = season_series.str.lower().str.contains(target.lower())
    return normalized[mask].copy()


def _season_sort_key(value: Optional[str]) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    try:
        if '/' in text:
            return int(text.split('/')[0])
        if text.isdigit():
            return int(text)
    except Exception:
        pass
    if text == 'All Time':
        return -1
    return 0


def _extract_year_from_season_text(text: Optional[str]) -> int:
    if not text:
        return 0
    match = re.search(r"(\d{4})", str(text))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _ensure_year_column(df: pd.DataFrame, target_col: str = 'season_year') -> pd.DataFrame:
    if df is None or df.empty:
        return df
    result = df.copy()
    year_series = pd.Series(0, index=result.index, dtype=int)
    if 'extracted_year' in result.columns:
        year_series = result['extracted_year'].fillna(0).astype(int)

    def _fill_from_column(col: str) -> None:
        nonlocal year_series
        if col not in result.columns:
            return
        mask = year_series == 0
        if not mask.any():
            return
        derived = result.loc[mask, col].astype(str).apply(_extract_year_from_season_text).fillna(0).astype(int)
        year_series.loc[mask] = derived

    for candidate in ('season', 'season_name', 'season_norm'):
        _fill_from_column(candidate)

    result[target_col] = year_series
    return result
