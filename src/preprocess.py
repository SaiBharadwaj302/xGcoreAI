
"""Consolidated preprocessing script for the project.

This script performs the full preprocessing required by the project:
- Builds `data/processed/shots_final.csv` and `data/processed/player_stats_final.csv`.
- Exports per-league CSVs under `data/processed/leagues/` (shots_... and stats_...).

It prefers existing processed CSVs if present; otherwise, when StatBomb raw
open-data exists under `data/raw/...`, it will build from raw JSON files.

Usage:
    python src/preprocess.py [--force-raw] [--raw-path PATH] [--limit N] [--verbose]

The script is designed to be quiet by default and produce concise, deterministic
outputs suitable for submission.
"""

from __future__ import annotations
import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / 'data' / 'processed'
LEAGUES_DIR = DATA_DIR / 'leagues'

RAW_CANDIDATES = [
    ROOT / 'data' / 'raw' / 'Statbomb' / 'open-data' / 'data',
    ROOT / 'data' / 'raw' / 'Statbomb' / 'open-data',
    ROOT / 'data' / 'raw' / 'open-data' / 'data',
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LEAGUES_DIR.mkdir(parents=True, exist_ok=True)


def find_raw_base(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for p in RAW_CANDIDATES:
        if p.exists():
            return p
    return None


def sanitize(name: str) -> str:
    if not name:
        return 'unknown'
    s = str(name).strip()
    s = s.replace('&', 'and')
    s = re.sub(r'[^0-9A-Za-z \-]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s.lower()


def visible_angle_from_coords(sx: float, sy: float) -> float:
    # Try to use utils.physics.calculate_visible_angle if available
    try:
        from utils.physics import calculate_visible_angle
        return float(calculate_visible_angle(sx, sy))
    except Exception:
        # fallback: approximate visible angle (radians -> degrees)
        # assume goal mouth half-width in pitch units approx 3.66 (meters) scaled
        # using StatBomb pitch length 120 units ~ 105 meters -> scale factor ~ 105/120
        try:
            goal_half_m = 3.66
            meters_per_unit = 105.0 / 120.0
            half_width_units = goal_half_m / meters_per_unit
            dx = 120.0 - float(sx)
            dy = 40.0 - float(sy)
            dist = math.hypot(dx, dy)
            if dist <= 0:
                return 0.0
            angle_rad = 2.0 * math.atan2(half_width_units, dist)
            return float(math.degrees(angle_rad))
        except Exception:
            return 0.0


def _extract_primary_position_from_lineup(entry: Dict) -> str:
    if not isinstance(entry, dict):
        return ''
    positions = entry.get('positions') or []
    if isinstance(positions, list) and positions:
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            start_reason = str(pos.get('start_reason') or '').lower()
            if 'starting' in start_reason and 'xi' in start_reason:
                return (pos.get('position') or pos.get('position_name') or '').strip()
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            if str(pos.get('from')).startswith('00:00'):
                return (pos.get('position') or pos.get('position_name') or '').strip()
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            return (pos.get('position') or pos.get('position_name') or '').strip()
    return (entry.get('position') or entry.get('role') or '').strip()


def _parse_event_file(
    ef: Path, match_players: Dict[str, Dict[int, str]]
) -> List[Dict]:
    rows: List[Dict] = []
    try:
        data = json.loads(ef.read_text(encoding='utf-8'))
    except Exception:
        return rows

    events = (
        data.get('events')
        if isinstance(data, dict) and 'events' in data
        else (data if isinstance(data, list) else [])
    )
    match_id = Path(ef).stem

    for ev in events:
        try:
            etype = (
                (ev.get('type') or {}).get('name')
                if isinstance(ev.get('type'), dict)
                else ev.get('type') or ev.get('event_type')
            )
            if not etype or str(etype).strip().lower() != 'shot':
                continue
            player = ev.get('player') or {}
            pid = player.get('player_id') or player.get('id') or ev.get('player_id')
            player_meta: Dict[str, str] = {}
            if pid is not None:
                meta = match_players.get(str(match_id), {}).get(int(pid))
                if isinstance(meta, dict):
                    player_meta = meta
            pname = (
                player.get('player_name') or
                player.get('name') or
                player_meta.get('player_name', '')
            )
            team = ev.get('team') or ev.get('team_name') or ''
            if isinstance(team, dict):
                team_name = team.get('name') or ''
            else:
                team_name = team
            loc = ev.get('location') or []
            sx = loc[0] if isinstance(loc, list) and len(loc) > 0 else ev.get('start_x') or None
            sy = loc[1] if isinstance(loc, list) and len(loc) > 1 else ev.get('start_y') or None
            shot = ev.get('shot') or {}
            is_goal = 0
            if isinstance(shot, dict):
                outcome = shot.get('outcome')
                if isinstance(outcome, dict):
                    if outcome.get('name') and 'goal' in str(outcome.get('name')).lower():
                        is_goal = 1
                else:
                    if outcome and 'goal' in str(outcome).lower():
                        is_goal = 1
                if shot.get('is_goal'):
                    is_goal = 1
            if ev.get('is_goal'):
                is_goal = 1

            row = {
                'match_id': match_id,
                'competition_name': ev.get('competition', ev.get('competition_name')) or ev.get('tournament') or '',
                'season': ev.get('season') or ev.get('season_name') or '',
                'player_id': int(pid) if pid is not None else None,
                'player_name': pname or '',
                'primary_pos': (
                    player.get('position') or player.get('position_name') or player_meta.get('primary_pos', '') or ''
                ),
                'team_name': team_name or '',
                'start_x': float(sx) if sx is not None else None,
                'start_y': float(sy) if sy is not None else None,
                'is_goal': int(is_goal),
                'minute': ev.get('minute') or ev.get('period') or None,
                'source_file': str(ef.relative_to(ROOT))
            }
            if row['start_x'] is not None and row['start_y'] is not None:
                row['distance'] = math.hypot(120.0 - row['start_x'], 40.0 - row['start_y'])
                row['visible_angle'] = visible_angle_from_coords(row['start_x'], row['start_y'])
            else:
                row['distance'] = None
                row['visible_angle'] = 0.0

            rows.append(row)
        except Exception:
            continue

    return rows


def write_per_league_shots(shots_df: pd.DataFrame) -> None:
    if shots_df.empty:
        return
    shots_df = shots_df.copy()
    if 'competition_name' not in shots_df.columns:
        shots_df['competition_name'] = shots_df.get('league', '')
    shots_df['competition_name'] = shots_df['competition_name'].fillna('Unknown')
    if 'league' not in shots_df.columns:
        shots_df['league'] = shots_df['competition_name']
    else:
        shots_df['league'] = shots_df['league'].fillna(shots_df['competition_name'])
    for name, group in shots_df.groupby('competition_name'):
        fk = sanitize(name)
        out = LEAGUES_DIR / f"shots_{fk}.csv"
        group.to_csv(out, index=False)


def write_per_league_stats(stats_df: pd.DataFrame) -> None:
    if stats_df.empty:
        return
    stats_df = _ensure_season_column(stats_df)
    stats_df = stats_df.copy()
    if 'league' not in stats_df.columns:
        stats_df['league'] = stats_df.get('competition_name', '')
    stats_df['league'] = stats_df['league'].fillna('Unknown')
    for name, group in stats_df.groupby('league'):
        fk = sanitize(name)
        out = LEAGUES_DIR / f"stats_{fk}.csv"
        group.to_csv(out, index=False)


def _ensure_season_column(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return stats_df
    season_series = stats_df['season'] if 'season' in stats_df.columns else pd.Series('', index=stats_df.index)
    season_series = season_series.fillna('').astype(str)
    for fallback in ('season_norm', 'season_name'):
        if fallback not in stats_df.columns:
            continue
        fallback_vals = stats_df[fallback].fillna('').astype(str)
        blank_mask = season_series.str.strip() == ''
        if blank_mask.any():
            season_series = season_series.mask(blank_mask, fallback_vals)
    stats_df = stats_df.copy()
    stats_df['season'] = season_series
    return stats_df


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    shots_file = DATA_DIR / 'shots_final.csv'
    stats_file = DATA_DIR / 'player_stats_final.csv'
    shots_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    if shots_file.exists():
        try:
            shots_df = pd.read_csv(shots_file)
        except Exception:
            shots_df = pd.DataFrame()
    if stats_file.exists():
        try:
            stats_df = pd.read_csv(stats_file)
        except Exception:
            stats_df = pd.DataFrame()
    return shots_df, stats_df


def build_from_raw(raw_base: Path, limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build mapping from match_id -> player_id -> player_name using lineups
    lineup_files = list(raw_base.glob('**/lineups/*.json'))
    match_players: Dict[str, Dict[int, str]] = {}
    for lp in lineup_files:
        try:
            data = json.loads(lp.read_text(encoding='utf-8'))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for team in items:
            if not isinstance(team, dict):
                continue
            mid = str(team.get('match_id') or Path(lp).stem)
            if mid not in match_players:
                match_players[mid] = {}
            lineup = team.get('lineup') or []
            for p in lineup if isinstance(lineup, list) else []:
                pid = p.get('player_id') or p.get('id')
                if pid is None:
                    continue
                player_name = p.get('player_name') or p.get('name') or ''
                match_players[mid][int(pid)] = {
                    'player_name': player_name,
                    'primary_pos': _extract_primary_position_from_lineup(p)
                }

    # Parse event files and extract shot rows
    event_files = list(raw_base.glob('**/events/*.json'))
    if limit and limit > 0:
        event_files = event_files[:limit]

    rows: List[Dict] = []
    if event_files:
        max_workers = max(1, (os.cpu_count() or 1))
        worker_count = min(max_workers, len(event_files))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_parse_event_file, ef, match_players) for ef in event_files]
            for future in futures:
                rows.extend(future.result())

    shots_df = pd.DataFrame(rows)
    if shots_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    return shots_df, pd.DataFrame()


def aggregate_player_stats(shots_df: pd.DataFrame) -> pd.DataFrame:
    if shots_df.empty:
        return pd.DataFrame()

    def _mode(series: pd.Series, default: str) -> str:
        cleaned = series.dropna()
        if cleaned.empty:
            return default
        return cleaned.mode().iloc[0]

    def _normalize_season(value: Optional[str], alt: Optional[str]) -> str:
        for candidate in (value, alt):
            if pd.isna(candidate):
                continue
            text = str(candidate).strip()
            if text:
                return text
        return 'All Time'

    def _extract_year(season_text: str) -> int:
        if not season_text:
            return 0
        m = re.search(r"(\d{4})", season_text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 0
        return 0

    shots_df = shots_df.copy()
    shots_df['season_norm'] = shots_df.apply(lambda r: _normalize_season(r.get('season'), r.get('season_name')), axis=1)
    if 'competition_name' not in shots_df.columns:
        shots_df['competition_name'] = shots_df.get('league', '')
    if 'team_name' not in shots_df.columns:
        shots_df['team_name'] = ''

    group_cols = ['player_name', 'season_norm']
    per_player = (
        shots_df.groupby(group_cols)
        .agg(goals=('is_goal', 'sum'), shots=('player_name', 'count'))
        .reset_index()
    )
    per_player['passes'] = 0
    per_player['team_name'] = (
        shots_df.groupby(group_cols)['team_name'].agg(lambda s: _mode(s, ''))
    ).values
    per_player['primary_pos'] = (
        shots_df.groupby(group_cols)['primary_pos'].agg(lambda s: _mode(s, ''))
    ).values if 'primary_pos' in shots_df.columns else ''
    per_player['att_score'] = per_player['shots'] * 5 + per_player['goals'] * 50
    per_player['mid_score'] = 0
    per_player['def_score'] = 0
    per_player['league'] = (
        shots_df.groupby(group_cols)['competition_name'].agg(lambda s: _mode(s, 'Unknown'))
    ).values
    season_name_series = (
        shots_df.groupby(group_cols)['season_name'].agg(lambda s: _mode(s, ''))
    )
    def _lookup_season_name(row: pd.Series) -> str:
        key = (row['player_name'], row['season_norm'])
        return season_name_series.get(key, row['season_norm']) or row['season_norm']
    per_player['season_name'] = per_player.apply(_lookup_season_name, axis=1)
    per_player['season'] = per_player['season_norm']
    per_player['extracted_year'] = per_player['season_norm'].apply(_extract_year)

    return per_player


def _normalize_match_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map: Dict[str, str] = {}
    if 'competition.competition_id' in df.columns:
        rename_map['competition.competition_id'] = 'competition_id'
    if 'competition.competition_name' in df.columns:
        rename_map['competition.competition_name'] = 'competition_name'
    if 'season.season_id' in df.columns:
        rename_map['season.season_id'] = 'season_id'
    if 'season.season_name' in df.columns:
        rename_map['season.season_name'] = 'season_name'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _coalesce_columns(df: pd.DataFrame, target: str, fallback: str = '') -> None:
    cols = [f"{target}_y", target, f"{target}_x"]
    for col in cols:
        if col in df.columns:
            df[target] = df[col].fillna(fallback)
            break
    else:
        df[target] = fallback
    for col in cols:
        if col in df.columns and col != target:
            df.drop(columns=col, inplace=True)


def load_matches_and_competitions(raw_base: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches and competitions from a StatBomb-style raw repo.
    Tries a few plausible locations under `raw_base`.
    """
    candidates = [
        raw_base / 'matches',
        raw_base / 'data' / 'matches',
        raw_base / '..' / 'data' / 'matches'
    ]
    matches_df = pd.DataFrame()
    for c in candidates:
        if not c.exists():
            continue
        parts = []
        for fp in c.rglob('*.json'):
            try:
                payload = json.loads(fp.read_text(encoding='utf-8'))
            except Exception:
                continue
            records = payload if isinstance(payload, list) else [payload]
            df = pd.json_normalize(records)
            if not df.empty:
                parts.append(df)
        if parts:
            matches_df = pd.concat(parts, ignore_index=True)
            matches_df = _normalize_match_records(matches_df)
            break

    comps_candidates = [
        raw_base / 'competitions.json',
        raw_base / 'data' / 'competitions.json',
        raw_base.parent / 'competitions.json'
    ]
    comps_df = pd.DataFrame()
    for c in comps_candidates:
        if not c.exists():
            continue
        try:
            payload = json.loads(c.read_text(encoding='utf-8'))
        except Exception:
            continue
        records = payload if isinstance(payload, list) else [payload]
        comps_df = pd.json_normalize(records)
        break

    return matches_df, comps_df


def merge_shots_with_matches(shots_df: pd.DataFrame, matches_df: pd.DataFrame, comps_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich shots with season/competition information if available.
    Performs defensive merges and sensible fallbacks.
    """
    if shots_df.empty:
        return shots_df

    df = shots_df.copy()
    # Normalize match id types
    if 'match_id' in matches_df.columns and matches_df['match_id'].dtype != object:
        matches_df['match_id'] = matches_df['match_id'].astype(str)
    if 'match_id' in df.columns:
        df['match_id'] = df['match_id'].astype(str)

    # pick sensible match columns
    match_cols = [c for c in ['match_id', 'season_id', 'competition_id', 'competition_name', 'season_name', 'date', 'match_date'] if c in matches_df.columns]
    if 'match_id' in match_cols:
        try:
            df = df.merge(matches_df[match_cols].drop_duplicates(subset=['match_id']), on='match_id', how='left')
        except Exception:
            pass

    # merge competitions to get readable season/competition names
    if not comps_df.empty:
        # prefer columns named season_name or competition_name
        comp_keys = []
        if 'competition_id' in df.columns and 'competition_id' in comps_df.columns:
            comp_keys.append('competition_id')
        if 'season_id' in df.columns and 'season_id' in comps_df.columns:
            comp_keys.append('season_id')

        # attempt merge; if it fails, skip gracefully
        try:
            if comp_keys:
                cols = comp_keys + [c for c in ['season_name', 'competition_name'] if c in comps_df.columns]
                cols = [c for c in cols if c in comps_df.columns]
                df = df.merge(comps_df[cols].drop_duplicates(), on=comp_keys, how='left')
        except Exception:
            pass

    _coalesce_columns(df, 'competition_name', '')
    _coalesce_columns(df, 'season_name', '')

    # fallback: ensure 'season' column exists
    if 'season' not in df.columns:
        if 'season_name' in df.columns:
            df['season'] = df['season_name']
        elif 'season_id' in df.columns:
            df['season'] = df['season_id'].astype(str)
        else:
            df['season'] = df.get('season', 'All Time')

    # ensure competition_name/league exists
    if 'competition_name' not in df.columns and 'competition' in df.columns:
        df['competition_name'] = df['competition']
    if 'competition_name' not in df.columns:
        df['competition_name'] = df.get('league', '')

    return df


def fix_per_league_assignments(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic: assign each team to its most frequent competition (league).
    Rebuilds per-league grouping on the enriched shots dataframe.
    Returns shots_df with a `file_key` column (sanitized competition key).
    """
    df = shots_df.copy()
    if df.empty:
        return df
    # ensure competition_name present
    if 'competition_name' not in df.columns:
        df['competition_name'] = df.get('league', '')

    # compute team -> competition frequency
    team_comp = df.groupby(['team_name', 'competition_name']).size().reset_index(name='cnt')
    # primary comp per team
    primary = team_comp.sort_values('cnt', ascending=False).drop_duplicates(subset=['team_name'])
    team_to_comp = dict(zip(primary['team_name'], primary['competition_name']))

    df['competition_name'] = df.apply(lambda r: team_to_comp.get(r['team_name'], r['competition_name']), axis=1)
    df['file_key'] = df['competition_name'].apply(sanitize)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Project preprocessing')
    parser.add_argument('--force-raw', action='store_true', help='Force building from raw JSON even if processed CSVs exist')
    parser.add_argument('--raw-path', type=str, default=None, help='Explicit raw data path')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of event files (debug)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args(argv)

    ensure_dirs()
    shots_df, stats_df = load_processed()
    raw_base = find_raw_base(args.raw_path)

    if args.force_raw or shots_df.empty:
        if raw_base is None:
            if args.verbose:
                print('No raw StatBomb data found and no processed files present; nothing to do.')
        else:
            if args.verbose:
                print(f'Building from raw data at: {raw_base}')
            shots_df, _ = build_from_raw(raw_base, limit=(args.limit or None))
            if not shots_df.empty:
                matches_df, comps_df = load_matches_and_competitions(raw_base)
                shots_df = merge_shots_with_matches(shots_df, matches_df, comps_df)
                shots_df = fix_per_league_assignments(shots_df)
                stats_df = aggregate_player_stats(shots_df)
                shots_out = DATA_DIR / 'shots_final.csv'
                stats_out = DATA_DIR / 'player_stats_final.csv'
                shots_df.to_csv(shots_out, index=False)
                stats_df.to_csv(stats_out, index=False)
                if args.verbose:
                    print(f'Wrote {shots_out} ({len(shots_df)} rows) and {stats_out} ({len(stats_df)} rows)')

    # Export per-league CSVs
    if not shots_df.empty:
        write_per_league_shots(shots_df)
    if not stats_df.empty:
        write_per_league_stats(stats_df)

    summary_parts: List[str] = []
    if not shots_df.empty:
        summary_parts.append(f"{len(shots_df)} shot rows")
    if not stats_df.empty:
        summary_parts.append(f"{len(stats_df)} player stats rows")
    if summary_parts:
        print('Preprocessing complete: ' + ' and '.join(summary_parts) + '.')
    else:
        print('Preprocessing finished: no shot or stats rows were generated.')

    if args.verbose:
        print('Verbose logging enabled.')


if __name__ == '__main__':
    main()
