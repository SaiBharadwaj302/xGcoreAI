
import sys
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add root to path so we can import utils
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.helpers import sanitize, _add_engineered_features

DATA_DIR = ROOT / 'data' / 'processed'
LEAGUES_DIR = DATA_DIR / 'leagues'
RAW_CANDIDATES = [
    ROOT / 'data' / 'raw' / 'Statbomb' / 'open-data' / 'data',
    ROOT / 'data' / 'raw' / 'Statbomb' / 'open-data',
]

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LEAGUES_DIR.mkdir(parents=True, exist_ok=True)

def find_raw_base(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for p in RAW_CANDIDATES:
        if p.exists(): return p
    return None

def _extract_primary_position(entry):
    """Safe extraction of position from lineup data."""
    if not isinstance(entry, dict): return ''
    positions = entry.get('positions', [])
    # 1. Check for starting XI
    for pos in positions:
        if 'start_reason' in pos and 'Starting XI' in pos['start_reason']:
            return pos.get('position', pos.get('position_name', '')).strip()
    # 2. Fallback to first listed position
    if positions:
        return positions[0].get('position', positions[0].get('position_name', '')).strip()
    return ''

def _parse_event_file(ef, match_players):
    rows = []
    try:
        data = json.loads(ef.read_text(encoding='utf-8'))
    except Exception:
        return rows
    
    match_id = ef.stem
    events = data if isinstance(data, list) else data.get('events', [])

    for ev in events:
        if ev.get('type', {}).get('name', '') != 'Shot':
            continue
            
        player = ev.get('player', {})
        pid = player.get('id')
        
        # Meta lookup
        p_meta = match_players.get(str(match_id), {}).get(pid, {})
        
        # Outcome check
        shot = ev.get('shot', {})
        outcome = shot.get('outcome', {}).get('name', '').lower()
        is_goal = 1 if 'goal' in outcome else 0
        
        # Location
        loc = ev.get('location', [None, None])

        row = {
            'match_id': match_id,
            'competition_name': ev.get('competition', {}).get('name', ''),
            'season': ev.get('season', {}).get('name', ''),
            'player_id': pid,
            'player_name': player.get('name', p_meta.get('player_name', '')),
            'primary_pos': p_meta.get('primary_pos', ''),
            'team_name': ev.get('team', {}).get('name', ''),
            'start_x': loc[0],
            'start_y': loc[1],
            'is_goal': is_goal,
            'minute': ev.get('minute'),
            'shot_body_part': shot.get('body_part', {}).get('name', ''),
            'shot_technique': shot.get('technique', {}).get('name', '')
        }
        rows.append(row)
    return rows

def build_from_raw(raw_base, limit=None):
    # 1. Parse Lineups
    lineup_files = list(raw_base.glob('**/lineups/*.json'))
    match_players = {}
    
    print(f"Parsing {len(lineup_files)} lineup files...")
    for lp in lineup_files:
        try:
            data = json.loads(lp.read_text(encoding='utf-8'))
            mid = lp.stem
            match_players[mid] = {}
            for team in data:
                for p in team.get('lineup', []):
                    match_players[mid][p['player_id']] = {
                        'player_name': p['player_name'],
                        'primary_pos': _extract_primary_position(p)
                    }
        except: continue

    # 2. Parse Events
    event_files = list(raw_base.glob('**/events/*.json'))
    if limit: event_files = event_files[:limit]
    
    print(f"Parsing {len(event_files)} event files...")
    rows = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_parse_event_file, ef, match_players) for ef in event_files]
        for f in futures:
            rows.extend(f.result())

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    raw_base = find_raw_base()

    if not raw_base:
        print("❌ No raw data found.")
        return

    # Build Shots
    shots_df = build_from_raw(raw_base, limit=args.limit)
    if shots_df.empty:
        print("No shots found.")
        return

    # --- CRITICAL FIX: Use Helpers for Features ---
    # This replaces the old manual angle calculations
    shots_df = _add_engineered_features(shots_df)

    # Save Main
    shots_df.to_csv(DATA_DIR / 'shots_final.csv', index=False)
    
    # Save Splits
    if 'competition_name' in shots_df.columns:
        for league, group in shots_df.groupby('competition_name'):
            safe_name = sanitize(league)
            group.to_csv(LEAGUES_DIR / f"shots_{safe_name}.csv", index=False)

    print(f"✅ Preprocessing complete. Processed {len(shots_df)} shots.")

if __name__ == '__main__':
    main()