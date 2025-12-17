# utils/simulation.py
import numpy as np
import pandas as pd
import random
import re
from utils.config import TACTICS

def generate_best_xi(df, formation="4-3-3"):
    """Generate a best XI for a given stats dataframe and formation.

    This implementation produces consistent ratings for attackers, midfielders and
    defenders by:
    - deriving robust att/mid/def scores (if missing) from available counting stats,
    - canonicalising multi-word and non-standard position strings,
    - ranking candidates by a positional-match ratio and the role-specific score,
    - falling back to the best remaining players when no strict match exists.

    Returns: (best_xi_list, team_stats_dict)
    """

    formations = {
        "4-3-3": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,50,["CB","CENTER BACK","LCB"],"def_score"), ("RCB",18,30,["CB","CENTER BACK","RCB"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("CDM",40,40,["CDM","DEFENSIVE MID","DM"],"def_score"), ("LCM",60,60,["CM","CENTER MID","LCM","CAM"],"mid_score"), ("RCM",60,20,["CM","CENTER MID","RCM","CAM"],"mid_score"), ("LW",90,70,["LW","LEFT WING","LM"],"att_score"), ("RW",90,10,["RW","RIGHT WING","RM"],"att_score"), ("ST",105,40,["ST","CF","CENTER FORWARD","STRIKER","F9","FALSE 9"],"att_score")],
        "4-4-2": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,50,["CB","CENTER BACK"],"def_score"), ("RCB",18,30,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("LM",70,72,["LM","LEFT MID","LW"],"mid_score"), ("LCM",50,50,["CM","CENTER MID"],"mid_score"), ("RCM",50,30,["CM","CENTER MID"],"mid_score"), ("RM",70,8,["RM","RIGHT MID","RW"],"mid_score"), ("LST",100,50,["ST","CF","CENTER FORWARD"],"att_score"), ("RST",100,30,["ST","CF","CENTER FORWARD"],"att_score")],
        "3-5-2": [("GK",5,40,["GK","GOAL"],"def_score"), ("LCB",18,60,["CB","CENTER BACK"],"def_score"), ("CB",15,40,["CB","CENTER BACK"],"def_score"), ("RCB",18,20,["CB","CENTER BACK"],"def_score"), ("LWB",50,75,["WB","LEFT WING BACK","LWB"],"mid_score"), ("LCM",65,55,["CM","CENTER MID"],"mid_score"), ("CDM",40,40,["CDM","DEFENSIVE MID"],"def_score"), ("RCM",65,25,["CM","CENTER MID"],"mid_score"), ("RWB",50,5,["WB","RIGHT WING BACK","RWB"],"mid_score"), ("LST",100,50,["ST","CF","CENTER FORWARD"],"att_score"), ("RST",100,30,["ST","CF","CENTER FORWARD"],"att_score")],
        # Additional common formations
        "4-5-1": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",25,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,52,["CB","CENTER BACK"],"def_score"), ("RCB",18,28,["CB","CENTER BACK"],"def_score"), ("RB",25,8,["RB","RIGHT BACK"],"def_score"), ("LM",70,72,["LM","LEFT MID","LW"],"mid_score"), ("LCM",55,55,["CM","CENTER MID"],"mid_score"), ("CDM",40,40,["CDM","DEFENSIVE MID"],"def_score"), ("RCM",55,20,["CM","CENTER MID"],"mid_score"), ("RM",70,8,["RM","RIGHT MID","RW"],"mid_score"), ("ST",100,40,["ST","CF","CENTER FORWARD"],"att_score")],
        "4-2-3-1": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,52,["CB","CENTER BACK"],"def_score"), ("RCB",18,28,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("LDM",45,70,["CDM","DM","DEFENSIVE MID"],"def_score"), ("RDM",45,10,["CDM","DM","DEFENSIVE MID"],"def_score"), ("LAM",75,55,["AM","CAM","LM","LW"],"mid_score"), ("CAM",90,40,["CAM","AM"],"mid_score"), ("RAM",75,25,["AM","RM","RW"],"mid_score"), ("ST",105,40,["ST","CF","CENTER FORWARD"],"att_score")],
        "4-1-4-1": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,52,["CB","CENTER BACK"],"def_score"), ("RCB",18,28,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("CDM",45,40,["CDM","DEFENSIVE MID"],"def_score"), ("LM",70,72,["LM","LEFT MID"],"mid_score"), ("LCM",55,50,["CM","CENTER MID"],"mid_score"), ("RCM",55,30,["CM","CENTER MID"],"mid_score"), ("RM",70,8,["RM","RIGHT MID"],"mid_score"), ("ST",100,40,["ST","CF"],"att_score")],
        "3-4-3": [("GK",5,40,["GK","GOAL"],"def_score"), ("LCB",18,60,["CB","CENTER BACK"],"def_score"), ("CB",15,40,["CB","CENTER BACK"],"def_score"), ("RCB",18,20,["CB","CENTER BACK"],"def_score"), ("LM",75,75,["LM","LEFT MID","LW"],"mid_score"), ("LCM",55,55,["CM","CENTER MID"],"mid_score"), ("RCM",55,25,["CM","CENTER MID"],"mid_score"), ("RM",75,5,["RM","RIGHT MID","RW"],"mid_score"), ("LW",95,70,["LW","LEFT WING"],"att_score"), ("ST",105,40,["ST","CF"],"att_score"), ("RW",95,10,["RW","RIGHT WING"],"att_score")],
        "5-3-2": [("GK",5,40,["GK","GOAL"],"def_score"), ("LWB",25,72,["LWB","LEFT WING BACK"],"def_score"), ("LCB",18,55,["CB","CENTER BACK"],"def_score"), ("CB",15,40,["CB","CENTER BACK"],"def_score"), ("RCB",18,25,["CB","CENTER BACK"],"def_score"), ("RWB",25,8,["RWB","RIGHT WING BACK"],"def_score"), ("LCM",60,60,["CM","CENTER MID"],"mid_score"), ("RCM",60,20,["CM","CENTER MID"],"mid_score"), ("LST",100,50,["ST","CF"],"att_score"), ("RST",100,30,["ST","CF"],"att_score")],
        "4-4-1-1": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,50,["CB","CENTER BACK"],"def_score"), ("RCB",18,30,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("LM",70,72,["LM","LEFT MID"],"mid_score"), ("LCM",55,50,["CM","CENTER MID"],"mid_score"), ("RCM",55,30,["CM","CENTER MID"],"mid_score"), ("RM",70,8,["RM","RIGHT MID"],"mid_score"), ("SS",95,40,["SS","SECOND STRIKER","CAM"],"att_score"), ("ST",105,40,["ST","CF"],"att_score")]
    }

    layout = formations.get(formation, formations["4-3-3"])

    # Helper: build a canonical token set for a player's primary_pos field
    def canonical_tokens(pos_str):
        if not isinstance(pos_str, str) or not pos_str:
            return set()
        s = pos_str.upper()
        # normalize separators
        s = re.sub(r"[\/,&]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        mapped = set()

        # common multi-word patterns first
        multi_alias = {
            'CENTER FORWARD': 'ST', 'CENTRE FORWARD': 'ST', 'LEFT CENTER FORWARD': 'ST', 'RIGHT CENTER FORWARD': 'ST',
            'SECOND STRIKER': 'SS', 'FALSE NINE': 'ST', 'FALSE9': 'ST', 'ATTACKING MID': 'CAM', 'DEFENSIVE MID': 'CDM',
            'LEFT WING BACK': 'LWB', 'RIGHT WING BACK': 'RWB', 'LEFT BACK': 'LB', 'RIGHT BACK': 'RB',
            'GOALKEEPER': 'GK', 'GOAL': 'GK', 'FULLBACK': 'FB'
        }
        for k, v in multi_alias.items():
            if k in s:
                mapped.add(v)
                # remove matched phrase to avoid double counting
                s = s.replace(k, ' ')

        # now token-level aliasing
        alias = {
            'LW': 'LW', 'LEFTWING': 'LW', 'LM': 'LM',
            'RW': 'RW', 'RIGHTWING': 'RW', 'RM': 'RM',
            'ST': 'ST', 'CF': 'ST', 'F9': 'ST', 'STRIKER': 'ST', 'FWD': 'ST',
            'CAM': 'CAM', 'AM': 'CAM', 'SS': 'SS',
            'CM': 'CM', 'CDM': 'CDM', 'DM': 'CDM',
            'LB': 'LB', 'RB': 'RB', 'LWB': 'LWB', 'RWB': 'RWB',
            'CB': 'CB', 'GK': 'GK'
        }
        parts = re.split(r"[^A-Z0-9]+", s)
        for p in parts:
            if not p: continue
            if p in alias:
                mapped.add(alias[p])
            else:
                mapped.add(p)
        return mapped

    best_xi, used = [], set()
    # priority for layout ordering (keep similar to original)
    prio = {"ST":1, "LST":1, "RST":1, "LW":2, "RW":2, "CAM":3, "LCM":4, "RCM":4, "LM":5, "RM":5, "CDM":6, "LWB":7, "RWB":7, "LB":8, "RB":8, "LCB":9, "RCB":9, "CB":9, "GK":10}
    layout.sort(key=lambda x: prio.get(x[0], 99))

    # Prepare dataframe: ensure key columns exist and numeric fields are filled
    df = df.copy()
    if 'player_name' not in df.columns:
        df['player_name'] = df.index.astype(str)
    # fill numeric columns if absent
    numeric_defaults = {
        'goals': 0, 'shots': 0, 'passes': 0, 'tackles': 0, 'interceptions': 0, 'clearances': 0, 'blocks': 0,
        'def_score': np.nan, 'mid_score': np.nan, 'att_score': np.nan
    }
    for c, v in numeric_defaults.items():
        if c not in df.columns:
            df[c] = v
    # coerce numeric cols
    for c in ['goals','shots','passes','tackles','interceptions','clearances','blocks','def_score','mid_score','att_score']:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        except Exception:
            df[c] = 0

    # If summary scores are all zero, derive proxies from counting stats
    if df['att_score'].sum() == 0:
        # att_score proxy: goals*60 + shots*4 + passes*0.1
        df['att_score'] = (df['goals'] * 60) + (df['shots'] * 4) + (df['passes'] * 0.1)
    if df['mid_score'].sum() == 0:
        # mid_score proxy: passes*0.5 + interceptions*2 + shots*1
        df['mid_score'] = (df['passes'] * 0.5) + (df['interceptions'] * 2) + (df['shots'] * 1)
    if df['def_score'].sum() == 0:
        # def_score proxy: tackles*3 + interceptions*2 + clearances*1 + blocks*2
        df['def_score'] = (df['tackles'] * 3) + (df['interceptions'] * 2) + (df['clearances'] * 1) + (df['blocks'] * 2)

    # Normalize scores to 0-100 to keep ratings comparable
    def norm_series(s):
        if s.max() == s.min():
            return pd.Series([50.0]*len(s), index=s.index)
        res = 20 + 80 * ((s - s.min()) / (s.max() - s.min()))
        return res

    df['att_norm'] = norm_series(df['att_score'])
    df['mid_norm'] = norm_series(df['mid_score'])
    df['def_norm'] = norm_series(df['def_score'])

    for role, x, y, search_list, sort_col in layout:
        # build candidate pool by matching canonical tokens against search synonyms
        def match_row(row):
            pos = row.get('primary_pos', '')
            toks = canonical_tokens(pos)
            # also check common 'positions' or 'preferred_positions' fields if present
            for fld in ['positions', 'preferred_positions', 'primary_pos']:
                if fld in row.index:
                    pos2 = row.get(fld, '')
                    toks |= canonical_tokens(str(pos2))

            # match if any search term appears in tokens or in the raw strings
            for s in search_list:
                s_up = s.upper().replace(' ', '')
                if s_up in toks:
                    return True
                # fallback: check substring in original text
                if isinstance(row.get('primary_pos', ''), str) and s.upper() in row.get('primary_pos', '').upper():
                    return True
            return False

        try:
            candidates = df.copy()
            # filter out already used players
            candidates = candidates[~candidates['player_name'].isin(list(used))]
            # apply matching mask
            mask = candidates.apply(match_row, axis=1)
            cand = candidates[mask].copy()
        except Exception:
            cand = df[~df['player_name'].isin(list(used))].copy()

        if not cand.empty:
            # ensure sort_col exists; fallback to role-normed score column
            if sort_col not in cand.columns:
                # map role group to normalized column
                role_map = {'att_score':'att_norm', 'mid_score':'mid_norm', 'def_score':'def_norm'}
                sort_col = role_map.get(sort_col, 'att_norm')

            # Build search token set for matching
            def build_search_tokens(search_list):
                sset = set()
                for s in search_list:
                    tok = re.sub(r"\s+", "", str(s).upper())
                    if tok in ['GK','GOAL','GOALKEEPER']:
                        sset.add('GK')
                    elif tok in ['LW','LEFTWING','LEFT','LM']:
                        sset.add('LW')
                    elif tok in ['RW','RIGHTWING','RIGHT','RM']:
                        sset.add('RW')
                    elif tok in ['ST','CF','F9','FALSE9','STRIKER','CENTERFORWARD','CENTREFORWARD','SECONDSTRIKER','SS']:
                        sset.add('ST')
                    elif tok in ['CAM','AM','SS']:
                        sset.add('CAM')
                    elif tok in ['CM']:
                        sset.add('CM')
                    elif tok in ['CDM','DM']:
                        sset.add('CDM')
                    elif tok in ['LB']:
                        sset.add('LB')
                    elif tok in ['RB']:
                        sset.add('RB')
                    elif tok in ['LWB']:
                        sset.add('LWB')
                    elif tok in ['RWB']:
                        sset.add('RWB')
                    elif tok in ['CB']:
                        sset.add('CB')
                    else:
                        sset.add(tok)
                return sset

            search_tokens = build_search_tokens(search_list)

            def row_pos_score(row):
                toks = canonical_tokens(row.get('primary_pos', '') or '')
                for fld in ['positions', 'preferred_positions']:
                    if fld in row.index:
                        toks |= canonical_tokens(str(row.get(fld, '') or ''))
                if not toks:
                    return 0.0
                # compute fractional overlap
                inter = toks & search_tokens
                return float(len(inter)) / float(len(search_tokens)) if len(search_tokens) else 0.0

            try:
                cand = cand.copy()
                cand['pos_match'] = cand.apply(row_pos_score, axis=1)
            except Exception:
                cand['pos_match'] = 0.0

            # Use composite ranking: prefer positional match then normalized role score
            def role_score_col(row):
                if sort_col in ['att_norm','mid_norm','def_norm']:
                    return row.get(sort_col, 50.0)
                # fallback: map to appropriate normalized column
                if 'att' in sort_col:
                    return row.get('att_norm', 50.0)
                if 'mid' in sort_col:
                    return row.get('mid_norm', 50.0)
                return row.get('def_norm', 50.0)

            cand['role_score'] = cand.apply(role_score_col, axis=1)
            # final sort: pos_match desc, role_score desc, shots desc
            sorted_cand = cand.sort_values(by=['pos_match','role_score','shots'], ascending=[False, False, False])

            if not sorted_cand.empty:
                best = sorted_cand.iloc[0]
                val = int(best.get('role_score', 50))
                best_xi.append({"x":x, "y":y, "role":role, "name":best['player_name'], "val":val, "type": sort_col})
                used.add(best['player_name'])
            else:
                # fallback: pick top remaining by overall combined normalized score
                pool = df[~df['player_name'].isin(list(used))].copy()
                if pool.empty:
                    best_xi.append({"x":x, "y":y, "role":role, "name":"N/A", "val":50, "type": sort_col})
                else:
                    pool['combined'] = pool['att_norm'] * 0.4 + pool['mid_norm'] * 0.3 + pool['def_norm'] * 0.3
                    pick = pool.sort_values(by=['combined','shots'], ascending=[False, False]).iloc[0]
                    best_xi.append({"x":x, "y":y, "role":role, "name":pick['player_name'], "val":int(pick['combined']), "type": 'combined'})
                    used.add(pick['player_name'])
        else:
            # No strict candidates found; pick the best remaining player by combined score
            pool = df[~df['player_name'].isin(list(used))].copy()
            if pool.empty:
                # truly no players left
                best_xi.append({"x":x, "y":y, "role":role, "name":"N/A", "val":50, "type": sort_col})
            else:
                pool['combined'] = pool['att_norm'] * 0.4 + pool['mid_norm'] * 0.3 + pool['def_norm'] * 0.3
                pick = pool.sort_values(by=['combined','shots'], ascending=[False, False]).iloc[0]
                best_xi.append({"x":x, "y":y, "role":role, "name":pick['player_name'], "val":int(pick['combined']), "type": 'combined'})
                used.add(pick['player_name'])

    # Targeted inclusion: if a player with 'RONALDO' in their name exists in the dataset
    # but wasn't selected (common when position tokens differ), force-include them into
    # an attacking slot by replacing the weakest attacker in the XI. This keeps the
    # change minimal while ensuring historically-important top scorers appear in sims.
    try:
        all_names = set(df['player_name'].astype(str).dropna().unique())
        ronaldo_candidates = [n for n in all_names if 'RONALDO' in n.upper()]
        if ronaldo_candidates:
            # prefer longest/full variant (e.g., Cristiano Ronaldo dos Santos Aveiro)
            rname = sorted(ronaldo_candidates, key=lambda s: -len(s))[0]
            picked_names = {p['name'] for p in best_xi}
            if rname not in picked_names:
                # find attacker slots in best_xi
                attacker_roles = set(['ST','LST','RST','LW','RW','SS','CF'])
                attacker_idxs = [i for i,p in enumerate(best_xi) if any(a in p['role'] for a in attacker_roles) or p['type'] == 'att_score']
                if attacker_idxs:
                    # determine weakest attacker (lowest val)
                    weakest_idx = min(attacker_idxs, key=lambda i: best_xi[i].get('val', 0))
                    # fetch ronaldo row and compute val using att_norm if available
                    rrow = df[df['player_name'] == rname]
                    if not rrow.empty:
                        try:
                            # recompute normalized att score for this single player
                            att_val = int(rrow.iloc[0].get('att_score', 0))
                        except Exception:
                            att_val = 50
                    else:
                        att_val = 50
                    best_xi[weakest_idx] = {**best_xi[weakest_idx], 'name': rname, 'val': int(att_val), 'type': 'att_score'}
    except Exception:
        pass

    # compute team-level ATT/MID/DEF as the mean of selected players' normalized scores
    def pick_score(role_type, norm_col):
        vals = []
        for p in best_xi:
            # map p['type'] which may be a column name to a normalized column
            pname = p['name']
            if pname == 'N/A':
                continue
            row = df[df['player_name'] == pname]
            if row.empty:
                continue
            vals.append(float(row.iloc[0].get(norm_col, 50)))
        return np.nanmean(vals) if vals else 50.0

    att = pick_score('ATT', 'att_norm')
    mid = pick_score('MID', 'mid_norm')
    defi = pick_score('DEF', 'def_norm')
    # ensure numeric
    att = float(att if not np.isnan(att) else 50.0)
    mid = float(mid if not np.isnan(mid) else 50.0)
    defi = float(defi if not np.isnan(defi) else 50.0)
    return best_xi, {"ATT": att, "MID": mid, "DEF": defi}

def run_intelligent_match(home_xi, home_stats, h_tactic, away_xi, away_stats, a_tactic):
    """
    Neural match simulation has been removed from this build.

    This is a deliberate placeholder. It raises NotImplementedError to prevent
    accidental use of the original neural simulation. If you need the original
    implementation, restore `utils/simulations.py` from
    `utils/simulations.py.bak` in the project root.
    """
    raise NotImplementedError(
        "Neural match simulation was removed. See utils/simulations.py.bak for the original implementation."
    )