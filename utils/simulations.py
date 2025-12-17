# utils/simulations.py
import numpy as np
import pandas as pd
import re
import sys

# --- OPTIMIZATION LIBRARY ---
try:
    import pulp
except ImportError:
    print("⚠️ 'pulp' library not found. Install it for perfect lineups: pip install pulp")
    pulp = None



def generate_best_xi(df, formation="4-3-3"):
    """
    Generate the mathematically optimal Best XI using Linear Programming.
    
    This replaces the old 'greedy' approach. It defines a cost matrix where
    Cost = -(Player Rating) and solves for the minimum cost (Maximum Rating)
    subject to constraints:
    1. Each slot in the formation must have exactly 1 player.
    2. Each player can be used at most once.
    """

    # --- 1. Define Formations (Same structure as before) ---
    formations = {
        "4-3-3": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,50,["CB","CENTER BACK","LCB"],"def_score"), ("RCB",18,30,["CB","CENTER BACK","RCB"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("CDM",40,40,["CDM","DEFENSIVE MID","DM"],"def_score"), ("LCM",60,60,["CM","CENTER MID","LCM","CAM"],"mid_score"), ("RCM",60,20,["CM","CENTER MID","RCM","CAM"],"mid_score"), ("LW",90,70,["LW","LEFT WING","LM"],"att_score"), ("RW",90,10,["RW","RIGHT WING","RM"],"att_score"), ("ST",105,40,["ST","CF","CENTER FORWARD","STRIKER","F9","FALSE 9"],"att_score")],
        "4-4-2": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,50,["CB","CENTER BACK"],"def_score"), ("RCB",18,30,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("LM",70,72,["LM","LEFT MID","LW"],"mid_score"), ("LCM",50,50,["CM","CENTER MID"],"mid_score"), ("RCM",50,30,["CM","CENTER MID"],"mid_score"), ("RM",70,8,["RM","RIGHT MID","RW"],"mid_score"), ("LST",100,50,["ST","CF","CENTER FORWARD"],"att_score"), ("RST",100,30,["ST","CF","CENTER FORWARD"],"att_score")],
        "3-5-2": [("GK",5,40,["GK","GOAL"],"def_score"), ("LCB",18,60,["CB","CENTER BACK"],"def_score"), ("CB",15,40,["CB","CENTER BACK"],"def_score"), ("RCB",18,20,["CB","CENTER BACK"],"def_score"), ("LWB",50,75,["WB","LEFT WING BACK","LWB"],"mid_score"), ("LCM",65,55,["CM","CENTER MID"],"mid_score"), ("CDM",40,40,["CDM","DEFENSIVE MID"],"def_score"), ("RCM",65,25,["CM","CENTER MID"],"mid_score"), ("RWB",50,5,["WB","RIGHT WING BACK","RWB"],"mid_score"), ("LST",100,50,["ST","CF","CENTER FORWARD"],"att_score"), ("RST",100,30,["ST","CF","CENTER FORWARD"],"att_score")],
        "4-2-3-1": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,52,["CB","CENTER BACK"],"def_score"), ("RCB",18,28,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("LDM",45,70,["CDM","DM","DEFENSIVE MID"],"def_score"), ("RDM",45,10,["CDM","DM","DEFENSIVE MID"],"def_score"), ("LAM",75,55,["AM","CAM","LM","LW"],"mid_score"), ("CAM",90,40,["CAM","AM"],"mid_score"), ("RAM",75,25,["AM","RM","RW"],"mid_score"), ("ST",105,40,["ST","CF","CENTER FORWARD"],"att_score")],
        "4-4-1-1": [("GK",5,40,["GK","GOAL"],"def_score"), ("LB",30,72,["LB","LEFT BACK"],"def_score"), ("LCB",18,50,["CB","CENTER BACK"],"def_score"), ("RCB",18,30,["CB","CENTER BACK"],"def_score"), ("RB",30,8,["RB","RIGHT BACK"],"def_score"), ("LM",70,72,["LM","LEFT MID"],"mid_score"), ("LCM",55,50,["CM","CENTER MID"],"mid_score"), ("RCM",55,30,["CM","CENTER MID"],"mid_score"), ("RM",70,8,["RM","RIGHT MID"],"mid_score"), ("SS",95,40,["SS","SECOND STRIKER","CAM"],"att_score"), ("ST",105,40,["ST","CF"],"att_score")]
    }

    layout = formations.get(formation, formations["4-3-3"])
    
    # --- 2. Data Cleaning & Scoring (Same as before) ---
    df = df.copy()
    if 'player_name' not in df.columns:
        df['player_name'] = df.index.astype(str)
        
    # Ensure numeric
    cols = ['goals','shots','passes','tackles','interceptions','clearances','blocks','def_score','mid_score','att_score']
    for c in cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Heuristic scoring if missing
    if df['att_score'].sum() == 0:
        df['att_score'] = (df['goals'] * 60) + (df['shots'] * 4) + (df['passes'] * 0.1)
    if df['mid_score'].sum() == 0:
        df['mid_score'] = (df['passes'] * 0.5) + (df['interceptions'] * 2) + (df['shots'] * 1)
    if df['def_score'].sum() == 0:
        df['def_score'] = (df['tackles'] * 3) + (df['interceptions'] * 2) + (df['clearances'] * 1) + (df['blocks'] * 2)

    # Normalization
    def norm_series(s):
        if s.max() == s.min(): return pd.Series([50.0]*len(s), index=s.index)
        return 20 + 80 * ((s - s.min()) / (s.max() - s.min()))

    df['att_norm'] = norm_series(df['att_score'])
    df['mid_norm'] = norm_series(df['mid_score'])
    df['def_norm'] = norm_series(df['def_score'])

    # Helper for Position Matching
    def canonical_tokens(pos_str):
        if not isinstance(pos_str, str) or not pos_str: return set()
        s = pos_str.upper().replace('/', ' ').replace(',', ' ')
        
        # Aliases
        aliases = {
            'CENTER FORWARD':'ST', 'STRIKER':'ST', 'CF':'ST', 'F9':'ST',
            'LEFT WING':'LW', 'RIGHT WING':'RW', 
            'ATTACKING MID':'CAM', 'CAM':'CAM', 
            'DEFENSIVE MID':'CDM', 'DM':'CDM',
            'CENTER MID':'CM', 'CM':'CM',
            'LEFT BACK':'LB', 'RIGHT BACK':'RB', 'FULLBACK':'FB',
            'CENTER BACK':'CB', 'CB':'CB',
            'GOALKEEPER':'GK', 'GK':'GK'
        }
        
        tokens = set()
        for word in s.split():
            if word in aliases: tokens.add(aliases[word])
            else: tokens.add(word)
        
        # Phrase checking
        for k, v in aliases.items():
            if k in s: tokens.add(v)
            
        return tokens

    # --- 3. LINEAR PROGRAMMING OPTIMIZATION ---
    # If pulp is missing, we fall back to a simplified greedy logic, but we assume it's there.
    if pulp is None:
        # Fallback (Shortened Greedy for safety)
        best_xi = []
        used = set()
        for role, x, y, search_list, sort_col in layout:
            best_xi.append({"x":x, "y":y, "role":role, "name":"Please install PuLP", "val":0, "type":sort_col})
        return best_xi, {"ATT":0, "MID":0, "DEF":0}

    # A. Setup Problem
    prob = pulp.LpProblem("Best_XI_Selection", pulp.LpMaximize)
    
    # B. Filter Candidates (To speed up solver)
    # We only keep players who have at least ONE matching token for ANY slot in the formation
    all_needed_tokens = set()
    for _, _, _, s_list, _ in layout:
        for item in s_list:
            all_needed_tokens.add(item.upper())
            # Add simple mapping for search terms
            if 'WING' in item.upper(): all_needed_tokens.add('LW'); all_needed_tokens.add('RW')
    
    # Pre-calculate player tokens
    df['tokens'] = df['primary_pos'].apply(canonical_tokens)
    
    # Create Binary Variables: x_player_slot
    # We store valid (player_idx, slot_idx) tuples to avoid creating variables for impossible positions (e.g. GK as ST)
    valid_vars = []
    
    # To map back later
    slot_definitions = [] # list of (role, sort_col)
    
    players = df.reset_index(drop=True)
    player_indices = players.index.tolist()
    
    # Weights for scores (0-100)
    scores = {} # Key: (player_idx, slot_idx), Value: Score
    
    for slot_idx, (role, x, y, search_list, sort_col) in enumerate(layout):
        slot_definitions.append((role, sort_col))
        
        # Define search set for this slot
        search_set = set()
        for s in search_list:
            normalized = canonical_tokens(s)
            search_set.update(normalized)
            search_set.add(s.upper())

        # Identify eligible players for this slot
        for p_idx, row in players.iterrows():
            p_tokens = row['tokens']
            # Also check 'positions' list if exists
            if 'positions' in row and isinstance(row['positions'], list):
                for p in row['positions']: p_tokens.update(canonical_tokens(str(p)))

            # Match Logic: Do they share a token?
            is_match = not p_tokens.isdisjoint(search_set)
            
            # Special case: If search_list has 'GK', strictly enforce 'GK'
            if 'GK' in search_set and 'GK' not in p_tokens:
                is_match = False
            elif 'GK' not in search_set and 'GK' in p_tokens:
                is_match = False # Don't put GK in outfield

            if is_match:
                # Calculate Score
                base_score = row.get(sort_col.replace('_score', '_norm'), 50)
                if pd.isna(base_score): base_score = 50
                
                # Add to variable list
                valid_vars.append((p_idx, slot_idx))
                scores[(p_idx, slot_idx)] = base_score

    # C. Create Pulp Variables
    # x[(p, s)] = 1 if player p is in slot s
    x_vars = pulp.LpVariable.dicts("x", valid_vars, cat=pulp.LpBinary)

    # D. Objective Function: Maximize Sum of Scores
    prob += pulp.lpSum([scores[k] * x_vars[k] for k in valid_vars])

    # E. Constraints
    
    # 1. One player per slot
    for s in range(len(layout)):
        # Sum of x_p,s for all p must be 1
        relevant_vars = [x_vars[(p, s)] for p, _ in valid_vars if _ == s]
        if relevant_vars:
            prob += pulp.lpSum(relevant_vars) == 1
        else:
            # If no player matches a slot (rare), relax constraint or warn
            pass 

    # 2. One slot per player (Max)
    # A player cannot be in two places at once
    involved_players = set([p for p, s in valid_vars])
    for p in involved_players:
        p_vars = [x_vars[(p, s)] for _, s in valid_vars if _ == p]
        prob += pulp.lpSum(p_vars) <= 1

    # F. Solve
    # Suppress output
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # G. Extract Results
    best_xi = []
    
    # Helper to find assigned player for a slot
    assigned_map = {} # slot_idx -> player_row
    
    if prob.status == pulp.LpStatusOptimal:
        for p, s in valid_vars:
            if x_vars[(p, s)].value() == 1:
                assigned_map[s] = players.iloc[p]
    
    # Build Output
    team_att, team_mid, team_def = [], [], []
    
    for slot_idx, (role, x, y, _, sort_col) in enumerate(layout):
        player = assigned_map.get(slot_idx)
        
        if player is not None:
            val = int(scores.get((player.name, slot_idx), player.get(sort_col.replace('_score', '_norm'), 50)))
            name = player['player_name']
            
            # Stats for team rating
            if 'att' in sort_col: team_att.append(val)
            elif 'mid' in sort_col: team_mid.append(val)
            elif 'def' in sort_col: team_def.append(val)
            else: team_def.append(val) # GK usually
            
            best_xi.append({"x":x, "y":y, "role":role, "name":name, "val":val, "type":sort_col})
        else:
            best_xi.append({"x":x, "y":y, "role":role, "name":"Vacant", "val":0, "type":sort_col})

    # Averages
    avg_att = np.mean(team_att) if team_att else 50
    avg_mid = np.mean(team_mid) if team_mid else 50
    avg_def = np.mean(team_def) if team_def else 50

    return best_xi, {"ATT": avg_att, "MID": avg_mid, "DEF": avg_def}

def run_intelligent_match(home_xi, home_stats, h_tactic, away_xi, away_stats, a_tactic):
    """
    Placeholder for match simulation logic.
    """
    raise NotImplementedError("Neural match simulation was removed.")