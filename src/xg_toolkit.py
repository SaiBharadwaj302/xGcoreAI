import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
from pathlib import Path

FEATURES = ['start_x', 'start_y', 'distance', 'visible_angle', 'body_part_code', 'technique_code']


def _load_model_and_calibrator(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    # try to find calibrator
    base = os.path.splitext(model_path)[0]
    calibrator_path = base + '_calibrator.joblib'
    calibrator = None
    if os.path.exists(calibrator_path):
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception:
            calibrator = None
    return model, calibrator


def _row_to_features(row: pd.Series):
    vals = []
    for f in FEATURES:
        if f in row.index:
            vals.append(row[f])
        else:
            vals.append(0)
    return np.array(vals).reshape(1, -1)


def predict_xg(row: pd.Series, model, calibrator=None):
    X = _row_to_features(row)
    if calibrator is not None:
        return float(calibrator.predict_proba(X)[:, 1][0])
    else:
        try:
            return float(model.predict_proba(X)[:, 1][0])
        except Exception:
            return float(model.predict(xgb.DMatrix(X)))


def predict_xg_array(X_arr: np.ndarray, model, calibrator=None):
    """Predict probabilities for a 2D numpy array of features."""
    try:
        if calibrator is not None:
            return calibrator.predict_proba(X_arr)[:, 1]
        else:
            return model.predict_proba(X_arr)[:, 1]
    except Exception:
        try:
            dmat = xgb.DMatrix(X_arr)
            return model.get_booster().predict(dmat)
        except Exception:
            return np.zeros(X_arr.shape[0])


def suggest_counterfactuals(row: pd.Series, model_path: str, max_suggestions: int = 5, model=None, calibrator=None, base_xg: float = None):
    """Given a shot row (pandas Series), return suggested small perturbations that increase xG.

    Strategy: try small decreases in distance and small increases in visible_angle and report top deltas.
    """
    # Load model (and calibrator if available)
    # Load model/calibrator only if not passed in (batch_suggest will pass them)
    if model is None or calibrator is None:
        model, calibrator = _load_model_and_calibrator(model_path)

    # base xG for the original shot (can be provided by caller to avoid re-predict)
    if base_xg is None:
        base_xg = predict_xg(row, model, calibrator)

    # simple candidate grid: a few distance reductions and angle increases
    dist_deltas = [-6, -4, -2, -1]
    angle_deltas = [3, 6, 10]

    candidates = []
    meta = []
    for dd in dist_deltas:
        for ad in angle_deltas:
            r2 = row.copy()
            if 'distance' in r2.index:
                try:
                    r2['distance'] = max(0.1, float(r2['distance']) + dd)
                except Exception:
                    r2['distance'] = dd
            if 'visible_angle' in r2.index:
                try:
                    r2['visible_angle'] = float(r2['visible_angle']) + ad
                except Exception:
                    r2['visible_angle'] = ad
            candidates.append(_row_to_features(r2).ravel())
            meta.append({'dist_delta': dd, 'angle_delta': ad})

    # Predict all candidates in one batch for speed
    Xcand = np.vstack(candidates)
    try:
        if calibrator is not None:
            probs = calibrator.predict_proba(Xcand)[:, 1]
        else:
            probs = model.predict_proba(Xcand)[:, 1]
    except Exception:
        # fallback: zeros if prediction fails
        probs = np.zeros(Xcand.shape[0])

    results = []
    for m, p in zip(meta, probs):
        results.append({'dist_delta': m['dist_delta'], 'angle_delta': m['angle_delta'], 'base_xg': base_xg, 'new_xg': float(p), 'xg_delta': float(p - base_xg)})

    df = pd.DataFrame(results).sort_values('xg_delta', ascending=False)
    return df.head(max_suggestions)


def batch_suggest(input_csv: str, model_path: str, output_csv: str = None):
    df = pd.read_csv(input_csv)
    rows = []

    # Load model/calibrator once and compute base xG for all rows in a single batch
    model, calibrator = _load_model_and_calibrator(model_path)
    feats = []
    for _, r in df.iterrows():
        feats.append(_row_to_features(r).ravel())
    if feats:
        Xall = np.vstack(feats)
        base_probs = predict_xg_array(Xall, model, calibrator)
    else:
        base_probs = np.array([])

    for idx, r in df.iterrows():
        try:
            if 'is_goal' in r.index and int(r['is_goal']) == 1:
                continue
        except Exception:
            pass
        base_xg = float(base_probs[idx]) if idx < len(base_probs) else None
        sug = suggest_counterfactuals(r, model_path, max_suggestions=3, model=model, calibrator=calibrator, base_xg=base_xg)
        for _, s in sug.iterrows():
            out = {
                'orig_index': idx,
                'player_name': r.get('player_name', ''),
                'dist_delta': s['dist_delta'],
                'angle_delta': s['angle_delta'],
                'base_xg': s['base_xg'],
                'new_xg': s['new_xg'],
                'xg_delta': s['xg_delta']
            }
            rows.append(out)
    out_df = pd.DataFrame(rows)
    if output_csv:
        out_df.to_csv(output_csv, index=False)
    return out_df


def list_player_stats(player_name: str | None = None, limit: int = 20) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'player_stats_final.csv'
    if not base_path.exists():
        raise FileNotFoundError(f"Processed stats file not found at {base_path}")
    df = pd.read_csv(base_path)
    if 'att_score' in df.columns:
        df = df.sort_values('att_score', ascending=False)
    if player_name:
        mask = df['player_name'].astype(str).str.contains(player_name, case=False, na=False)
        df = df[mask]
    if not player_name:
        df = df.head(limit)
    return df[['player_name', 'team_name', 'primary_pos', 'league', 'season', 'goals', 'shots', 'att_score']]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Counterfactual xG suggestions / player stats')
    parser.add_argument('--input', '-i', help='Input shots CSV (missed shots will be suggested)')
    parser.add_argument('--model', '-m', help='Path to a saved model JSON (e.g., models/goal_predictor_League.json)')
    parser.add_argument('--out', '-o', help='Optional output CSV for suggestions')
    parser.add_argument('--player-stats', '-p', action='store_true', help='Show per-player stats from processed data')
    parser.add_argument('--player-name', help='Filter player stats to names matching this string (case-insensitive)')
    parser.add_argument('--limit', '-l', type=int, default=20, help='Number of rows to show when listing stats without a player filter')
    args = parser.parse_args()
    if args.player_stats:
        stats = list_player_stats(args.player_name, limit=args.limit)
        if stats.empty:
            print('No players matched the query.')
        else:
            print(stats.to_string(index=False))
        sys.exit(0)
    if not args.input or not args.model:
        parser.error('The --input and --model arguments are required unless --player-stats is used.')
    res = batch_suggest(args.input, args.model, args.out)
    print(f"Generated {len(res)} suggestions")