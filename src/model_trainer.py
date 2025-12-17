import os
import json
from datetime import datetime

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

# Make sure project root is on sys.path so `from utils...` works when running this script directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.config import MIN_ROWS_PER_LEAGUE, MANIFEST_PATH


# --- PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
LEAGUES_DIR = os.path.join(ROOT_DIR, "data", "processed", "leagues")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _ensure_visible_angle(ldf):
    if 'visible_angle' not in ldf.columns or ldf['visible_angle'].isnull().all():
        if 'start_x' in ldf.columns and 'start_y' in ldf.columns:
            from utils.physics import calculate_visible_angle
            ldf['visible_angle'] = ldf.apply(lambda r: calculate_visible_angle(r['start_x'], r['start_y']), axis=1)
        else:
            ldf['visible_angle'] = 0.0


def _add_engineered_features(ldf):
    """Add simple, student-friendly engineered features.

    Features added:
    - angle_sin, angle_cos (from visible_angle in degrees)
    - dist_to_goal_center (distance to goal center at (120,40))
    - is_header (1 if shot_body_part indicates head)
    - start_x_norm, start_y_norm (normalized coordinates 0..1)
    - player_last5_goals, player_last5_attempts, player_last5_conv
    """
    # angle transforms
    try:
        ang_rad = np.deg2rad(ldf['visible_angle'].fillna(0).astype(float))
        ldf['angle_sin'] = np.sin(ang_rad)
        ldf['angle_cos'] = np.cos(ang_rad)
    except Exception:
        ldf['angle_sin'] = 0.0
        ldf['angle_cos'] = 0.0

    # distance to goal center (assuming statsbomb coords, goal at x=120,y=40)
    if 'start_x' in ldf.columns and 'start_y' in ldf.columns:
        ldf['dist_to_goal_center'] = np.sqrt((120 - ldf['start_x'])**2 + (40 - ldf['start_y'])**2)
        ldf['start_x_norm'] = ldf['start_x'] / 120.0
        ldf['start_y_norm'] = ldf['start_y'] / 80.0
    else:
        ldf['dist_to_goal_center'] = 0.0
        ldf['start_x_norm'] = 0.0
        ldf['start_y_norm'] = 0.0

    # header indicator
    if 'shot_body_part' in ldf.columns:
        ldf['is_header'] = ldf['shot_body_part'].astype(str).str.lower().str.contains('head').astype(int)
    else:
        ldf['is_header'] = 0

    # rolling player form features (last 5 shots before current)
    if 'player_name' in ldf.columns and 'is_goal' in ldf.columns:
        # group-preserve order as-is; compute shifted rolling sum
        def compute_rolling(g):
            g = g.copy()
            g['player_last5_goals'] = g['is_goal'].shift().rolling(window=5, min_periods=1).sum().fillna(0)
            g['player_last5_attempts'] = g['is_goal'].shift().rolling(window=5, min_periods=1).count().fillna(0)
            g['player_last5_conv'] = g['player_last5_goals'] / (g['player_last5_attempts'] + 1e-6)
            return g
        ldf = ldf.groupby('player_name', sort=False).apply(compute_rolling).reset_index(drop=True)

    # fill any remaining NaNs
    ldf.fillna(0, inplace=True)
    return ldf


def _encode_categoricals(ldf):
    if 'shot_body_part' in ldf.columns:
        ldf['body_part_code'] = ldf['shot_body_part'].astype('category').cat.codes
    else:
        ldf['body_part_code'] = 0
    if 'shot_technique' in ldf.columns:
        ldf['technique_code'] = ldf['shot_technique'].astype('category').cat.codes
    else:
        ldf['technique_code'] = 0


def _update_manifest(league_key, model_relpath, rows, auc, acc):
    manifest = {}
    try:
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, 'r', encoding='utf-8') as fh:
                manifest = json.load(fh)
    except Exception:
        manifest = {}

    manifest_entry = {
        'league_key': league_key,
        'model_file': model_relpath,
        'rows': rows,
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'auc': auc,
        'accuracy': acc
    }
    manifest[league_key] = manifest_entry
    try:
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write manifest: {e}")


def run_training(hpo: bool = False, league_filter: str = None):
    if not os.path.exists(LEAGUES_DIR):
        print(f"❌ No leagues folder found at {LEAGUES_DIR}. Place per-league shot CSVs in that folder and re-run.")
        return

    features = ['start_x', 'start_y', 'distance', 'visible_angle', 'body_part_code', 'technique_code']

    for fname in os.listdir(LEAGUES_DIR):
        if not (fname.startswith('shots_') and fname.endswith('.csv')):
            continue
        league_key = fname[len('shots_'):-len('.csv')]
        in_file = os.path.join(LEAGUES_DIR, fname)
        try:
            ldf = pd.read_csv(in_file)
        except Exception as e:
            print(f"⚠️ Could not read {in_file}: {e}")
            continue

        if len(ldf) < MIN_ROWS_PER_LEAGUE:
            print(f"ℹ️  Skipping league '{league_key}' (only {len(ldf)} rows < {MIN_ROWS_PER_LEAGUE})")
            continue

        # If caller requested a single league, skip others
        if league_filter and league_key != league_filter:
            continue

        if 'is_goal' not in ldf.columns:
            print(f"⚠️ League file {in_file} missing 'is_goal' — skipping.")
            continue

        # Prepare data
        _ensure_visible_angle(ldf)
        _encode_categoricals(ldf)
        ldf = _add_engineered_features(ldf)

        # Extend feature list with engineered ones (student/simple selection)
        feat_list = features + ['angle_sin', 'angle_cos', 'dist_to_goal_center', 'is_header', 'start_x_norm', 'start_y_norm', 'player_last5_conv']
        # Ensure features exist
        feat_list = [f for f in feat_list if f in ldf.columns]

        X = ldf[feat_list].fillna(0)
        y = ldf['is_goal']

        # Cross-validation AUC (quick check)
        try:
            cv_auc = float(np.mean(cross_val_score(xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, use_label_encoder=False), X, y, cv=3, scoring='roc_auc')))
        except Exception:
            cv_auc = None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training model for league '{league_key}' with {len(ldf)} rows...")

        clf = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric='logloss')

        # Hyperparameter tuning (RandomizedSearchCV) when requested and dataset is reasonably large
        if hpo and len(ldf) >= 300:
            print(f"🔍 Running hyperparameter search for league '{league_key}' (this may take a while)...")
            param_dist = {
                'n_estimators': [100, 200, 300, 400],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            rnd = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_distributions=param_dist, n_iter=20, scoring='roc_auc', cv=3, random_state=42, n_jobs=1)
            try:
                rnd.fit(X_train, y_train)
                clf = rnd.best_estimator_
                best_params = rnd.best_params_
                print(f"✨ Best params for {league_key}: {best_params}")
            except Exception as e:
                print(f"⚠️ HPO failed for {league_key}: {e}")
                clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train)

        # Calibrate probabilities with Platt scaling if we have enough samples
        calibrator = None
        try:
            calibrator = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
            calibrator.fit(X_train, y_train)
        except Exception:
            calibrator = None

        # Save raw model and calibrator (if present)
        out_model = os.path.join(MODEL_DIR, f"goal_predictor_{league_key}.json")
        clf.save_model(out_model)
        if calibrator is not None:
            try:
                joblib.dump(calibrator, os.path.join(MODEL_DIR, f"goal_predictor_{league_key}_calibrator.joblib"))
            except Exception:
                pass

        # Evaluate
        try:
            if calibrator is not None:
                y_prob = calibrator.predict_proba(X_test)[:, 1]
            else:
                y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            auc = float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else None
            acc = float(accuracy_score(y_test, y_pred))
            # Brier score for probability calibration quality
            try:
                brier = float(brier_score_loss(y_test, y_prob))
            except Exception:
                brier = None
        except Exception:
            auc = None
            acc = None
            brier = None

        # Update manifest and add Brier
        relpath = os.path.relpath(out_model, ROOT_DIR)
        _update_manifest(league_key, relpath, len(ldf), auc, acc)
        # Also save a short training report per league
        try:
            report = {
                'league': league_key,
                'rows': len(ldf),
                'auc': auc,
                'accuracy': acc,
                'brier': brier,
                'cv_auc': cv_auc
            }
            with open(os.path.join(MODEL_DIR, f'training_report_{league_key}.json'), 'w', encoding='utf-8') as fh:
                json.dump(report, fh, indent=2)
        except Exception:
            pass

        # Produce over/under performer report: per-player goals vs xG (sum of model predictions)
        try:
            # Generate model_xg for the whole league dataset
            XB = X.fillna(0)
            if calibrator is not None:
                preds = calibrator.predict_proba(XB)[:, 1]
            else:
                preds = clf.predict_proba(XB)[:, 1]
            ldf['model_xg'] = preds
            per_player = ldf.groupby('player_name').agg(actual_goals=('is_goal', 'sum'), xg=('model_xg', 'sum'), attempts=('is_goal', 'count')).reset_index()
            per_player['overperf'] = per_player['actual_goals'] - per_player['xg']
            out_csv = os.path.join(MODEL_DIR, f"overperformers_{league_key}.csv")
            per_player.sort_values('overperf', ascending=False).to_csv(out_csv, index=False)
        except Exception as e:
            print(f"⚠️ Could not generate overperformance report for {league_key}: {e}")

        print(f"✅ Saved league model: {out_model} ({len(ldf)} rows) | AUC: {auc} | ACC: {acc} | CV_AUC: {cv_auc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train per-league xG models')
    parser.add_argument('--hpo', action='store_true', help='Run hyperparameter optimization (may be slow).')
    parser.add_argument('--league', type=str, help='(Optional) league key to train only (e.g. Champions_League)')
    args = parser.parse_args()
    run_training(hpo=args.hpo, league_filter=args.league)