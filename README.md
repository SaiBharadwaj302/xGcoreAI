# xG CoreAI

A Streamlit-based AI toolkit for exploring football shot data, predicting xG outcomes, and benchmarking squad performance per-league. The app loads per-league shots, stats, and pre-trained models/calibrators to paint tactical pitches, squad selection, confidence radars, and sniper-shot maps.

## Key Features
- **Simulation & Best XI tabs** powered by `app/tabs.py`, showing calibrated metrics, tactical pitch plots, and squad lineups with player annotations.
- **Model confidence radar** that compares actual goals vs predicted xG for standout players and provides selectable focus insights.
- **Sniper Map** that overlays goals/misses on a cyber pitch with per-player filtering (league, team, season/year).
- **Modular utilities** (`utils/`) supporting configuration, visualization helpers, and data loading that discovers leagues, stats, suggestions, and model/calibrator files.

## Repository Layout
```
app.py                 # Streamlit entrypoint (tabs, context, data prep)
app/                   # Refactored UI tab helpers + TabContext
src/                   # Model training toolkit (xgboost, preprocessing, trainers)
utils/                 # Shared helpers, configurations, visualizers
Data/                  # Raw league shot exports (ignored in git by default)
models/                # Trained models + calibrators per league
requirements.txt       # Python packages to install
Dockerfile             # Container recipe for running the Streamlit app
.dockerignore          # Keeps local artifacts out of Docker context
```

## Local Setup
1. **Clone & activate environment**
   ```bash
   git clone https://github.com/<you>/xGcoreAI.git
   cd xGcoreAI
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Prepare per-league data**
   - Place per-league CSV exports in `Data/processed/leagues/` using the naming schema `shots_<league>.csv` and `stats_<league>.csv` (e.g. `shots_premier_league.csv`).
   - You can derive exports from your own database or use the preprocessing scripts under `src/` to convert a raw source into the required format.
   - If raw data is not available, the app will fall back to aggregated stats derived from the global `shots_df`/`stats_df` loaded via `utils.data_loader`.
4. **Download/Train Models**
   - Pre-trained models/calibrators go inside `models/` with names like `goal_predictor_<league>.json` and `_calibrator.joblib`. Replace them with your own training outputs if needed.
   - Optionally train models using `src/model_trainer.py` once data is in place: `python src/model_trainer.py`.
5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Access the UI at `http://localhost:8501`.

## Dockerized Deployment
1. **Build the image**
   ```bash
   docker build -t xgcoreai:latest .
   ```
2. **Run the container (bind data/models if not baked into image)**
   ```bash
   docker run -p 8501:8501 \
     -v "$(pwd)/Data:/app/Data" \
     -v "$(pwd)/models:/app/models" \
     xgcoreai:latest
   ```
3. **Visit `http://localhost:8501`** in your browser. The container uses `streamlit` to serve the app on port 8501 with league data/models available through the mounted volumes.

## Data Pipeline Notes
- `Data/raw/` and `Data/processed/` are ignored from git; keep your raw shot exports there.
- Use the preprocessing scripts to unify column names and compute engineered features via `src/preprocess.py`. The global loader automatically looks for `models/manifest.json` to populate `league_file_map` and metrics.
- **Visualizations** are handled in `utils/visualisations.py` and `app/tabs.py`. `draw_cyber_pitch()` uses `mplsoccer.Pitch` for neon-style shot and best-XI boards, while other charts rely on matplotlib (radar plots in the Model Confidence tab, scatter layouts, and pitch overlays). The Streamlit renderers call these helpers to keep consistent colors, legends, and layout tweaks.
- If you only have raw shots and not stats, the app attempts to rebuild player stats by aggregating goals/shots per player.

## Contribution Tips
- Keep `app/tabs.py` synced with new UI requirements; it now houses `TabContext` plus the `render_simulation_tab`/`render_squad_genome_tab` helpers.
- Update `utils/helpers.py` or `utils/visualisations.py` when adding new derived columns or visualization tweaks.
- Use `pip freeze` to capture any added dependencies before editing `requirements.txt`.



