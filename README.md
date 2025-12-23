# xG CoreAI

**Where xG meets intelligence in football analytics.**

xG CoreAI is a modular Streamlit application plus supporting utilities that help analysts, scouts, and coaches go from raw shot and player data to actionable tactical narratives. The project combines calibrated xG models, squad-visualization dashboards, and export-friendly charts that make deep dives into finishing, positioning, and model confidence effortless.

---

##  Core Capabilities

- **Simulation & tactical analysis:** The tabbed UI (app/tabs.py) offers simulations, calibrated metrics, and tactical pitch plots powered by mplsoccer.
- **Best XI generator:** Auto-assembles squad lineups with performance annotations to highlight finishing efficiency and defensive balance.
- **Model confidence radar:** Compares actual goals against predicted xG to spotlight under/over-performing players.
- **Sniper maps:** Plot goals and misses on a neon-inspired cyber pitch with filters for leagues, seasons, teams, and players.
- **Auto-discovery loader:** The utils.data_loader module pulls shots, stats, and models, handling missing files gracefully while surfacing helpful diagnostics.

---

##  Repository layout

```
xGcoreAI/
├── app.py                 # Main Streamlit entry point
├── app/                   # UI components, tabs, TabContext, helpers
├── src/                   # Model tooling: trainers, preprocessors
├── utils/                 # Shared helpers, visualizers, data_loader
├── Data/                  # Raw + processed league exports (ignored)
├── models/                # Calibrated models and manifest metadata
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container recipe
└── scripts/               # Deployment helpers (prepare_data.py)
```

##  Local setup

1. Clone & create virtual environment

```
git clone https://github.com/<you>/xGcoreAI.git
cd xGcoreAI
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

3. Prepare processed data

Place league exports in Data/processed/leagues/ with the following naming convention:

- Shots: shots_<league_key>.csv (e.g., shots_premier_league.csv)
- Stats: stats_<league_key>.csv (e.g., stats_premier_league.csv)

If a per-league stats file is missing, the loader aggregates goals and shots from the global shots feed to keep the tabs functional.

4. Provide models

Ensure trained models (goal_predictor_<league>.json) and calibrators (*_calibrator.joblib) live inside the models/ directory.[^1]

```
python src/model_trainer.py
```

5. Run the app

```
streamlit run app.py
```

Browse http://localhost:8501/ to explore the dashboard.

---

##  Dockerized deployment

1. Build the image

```
docker build -t xgcoreai:latest .
```

2. Run the container with mounted data/model volumes

```
docker run -p 8501:8501 \
	-v "$(pwd)/Data:/app/Data" \
	-v "$(pwd)/models:/app/models" \
	xgcoreai:latest
```

Optionally, run the helper script before launching Streamlit so the container path contains the processed CSVs:

```
python scripts/prepare_data.py
streamlit run app.py
```

---

##  Data & visualization notes

- `Data/raw/` and `Data/processed/` stay out of GitHub by default; preprocess raw exports with src/preprocess.py.
- `utils/visualisations.py` contains shared plotting utilities, including mplsoccer pitches and radar helpers.
- The global loader scans `models/manifest.json` to auto-populate league metadata and performance metrics.
