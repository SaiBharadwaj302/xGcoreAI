# utils/config.py
import os

# --- THEME CSS ---
CSS_STYLE = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    .stApp { background-color: #050505; color: #e0f2fe; font-family: 'Rajdhani', sans-serif; }
    div[data-testid="metric-container"] { background: rgba(15, 23, 42, 0.6); border-left: 3px solid #00f3ff; border-radius: 4px; padding: 10px; }
    [data-testid="stMetricLabel"] { font-family: 'Orbitron', sans-serif; color: #94a3b8; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; color: #f8fafc; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #334155; }
    .stTabs [aria-selected="true"] { color: #00f3ff !important; border-bottom-color: #00f3ff !important; }
    .stSlider [data-baseweb="slider"] { color: #f43f5e; }
    div[data-baseweb="select"] { background-color: #0f172a; border-color: #334155; }
    .stProgress > div > div > div > div { background-color: #00f3ff; }
    
    /* Sentient Feed Styling */
    .neural-feed { font-family: 'Consolas', monospace; font-size: 0.85rem; color: #94a3b8; background: #0f172a; padding: 10px; border-radius: 5px; height: 300px; overflow-y: auto; border: 1px solid #1e293b; }
    .event-goal { color: #00f3ff; font-weight: bold; text-shadow: 0 0 5px #00f3ff; }
    .event-card { color: #fbbf24; }
    .event-drama { color: #f43f5e; font-weight: bold; }
    .event-flow { color: #10b981; font-style: italic; }
    
    /* AI Report */
    .ai-report { border: 1px solid #00f3ff; background: rgba(0, 243, 255, 0.05); padding: 15px; border-radius: 8px; font-family: 'Orbitron', sans-serif; margin-top: 20px; }
    </style>
"""

# --- TACTICAL MATRIX ---
TACTICS = {
    "Balanced": {"ATT": 1.0, "MID": 1.0, "DEF": 1.0, "desc": "Standard formation."},
    "Gegenpress": {"ATT": 1.15, "MID": 1.1, "DEF": 0.85, "desc": "High intensity. High risk, high reward."},
    "Catenaccio": {"ATT": 0.8, "MID": 0.9, "DEF": 1.3, "desc": "Defensive fortress. Hard to break down."},
    "Tiki-Taka": {"ATT": 0.95, "MID": 1.25, "DEF": 0.9, "desc": "Total possession control."},
    "Route One": {"ATT": 1.1, "MID": 0.8, "DEF": 1.0, "desc": "Direct long balls. Bypasses midfield."}
}

# --- TRAINING / MODEL CONFIG ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Minimum rows required to train a per-league model (can be overridden with env var)
MIN_ROWS_PER_LEAGUE = int(os.environ.get('MIN_ROWS_PER_LEAGUE', '50'))
# Path to models manifest
MANIFEST_PATH = os.path.join(ROOT_DIR, 'models', 'manifest.json')