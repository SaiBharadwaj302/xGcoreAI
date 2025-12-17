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
