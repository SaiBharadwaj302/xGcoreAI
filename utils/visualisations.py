# utils/visualization.py
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import numpy as np
import streamlit as st

try:
    from mplsoccer import Pitch
except ImportError:
    Pitch = None
    st.error("Library `mplsoccer` is missing. Please run: `pip install mplsoccer`")


def draw_cyber_pitch() -> tuple:
    """Draws a 2D MPLSoccer pitch with cyber aesthetics."""
    if Pitch is None:
        raise ImportError("mplsoccer is required for draw_cyber_pitch; please install mplsoccer.")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='#00f3ff', linewidth=2, goal_type='box')
    fig, ax = pitch.draw(figsize=(10, 6))
    return fig, ax

