# app/tabs/radar.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_radar_tab(df):
    st.markdown("### 🧠 MODEL CONFIDENCE RADAR")
    
    if 'model_xg' not in df.columns or df['model_xg'].sum() == 0:
        st.info('Model confidence data is missing or model not loaded.')
        return

    # Filter minimal attempts
    valid_shots = df[['player_name', 'is_goal', 'model_xg', 'start_x']].dropna()
    player_metrics = (
        valid_shots.groupby('player_name')
        .agg(actual_goals=('is_goal', 'sum'), expected_goals=('model_xg', 'sum'), attempts=('start_x', 'count'))
        .reset_index()
    )
    player_metrics = player_metrics[player_metrics['attempts'] >= 5]
    
    if player_metrics.empty:
        st.info('Need more shot data to generate radar (min 5 attempts).')
        return

    player_metrics['confidence_gap'] = (player_metrics['actual_goals'] - player_metrics['expected_goals']).abs()
    highlight = player_metrics.sort_values('confidence_gap', ascending=False).head(6)

    c1, c2 = st.columns([1, 2])
    with c1:
        focus_options = ['Top 6 Players'] + highlight['player_name'].tolist()
        focus_player = st.selectbox('Focus Player', focus_options)
        st.dataframe(highlight[['player_name', 'actual_goals', 'expected_goals']], hide_index=True, use_container_width=True)

    with c2:
        _plot_radar(highlight, focus_player)

def _plot_radar(data, focus):
    # Setup Data
    if focus == 'Top 6 Players':
        cats = data['player_name'].tolist()
        actual = data['actual_goals'].tolist()
        expected = data['expected_goals'].tolist()
    else:
        row = data[data['player_name'] == focus].iloc[0]
        cats = ['Actual Goals', 'Expected xG', 'Gap']
        actual = [row['actual_goals'], row['expected_goals'], row['confidence_gap']]
        expected = [] # Not used for single player view

    # Plotting
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    actual += actual[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'polar': True})
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#0f172a')
    ax.spines['polar'].set_color('#334155')
    ax.tick_params(colors='#94a3b8')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    if focus == 'Top 6 Players':
        if expected: expected += expected[:1]
        ax.plot(angles, actual, linewidth=2, color='#ff0055', label='Actual')
        ax.fill(angles, actual, color='#ff0055', alpha=0.25)
        ax.plot(angles, expected, linewidth=2, color='#00f3ff', label='xG')
        ax.fill(angles, expected, color='#00f3ff', alpha=0.25)
        ax.legend(facecolor='#0f172a', labelcolor='white')
    else:
        ax.plot(angles, actual, linewidth=2, color='#00f3ff')
        ax.fill(angles, actual, color='#00f3ff', alpha=0.3)
        ax.set_title(focus, color='white', pad=20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9)
    st.pyplot(fig, use_container_width=False)