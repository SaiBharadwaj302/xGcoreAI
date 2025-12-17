import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Fix: Import specific logic from simulations, NOT physics
from utils.simulations import generate_best_xi
from utils.helpers import sanitize

@dataclass
class TabContext:
    """Holds the state required by various tabs to render."""
    f_shots: pd.DataFrame
    normalized_stats: pd.DataFrame
    normalized_shots: pd.DataFrame
    shots_df: pd.DataFrame
    league_file_map: Dict[str, str]
    active_league: str
    file_key: Optional[str]
    model: Any
    calibrator: Any

def render_simulation_tab(tab, ctx: TabContext):
    with tab:
        st.markdown("### 🎯 MATCH SIMULATION & PREDICTION")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("Select two teams to compare their aggregate stats and win probability based on xG performance.")
            
            # Team Selection
            if 'team_name' in ctx.f_shots.columns:
                teams = sorted(ctx.f_shots['team_name'].dropna().unique())
                team_a = st.selectbox("Home Team", teams, index=0, key="sim_team_a")
                team_b = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0, key="sim_team_b")
                
                if team_a and team_b:
                    # Simple comparison logic
                    stats_a = ctx.f_shots[ctx.f_shots['team_name'] == team_a]
                    stats_b = ctx.f_shots[ctx.f_shots['team_name'] == team_b]
                    
                    xg_a = stats_a['model_xg'].sum() if 'model_xg' in stats_a.columns else 0
                    xg_b = stats_b['model_xg'].sum() if 'model_xg' in stats_b.columns else 0
                    
                    st.metric(f"{team_a} Total xG", f"{xg_a:.2f}")
                    st.metric(f"{team_b} Total xG", f"{xg_b:.2f}")
                    
                    # Basic Win Prob Visualization (Heuristic)
                    total_xg = xg_a + xg_b + 0.001
                    prob_a = xg_a / total_xg
                    prob_b = xg_b / total_xg
                    
                    st.progress(prob_a, text=f"Win Probability: {team_a} ({prob_a:.0%}) vs {team_b} ({prob_b:.0%})")
            else:
                st.warning("Team data not found in this dataset.")

def render_squad_genome_tab(tab, ctx: TabContext, formations: List[str]):
    with tab:
        st.markdown("### 🧬 SQUAD GENOME (BEST XI)")
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            formation = st.selectbox("Formation", formations, key="best_xi_fmt")
            
            # Team Filter for Best XI
            teams = ["All Teams"]
            if 'team_name' in ctx.normalized_stats.columns:
                teams += sorted(ctx.normalized_stats['team_name'].dropna().unique().tolist())
            
            sel_team = st.selectbox("Filter Team", teams, key="best_xi_team")
            
        with c2:
            st.caption("Generating optimal lineup using Linear Programming...")
            
            # Filter Data
            df_pool = ctx.normalized_stats.copy()
            if sel_team != "All Teams":
                df_pool = df_pool[df_pool['team_name'] == sel_team]
            
            if df_pool.empty:
                st.warning("Not enough player data to generate a lineup.")
            else:
                # Run Optimization
                best_xi, team_stats = generate_best_xi(df_pool, formation)
                
                # Display Pitch
                _render_best_xi_pitch(best_xi)
                
        with c3:
            st.markdown("#### Team Rating")
            st.metric("ATTACK", f"{team_stats.get('ATT', 0):.0f}")
            st.metric("MIDFIELD", f"{team_stats.get('MID', 0):.0f}")
            st.metric("DEFENSE", f"{team_stats.get('DEF', 0):.0f}")
            
            st.markdown("#### Selected Players")
            st.dataframe(pd.DataFrame(best_xi)[['role', 'name', 'val']], hide_index=True)

def _render_best_xi_pitch(best_xi):
    """Draws the Best XI on a Plotly pitch."""
    fig = go.Figure()

    # Pitch Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="#334155", width=2), fillcolor="#0f172a")
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="#334155", width=2))
    fig.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="#334155", width=2))
    
    # Player Markers
    for p in best_xi:
        # Check if coordinates exist
        if 'x' in p and 'y' in p:
            # Color based on score
            val = p.get('val', 50)
            color = "#00f3ff" if val > 80 else ("#facc15" if val > 65 else "#f43f5e")
            
            fig.add_trace(go.Scatter(
                x=[p['x']], y=[p['y']],
                mode='markers+text',
                marker=dict(size=20, color=color, line=dict(width=2, color='white')),
                text=[f"<b>{p['role']}</b><br>{p['name']}"],
                textposition="bottom center",
                hoverinfo='text',
                showlegend=False
            ))

    fig.update_layout(
        xaxis=dict(range=[-5, 125], showgrid=False, visible=False),
        yaxis=dict(range=[-5, 85], showgrid=False, visible=False),
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)