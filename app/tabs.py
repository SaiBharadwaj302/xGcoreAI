import numpy as np
import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from dataclasses import dataclass
from typing import Any, Dict

from utils.physics import calculate_visible_angle, simulate_trajectory
from utils.simulations import generate_best_xi
from utils.visualisations import draw_cyber_pitch
from utils.helpers import (
    _add_engineered_features,
    _ensure_season_norm,
    _filter_by_season,
    _season_sort_key,
    _ensure_year_column,
)


@dataclass(frozen=True)
class TabContext:
    f_shots: pd.DataFrame
    normalized_stats: pd.DataFrame
    normalized_shots: pd.DataFrame
    shots_df: pd.DataFrame
    league_file_map: Dict[str, str]
    active_league: str
    file_key: str
    model: Any
    calibrator: Any


def render_simulation_tab(tab: st.delta_generator.DeltaGenerator, ctx: TabContext) -> None:
    with tab:
        c_ui, c_viz = st.columns([1, 2])
        with c_ui:
            st.markdown("#### POSITION VECTOR")
            x_in = st.slider("X (Depth)", 60, 120, 95)
            y_in = st.slider("Y (Width)", 0, 80, 40)
            power_in = st.slider("Shot Power (m/s)", 10, 40, 28)
            loft_in = st.slider("Loft Angle (deg)", 0, 45, 12)
            curl_in = st.slider("Curve / Spin", -10.0, 10.0, 0.0)

            dist = np.sqrt((120 - x_in) ** 2 + (40 - y_in) ** 2)
            vis_angle = calculate_visible_angle(x_in, y_in)
            phys_status, end_height, flight_time, traj_dict = simulate_trajectory(x_in, y_in, power_in, loft_in, curl_in)

            row = pd.DataFrame([
                {
                    'start_x': x_in,
                    'start_y': y_in,
                    'distance': dist,
                    'visible_angle': vis_angle,
                    'body_part_code': 3,
                    'technique_code': 4,
                }
            ])
            try:
                row = _add_engineered_features(row)
                try:
                    row_features = list(ctx.model.get_booster().feature_names) if ctx.model is not None and ctx.model.get_booster() is not None else None
                except Exception:
                    row_features = None
                if not row_features:
                    row_features = [
                        'start_x', 'start_y', 'distance', 'visible_angle', 'body_part_code', 'technique_code',
                        'angle_sin', 'angle_cos', 'dist_to_goal_center', 'is_header', 'start_x_norm', 'start_y_norm', 'player_last5_conv',
                    ]
                Xrow = pd.DataFrame()
                for raw_feat in row_features:
                    Xrow[raw_feat] = row[raw_feat] if raw_feat in row.columns else 0
                Xrow = Xrow.fillna(0)
                if ctx.calibrator is not None:
                    prob = float(ctx.calibrator.predict_proba(Xrow)[:, 1][0])
                else:
                    prob = float(ctx.model.predict_proba(Xrow)[:, 1][0])
            except Exception:
                prob = 0.0

            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.metric("ML PROBABILITY", f"{prob:.1%}", delta_color="normal")
            with c_m2:
                color = "#00f3ff" if phys_status == "GOAL" else "#ff0055"
                st.markdown(
                    f"<div style='background:{color}33; padding:5px; border-left:3px solid {color}; border-radius:4px;'>"
                    f"<strong style='color:white'>{phys_status}</strong></div>",
                    unsafe_allow_html=True,
                )

            st.caption(f"Dist: {dist:.1f}y | H: {end_height:.2f}m | T: {flight_time:.2f}s")

        with c_viz:
            fig, ax = draw_cyber_pitch()
            ax.scatter(x_in, y_in, s=200, c='#ff0055', edgecolors='white', zorder=10, marker='+')
            cone = plt.Polygon([[x_in, y_in], [120, 36], [120, 44]], color='#00f3ff', alpha=0.15)
            ax.add_patch(cone)
            if phys_status == "GOAL":
                con = ConnectionPatch(
                    xyA=(x_in, y_in),
                    xyB=(120, 40),
                    coordsA="data",
                    coordsB="data",
                    axesA=ax,
                    axesB=ax,
                    color='#00f3ff',
                    ls='--',
                    alpha=0.5,
                )
                ax.add_artist(con)
            st.pyplot(fig, use_container_width=True)

        st.markdown("#### MATCHING SIGNATURES")
        f_shots = ctx.f_shots.copy()
        f_shots['d'] = np.sqrt((f_shots['start_x'] - x_in) ** 2 + (f_shots['start_y'] - y_in) ** 2)
        nearby = f_shots[f_shots['d'] <= 5]
        if not nearby.empty:
            if 'efficiency' not in nearby.columns:
                nearby['efficiency'] = nearby['is_goal'] - nearby['model_xg']
            st.dataframe(
                nearby.groupby(['player_name', 'league'])
                .agg({'is_goal': 'sum', 'efficiency': 'sum'})
                .reset_index()
                .sort_values('efficiency', ascending=False)
                .head(5),
                use_container_width=True,
            )
        else:
            st.info("NO MATCHES")

        try:
            overperf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', f'overperformers_{ctx.file_key}.csv')
        except Exception:
            overperf_path = None
        if overperf_path and os.path.exists(overperf_path):
            op_df = pd.read_csv(overperf_path)
            if 'overperf' in op_df.columns:
                op_df = op_df.sort_values('overperf', ascending=False)
            st.markdown('#### Top Overperformers (Actual Goals − xG)')
            st.dataframe(
                op_df[['player_name', 'actual_goals', 'xg', 'attempts', 'overperf']].head(10),
                use_container_width=True,
            )
        else:
            st.info('No overperformer report available for this league.')


def render_squad_genome_tab(tab: st.delta_generator.DeltaGenerator, ctx: TabContext, formations: list[str]) -> None:
    with tab:
        c_opt, c_view = st.columns([1, 3])
        with c_opt:
            fmt = st.selectbox("SYSTEM", formations)
            seasons_set = set()
            try:
                if not ctx.normalized_stats.empty:
                    seasons_set.update(ctx.normalized_stats['season_norm'].dropna().astype(str).unique())
            except Exception:
                pass
            try:
                if not ctx.normalized_shots.empty:
                    seasons_set.update(ctx.normalized_shots['season_norm'].dropna().astype(str).unique())
            except Exception:
                pass

            if not seasons_set:
                seasons = ['All Time']
            else:
                seasons = list(seasons_set)
                if 'All Time' not in seasons:
                    seasons.append('All Time')
            seasons = sorted(seasons, key=_season_sort_key, reverse=True)
            season_choice = st.selectbox('Season', seasons, index=0)
            show_all = st.checkbox('Show best lineup for every season', value=False)

        def render_lineup_table(df_stats: pd.DataFrame, title: str | None = None) -> None:
            if df_stats is None or df_stats.empty:
                st.info(f'No players available for {title or "selected season"}.')
                return
            best_xi, _ = generate_best_xi(df_stats, fmt)
            table = (
                pd.DataFrame(best_xi)[['role', 'name', 'val']]
                .rename(columns={'role': 'Role', 'name': 'Player', 'val': 'Rating'})
            )
            fig, ax = draw_cyber_pitch()
            color_map = {
                'att_score': '#ff6b6d',
                'mid_score': '#ffd166',
                'def_score': '#4cc9f0',
                'combined': '#94d2bd'
            }
            x_vals, y_vals, colors = [], [], []
            for player in best_xi:
                x_vals.append(player.get('x', 50))
                y_vals.append(player.get('y', 40))
                typ = player.get('type', 'combined')
                colors.append(color_map.get(typ, '#ff0055'))
            scatter = ax.scatter(x_vals, y_vals, c=colors, edgecolors='#00f3ff', s=400, zorder=10)
            for player, x, y in zip(best_xi, x_vals, y_vals):
                name = player.get('name', '')
                ax.text(x, y + 3, name, color='white', ha='center', va='top', fontsize=8, weight='bold')
            legend_handles = [
                Line2D([], [], marker='o', color='w', markerfacecolor=color_map['att_score'], markersize=10, label='Attackers'),
                Line2D([], [], marker='o', color='w', markerfacecolor=color_map['mid_score'], markersize=10, label='Midfielders'),
                Line2D([], [], marker='o', color='w', markerfacecolor=color_map['def_score'], markersize=10, label='Defenders'),
                Line2D([], [], marker='o', color='w', markerfacecolor=color_map['combined'], markersize=10, label='Fullback'),
            ]
            ax.legend(handles=legend_handles, loc='upper right', title='Role Group')
            st.markdown(f'#### Pitch View {"- " + title if title else ""}')
            st.pyplot(fig, use_container_width=True)
            st.markdown(f'#### Selected Lineup {"- " + title if title else ""}')
            st.dataframe(table, use_container_width=True)

        def _build_season_stats_from_shots(season_choice: str) -> pd.DataFrame:
            try:
                ss = _filter_by_season(ctx.normalized_shots, season_choice)
                if ss.empty:
                    return pd.DataFrame()
                agg = ss.groupby('player_name').agg({'is_goal': 'sum', 'start_x': 'count'}).rename(columns={'start_x': 'shots'}).reset_index()
            except Exception:
                return pd.DataFrame()
            rows = []
            for _, r in agg.iterrows():
                pname = r['player_name']
                player_rows = ss[ss['player_name'] == pname]
                def mode_col(df: pd.DataFrame, col: str) -> str:
                    if col in df.columns and not df[col].dropna().empty:
                        return str(df[col].mode().iloc[0])
                    return ''
                primary_pos = mode_col(player_rows, 'primary_pos')
                team = (
                    mode_col(player_rows, 'team')
                    if 'team' in player_rows.columns
                    else mode_col(player_rows, 'team_name')
                    if 'team_name' in player_rows.columns
                    else ''
                )
                goals = int(r['is_goal']) if 'is_goal' in r else int(player_rows['is_goal'].sum()) if 'is_goal' in player_rows.columns else 0
                shots_count = int(r['shots'])
                att_score = int(shots_count * 5 + goals * 50)
                rows.append({
                    'player_name': pname,
                    'league': ctx.active_league,
                    'team_name': team,
                    'primary_pos': primary_pos,
                    'goals': goals,
                    'shots': shots_count,
                    'passes': 0,
                    'tackles': 0,
                    'interceptions': 0,
                    'clearances': 0,
                    'blocks': 0,
                    'def_score': 0,
                    'mid_score': 0,
                    'att_score': att_score,
                    'extracted_year': 0,
                    'season': season_choice,
                })
            return pd.DataFrame(rows)

        with c_view:
            if show_all:
                for s in seasons:
                    df_s = _filter_by_season(ctx.normalized_stats, s)
                    if df_s.empty:
                        df_s = _build_season_stats_from_shots(s)
                    with st.expander(f"{s} - Best XI", expanded=False):
                        render_lineup_table(df_s, title=s)
            else:
                sel_df = _filter_by_season(ctx.normalized_stats, season_choice)
                if sel_df.empty:
                    sel_df = _build_season_stats_from_shots(season_choice)
                render_lineup_table(sel_df, title=season_choice)
