import streamlit as st
from utils.visualisations import draw_cyber_pitch
from utils.state import load_league_data
from matplotlib.patches import ConnectionPatch

def render_sniper_tab(active_league_name, active_league_df, league_file_map, root_dir):
    st.markdown("### 🥅 SNIPER MAP")
    
    f1, f2, f3, f4 = st.columns(4)
    
    # A. League
    with f1:
        leagues = sorted(list(league_file_map.keys()))
        # Safe index lookup
        idx = leagues.index(active_league_name) if active_league_name in leagues else 0
        
        # FIX: Added key='sniper_league' to make this dropdown unique
        sel_league = st.selectbox("LEAGUE", leagues, index=idx, key="sniper_league_select")

    # Load data if the user picked a different league in this tab
    if sel_league == active_league_name:
        df = active_league_df
    else:
        df, _ = load_league_data(root_dir, sel_league, league_file_map)

    # B. Team
    with f2:
        col = 'team_name' if 'team_name' in df.columns else 'team'
        teams = sorted(df[col].dropna().astype(str).unique()) if col in df.columns else []
        
        # FIX: Added unique key
        sel_team = st.selectbox("TEAM", ["All"] + teams, key="sniper_team_select")

    # C. Player
    with f3:
        if sel_team != "All":
            df = df[df[col] == sel_team]
        
        players = sorted(df['player_name'].dropna().unique())
        # Smart default: Default to Messi if available, otherwise first player
        def_idx = players.index("Lionel Messi") if "Lionel Messi" in players else 0
        
        # FIX: Added unique key
        sel_player = st.selectbox("PLAYER", players, index=def_idx if players else 0, key="sniper_player_select")

    # D. Season
    with f4:
        p_data = df[df['player_name'] == sel_player]
        scol = 'season_name' if 'season_name' in p_data.columns else 'season'
        seasons = sorted(p_data[scol].dropna().astype(str).unique(), reverse=True)
        
        # FIX: Added unique key
        sel_season = st.selectbox("SEASON", seasons, key="sniper_season_select") if seasons else None

    # Render Map
    if sel_season:
        shots = p_data[p_data[scol].astype(str) == sel_season]
        _draw_map(shots)

def _draw_map(shots):
    if shots.empty:
        st.info("No data found.")
        return

    goals = shots[shots['is_goal'] == 1]
    misses = shots[shots['is_goal'] == 0]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("ATTEMPTS", len(shots))
    m2.metric("GOALS", len(goals))
    rate = (len(goals)/len(shots))*100 if len(shots) > 0 else 0
    m3.metric("CONVERSION", f"{rate:.1f}%")

    fig, ax = draw_cyber_pitch()
    
    for _, row in goals.iterrows():
        con = ConnectionPatch(xyA=(row['start_x'], row['start_y']), xyB=(120, 40), 
                              coordsA="data", coordsB="data", axesA=ax, axesB=ax, 
                              color='#00f3ff', alpha=0.4, lw=1.5)
        ax.add_artist(con)

    ax.scatter(goals.start_x, goals.start_y, c='#00f3ff', edgecolors='white', s=120, marker='h', zorder=10, label='Goal')
    ax.scatter(misses.start_x, misses.start_y, c='#f43f5e', alpha=0.3, s=40, zorder=5, label='Miss')
    ax.legend(facecolor='#050505', edgecolor='#334155', labelcolor='white', loc='lower center', ncol=2)
    st.pyplot(fig, use_container_width=True)