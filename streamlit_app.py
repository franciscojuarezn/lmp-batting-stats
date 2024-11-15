import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob
import os
import plotly.express as px
import plotly.graph_objects as go

# Streamlit page config
st.set_page_config(page_title="LMP Batting Stats", layout="wide")

# Load data with caching to improve performance
@st.cache_data
def load_players_data():
    players_data_files = glob.glob(os.path.join('stats_data', 'players_data_*.csv'))
    players_df_list = [pd.read_csv(file) for file in players_data_files]
    return pd.concat(players_df_list, ignore_index=True)

@st.cache_data
def load_standard_stats():
    standard_stats_files = glob.glob(os.path.join('stats_data', 'df_standard_stats_*.csv'))
    standard_stats_df_list = [pd.read_csv(file) for file in standard_stats_files]
    return pd.concat(standard_stats_df_list, ignore_index=True)

@st.cache_data
def load_advanced_stats():
    advanced_stats_files = glob.glob(os.path.join('stats_data', 'df_advanced_stats_*.csv'))
    advanced_stats_df_list = [pd.read_csv(file) for file in advanced_stats_files]
    return pd.concat(advanced_stats_df_list, ignore_index=True)

@st.cache_data
def load_hit_trajectory():
    hit_trajectory_files = glob.glob(os.path.join('stats_data', 'hit_trajectory_lmp_*.csv'))
    hit_trajectory_df_list = [pd.read_csv(file) for file in hit_trajectory_files]
    return pd.concat(hit_trajectory_df_list, ignore_index=True)

@st.cache_data
def load_stadium_data():
    return pd.read_csv(os.path.join('stats_data', 'stadium.csv'))

@st.cache_data
def load_headshots():
    conn = sqlite3.connect(os.path.join('stats_data', 'player_headshots_lmp.db'))
    headshots_df = pd.read_sql_query("SELECT playerId, headshot_url FROM player_headshots_lmp", conn)
    conn.close()
    return headshots_df

# Load team data
team_data_std = pd.read_csv('team_data_std.csv')
team_data_adv = pd.read_csv('team_data_adv.csv')

# --- Calculate League Averages ---
# Aggregate data for standard stats
total_hits = team_data_std['H'].sum()
total_walks = team_data_std['BB'].sum()
total_hbp = team_data_std['HBP'].sum()
total_ab = team_data_std['AB'].sum()
total_sf = team_data_std['SF'].sum()
total_pa = team_data_std['PA'].sum()
total_hr = team_data_std['HR'].sum()
total_2b = team_data_std['2B'].sum()
total_3b = team_data_std['3B'].sum()
total_k = team_data_std['K'].sum()

# Standard Stats Calculations
league_avg = round(total_hits / total_ab, 3)
league_obp = round((total_hits + total_walks + total_hbp) / (total_ab + total_walks + total_hbp + total_sf), 3)
league_slg = round((total_hits - total_2b - total_3b - total_hr + (total_2b * 2) + (total_3b * 3) + (total_hr * 4)) / total_ab, 3)
league_ops = round(league_obp + league_slg, 3)
league_babip = round((total_hits - total_hr) / (total_ab - total_k - total_hr + total_sf), 3)

# Aggregate data for advanced stats
total_swing_misses = team_data_adv['swingAndMisses'].sum()
total_pitches = team_data_adv['numP'].sum()
total_swings = team_data_adv['totalSwings'].sum()
total_fb = team_data_adv['FO'].sum() + team_data_adv['flyHits'].sum()
total_bip = team_data_adv['BIP'].sum()
total_gb = team_data_adv['GO'].sum() + team_data_adv['groundHits'].sum()
total_ld = team_data_adv['lineOuts'].sum() + team_data_adv['lineHits'].sum()
total_pop_up = team_data_adv['popOuts'].sum() + team_data_adv['popHits'].sum()
total_hr_fb = team_data_adv['HR'].sum()

# Advanced Stats Calculations
league_k_percent = round((total_k / total_pa) * 100, 1)
league_bb_percent = round((total_walks / total_pa) * 100, 1)
league_bb_k = round(total_walks / total_k, 3)
league_swstr_percent = round((total_swing_misses / total_pitches) * 100, 1)
league_whiff_percent = round((total_swing_misses / total_swings) * 100, 1)
league_fb_percent = round((total_fb / total_bip) * 100, 1)
league_gb_percent = round((total_gb / total_bip) * 100, 1)
league_ld_percent = round((total_ld / total_bip) * 100, 1)
league_popup_percent = round((total_pop_up / total_bip) * 100, 1)
league_hr_fb_percent = round((total_hr_fb / total_fb) * 100, 1)

# Combined League Averages DataFrame
league_averages = pd.DataFrame({
    'AVG': [league_avg],
    'OBP': [league_obp],
    'SLG': [league_slg],
    'OPS': [league_ops],
    'BABIP': [league_babip],
    'K%': [league_k_percent],
    'BB%': [league_bb_percent],
    'BB/K': [league_bb_k],
    'SwStr%': [league_swstr_percent],
    'Whiff%': [league_whiff_percent],
    'FB%': [league_fb_percent],
    'GB%': [league_gb_percent],
    'LD%': [league_ld_percent],
    'PopUp%': [league_popup_percent],
    'HR/FB%': [league_hr_fb_percent]
})


# Toggle for selecting Players or Teams view
view_selection = st.radio("", ["Players", "Teams", "Leaderboard"], index=0, horizontal=True)
# st.divider()
if view_selection == "Players":
        
    # Load datasets
    players_df = load_players_data()
    standard_stats_df = load_standard_stats()
    advanced_stats_df = load_advanced_stats()
    hit_trajectory_df = load_hit_trajectory()
    team_data = load_stadium_data()
    headshots_df = load_headshots()
    batters_df = pd.read_csv('batters_df.csv')

    # Ensure 'playerId' and 'id' are of the same type
    headshots_df['playerId'] = headshots_df['playerId'].astype(int)
    players_df = pd.merge(players_df, headshots_df, left_on='id', right_on='playerId', how='left')

    # Ensure 'player_id' in stats DataFrames is of type integer
    standard_stats_df['player_id'] = standard_stats_df['player_id'].astype(int)
    advanced_stats_df['player_id'] = advanced_stats_df['player_id'].astype(int)

    # st.set_page_config(page_title="LMP Batting Stats", layout="wide")
    logo_and_title = """
        <div style="display: flex; align-items: center;">
            <img src="https://www.lmp.mx/assets/img/header/logo_80_aniversary.webp" alt="LMP Logo" width="50" height="50">
            <h1 style="margin-left: 10px;">LMP Batting Stats</h1>
        </div>
    """

    # Display the logo and title using st.markdown
    st.markdown(logo_and_title, unsafe_allow_html=True)
    st.divider()

    # Filter to exclude players whose position is 'P' (Pitcher)
    non_pitchers_df = players_df[players_df['POS'] != 'P']
    non_pitchers_df_unique = non_pitchers_df.drop_duplicates(subset=['id'])
    non_pitchers_df_unique = non_pitchers_df_unique.sort_values('fullName')

    default_player = 'Esteban Quiroz'
    default_index = next((i for i, name in enumerate(non_pitchers_df_unique['fullName']) if name == default_player), 0)

    col1, col2 = st.columns([1, 3])  # col1 will be smaller, and col2 will be wider

    # Place the select box inside the smaller column
    with col1:
        selected_batter = st.selectbox("Select a Batter", non_pitchers_df_unique['fullName'], index=default_index)

    # Get player data for the selected batter
    player_data = non_pitchers_df[non_pitchers_df['fullName'] == selected_batter].iloc[0]

    # Filter player stats for the 2024 season
    def filter_2024_season_data(player_name, df):
        player_ops_data = df[df['FullName'] == player_name]
        
        # Filter for the 2024 season
        player_ops_data['Date'] = pd.to_datetime(player_ops_data['Date'])
        player_ops_data['season'] = player_ops_data['Date'].dt.year
        player_2024_ops_data = player_ops_data[player_ops_data['season'] == 2024]
        
        return player_2024_ops_data

    # Adjusted plotting function to handle missing 2024 season data
    def plot_player_ops_styled_2024(player_name, player_ops_data, league_avg_ops):
        if player_ops_data.empty:
            st.write("")
            return
        
        # Convert 'Date' column to datetime for proper sorting and plotting
        player_ops_data['Date'] = pd.to_datetime(player_ops_data['Date'])
        
        # Sort by date for accurate plotting
        player_ops_data = player_ops_data.sort_values('Date')
        
        # Plot the player's OPS over time with custom styles
        plt.figure(figsize=(8, 4))
        plt.gca().set_facecolor('beige')   # Set plot background color
        plt.gcf().set_facecolor('beige')   # Set figure background color

        plt.plot(
            player_ops_data['Date'], player_ops_data['OPS'], 
            color='blue',         # Line color
            linestyle='-',        # Solid line
            marker='o',           # Circle markers
            linewidth=2,          # Line width
            label='_nolegend_'    # Exclude from legend
        )
        
        # Add a horizontal line for the league average OPS with custom style
        plt.axhline(
            y=league_avg_ops, 
            color='red',          # Red color
            linestyle='--',       # Dashed line
            linewidth=2,          # Line width
            label=f'League Avg OPS: {league_avg_ops:.3f}'
        )
        
        # Add titles and labels
        plt.title(f'Rolling OPS {player_name}')
        # plt.xlabel('Date')
        # plt.ylabel('OPS')
        plt.legend()
        plt.grid(False)
        
        # Format the x-axis to show only the month and day
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        
        # Format the y-axis to display 3 decimal places
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Show the plot
        plt.tight_layout()
        st.pyplot(plt)

    # Display player information in three columns
    st.subheader("Player Information")
    col1, col2, col3, col4 = st.columns([.5, .5, .5, .8])

    with col1:
        st.write(f"**Full Name:** {player_data['fullName']}")
        st.write(f"**Position:** {player_data['POS']}")
        st.write(f"**B/T:** {player_data['B/T']}")

    with col2:
        st.write(f"**Birthdate:** {player_data['birthDate']}")
        st.write(f"**Birthplace:** {player_data['Birthplace']}")

    # Filter for 2024 season data
    with col4:
        player_ops_data_2024 = filter_2024_season_data(selected_batter, batters_df)  # Filter data for the selected player in 2024
        league_avg_ops = 0.671  # Replace with the actual calculated league average OPS
        plot_player_ops_styled_2024(selected_batter, player_ops_data_2024, league_avg_ops)

    with col3:
        # Check if headshot_url exists and display the image
        if pd.notna(player_data['headshot_url']):
            st.image(player_data['headshot_url'], width=150)
        else:
            st.image(os.path.join('stats_data', 'current.jpg'), width=150)
    # --- Standard Stats ---

    # --- Standard Stats ---
    # Filter stats for the selected player (can have multiple rows if player has stats for multiple seasons/teams)
    standard_stats = standard_stats_df[standard_stats_df['player_id'] == player_data['id']]

    # Convert 'season' to integer for proper sorting
    standard_stats.loc[:, 'season'] = standard_stats['season'].astype(int)

    # Select specific columns and order for standard stats
    standard_columns = ['season', 'Name', 'team', 'POS', 'G', 'PA', 'AB', 'H', 'RBI', 'SB', '2B', '3B', 'HR', 'R','TB', 'HBP', 'GIDP', 'SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS']
    standard_stats_filtered = standard_stats[standard_columns].copy()

    # Sort by season in descending order and by team
    standard_stats_filtered = standard_stats_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

    # Apply formatting to highlight rows where 'team' is '2 Teams'
    def highlight_two_teams(row):
        return ['background-color: #2E2E2E; color:white' if row['team'] == '2 Teams' else '' for _ in row]

    # Format numeric columns in standard stats to three decimal places
    standard_stats_formatted = standard_stats_filtered.style.format({
        'AVG': '{:.3f}',
        'OBP': '{:.3f}',
        'SLG': '{:.3f}',
        'OPS': '{:.3f}'
    }).apply(highlight_two_teams, axis=1)

    # Display Standard Stats table
    st.subheader("Standard Stats", divider='gray')
    st.dataframe(standard_stats_formatted, hide_index=True, use_container_width=True)

    # --- Advanced Stats ---
    # Filter stats for the selected player (can have multiple rows if player has stats for multiple seasons/teams)
    advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]

    # Convert 'season' to integer for proper sorting
    advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)

    # Select specific columns and order for advanced stats
    advanced_columns = ['season', 'Name', 'team', 'BABIP', 'K%', 'BB%', 'HR/PA', 'BB/K', 'HR/FB%', 'SwStr%', 'Whiff%', 'FB%', 'GB%', 'LD%', 'PopUp%']
    advanced_stats_filtered = advanced_stats[advanced_columns].copy()

    # Sort by season in descending order and by team
    advanced_stats_filtered = advanced_stats_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

    # Apply formatting to highlight rows where 'team' is '2 Teams'
    def highlight_two_teams(row):
        return ['background-color: #2E2E2E; color:white' if row['team'] == '2 Teams' else '' for _ in row]

    # Format numeric columns in advanced stats to the appropriate decimal places
    advanced_stats_formatted = advanced_stats_filtered.style.format({
        'BABIP': '{:.3f}',
        'K%': '{:.1f}',
        'BB%': '{:.1f}',
        'HR/PA': '{:.3f}',
        'BB/K': '{:.3f}',
        'HR/FB%': '{:.1f}',
        'SwStr%': '{:.1f}',
        'Whiff%': '{:.1f}',
        'FB%': '{:.1f}',
        'GB%': '{:.1f}',
        'LD%': '{:.1f}',
        'PopUp%': '{:.1f}'
    }).apply(highlight_two_teams, axis=1)

    # Display Advanced Stats table
    st.subheader("Advanced Stats & Batted Ball", divider='gray')
    st.dataframe(advanced_stats_formatted, hide_index=True, use_container_width=True)

    # Batted Ball Distribution Section
    st.subheader(f"Batted Ball Distribution for {selected_batter}")

    # Create season column from date in hit_trajectory_df
    hit_trajectory_df['date'] = pd.to_datetime(hit_trajectory_df['date'])
    hit_trajectory_df['season'] = hit_trajectory_df['date'].dt.year

    # Get available seasons
    available_seasons = sorted(hit_trajectory_df['season'].unique(), reverse=True)

    col1, col2 =st.columns([1,3])
    with col1:
        selected_season = st.selectbox("Select Season", available_seasons)

    # Filter the hit trajectory data based on the selected season and batter
    filtered_hit_trajectory = hit_trajectory_df[
        (hit_trajectory_df['season'] == selected_season) &
        (hit_trajectory_df['batter_name'] == selected_batter)
    ]

    # Event types
    event_types = ['single', 'double', 'triple', 'home_run', 'out']
    col1, col2 =st.columns([1,2])
    with col1:
        selected_events = st.multiselect("Select Event Types", event_types, default=event_types)

    # All 'outs'
    out_events = ['field_out', 'double_play', 'force_out', 'sac_bunt', 'grounded_into_double_play', 'sac_fly', 'fielders_choice_out', 'field_error', 'sac_fly_double_play']
    filtered_hit_trajectory.loc[:, 'event'] = filtered_hit_trajectory['event'].apply(lambda x: 'out' if x in out_events else x)


    # Define splits for LHP and RHP
    vs_LHP = filtered_hit_trajectory[filtered_hit_trajectory['split_batter'] == 'vs_LHP']
    vs_RHP = filtered_hit_trajectory[filtered_hit_trajectory['split_batter'] == 'vs_RHP']

    # Filter the data for the selected events
    vs_LHP = vs_LHP[vs_LHP['event'].isin(selected_events)]
    vs_RHP = vs_RHP[vs_RHP['event'].isin(selected_events)]

    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)

    # Function to plot the field and hit outcomes
    def plot_field_and_hits(team_data, hit_data, selected_column, palette, plot_title):
        plt.figure(figsize=(8,8))
        y_offset = 275
        excluded_segments = ['outfield_inner']
        
        # Plot the field layout
        for segment_name in team_data['segment'].unique():
            if segment_name not in excluded_segments:
                segment_data = team_data[team_data['segment'] == segment_name]
                plt.plot(segment_data['x'], -segment_data['y'] + y_offset, linewidth=4, zorder=1, color='forestgreen', alpha=0.5)

        # Adjust hit coordinates and plot the hits
        hit_data['adj_coordY'] = -hit_data['coordY'] + y_offset
        sns.scatterplot(data=hit_data, x='coordX', y='adj_coordY', hue=selected_column, palette=palette, edgecolor='black', s=100, alpha=0.7)

        plt.text(295, 23, 'Created by: @iamfrankjuarez', fontsize=8, color='grey', alpha=0.3, ha='right')

        plt.title(plot_title, fontsize=15)
        plt.xlabel("")
        plt.ylabel("")
        plt.legend(title=selected_column, title_fontsize='11', fontsize='11', borderpad=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-50, 300)
        plt.ylim(20, 300)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        st.pyplot(plt)

    # Plot for vs LHP
    with col1:
        if not vs_LHP.empty:
            plot_title = f"Batted Ball Outcomes vs LHP for {selected_batter}"
            plot_field_and_hits(team_data, vs_LHP, 'event', {
                'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
            }, plot_title)
        else:
            st.write("No data available for vs LHP.")

    # Plot for vs RHP
    with col2:
        if not vs_RHP.empty:
            plot_title = f"Batted Ball Outcomes vs RHP for {selected_batter}"
            plot_field_and_hits(team_data, vs_RHP, 'event', {
                'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
            }, plot_title)
        else:
            st.write("No data available for vs RHP.")

elif view_selection == "Teams":
    # Teams view for League Averages and Team Stats Dashboard
    # st.markdown("<h1>League & Teams</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>League & Teams</h1>", unsafe_allow_html=True)

    
    # Display Combined League Averages
    st.subheader("League Averages", divider='gray')
    st.dataframe(league_averages, use_container_width=True, hide_index=True)
    team_abbreviations = {
    'MXC': 'Aguilas de Mexicali',
    'JAL': 'Charros de Jalisco',
    'MOC': 'Caneros de los Mochis',
    'NAV': 'Mayos de Navojoa',
    'HER': 'Naranjeros de Hermosillo',
    'CUL': 'Tomateros de Culiacan',
    'MAZ': 'Venados de Mazatlan',
    'OBR': 'Yaquis de Obregon',
    'GSV': 'Algodoneros de Guasave',
    'MTY': 'Sultanes de Monterrey'
}

    # Replace team abbreviations with full team names in both DataFrames
    team_data_std['team'] = team_data_std['team'].replace(team_abbreviations)
    team_data_adv['team'] = team_data_adv['team'].replace(team_abbreviations)

    # Team-Level Stats Dashboard for Standard and Advanced Stats
    st.subheader("Team Standard Stats", divider='gray')
    standard_columns = ['team', 'PA', 'AB', 'H', 'RBI', 'SB', '2B', '3B', 'HR', 'R', 'TB', 'HBP', 'GIDP', 'SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS','1B', 'G', 'season', 'BABIP']
    team_standard_formatted = team_data_std[standard_columns].drop(columns=['season', 'G', '1B', 'BABIP']).style.format({
        'AVG': '{:.3f}',
        'OBP': '{:.3f}',
        'SLG': '{:.3f}',
        'OPS': '{:.3f}'
    })
    st.dataframe(team_standard_formatted, use_container_width=True, hide_index=True)

    st.subheader("Team Advanced Stats", divider='gray')
    advanced_columns = ['team', 'BABIP', 'K%', 'BB%', 'HR/PA', 'BB/K', 'SwStr%', 'Whiff%', 'FB%', 'GB%', 'LD%', 'PopUp%', 'HR/FB%']
    team_advanced_formatted = team_data_adv[advanced_columns].style.format({
        'BABIP': '{:.3f}',
        'K%': '{:.1f}',
        'BB%': '{:.1f}',
        'HR/PA': '{:.3f}',
        'BB/K': '{:.3f}',
        'HR/FB%': '{:.1f}',
        'SwStr%': '{:.1f}',
        'Whiff%': '{:.1f}',
        'FB%': '{:.1f}',
        'GB%': '{:.1f}',
        'LD%': '{:.1f}',
        'PopUp%': '{:.1f}'
    })
    st.dataframe(team_advanced_formatted, use_container_width=True, hide_index=True)

    import matplotlib.dates as mdates

    # Load the OPS data for teams
    ops_teams = pd.read_csv('ops_teams.csv').rename(columns={'date':'Date', 'team_name':'team','ops':'OPS'})

    # Define team colors
    team_color_map = {
        'MXC': '#19255b',
        'HER': '#fc8708',
        'OBR': '#134489',
        'NAV': '#fcef04',
        'CUL': '#701d45',
        'MAZ': '#ea0a2a',
        'JAL': '#b99823',
        'MTY': '#1f2344',
        'MOC': '#10964c',
        'GSV': '#85a8e2'
    }

    # Convert the 'Date' column to datetime format for proper plotting
    ops_teams['Date'] = pd.to_datetime(ops_teams['Date'])

    ops_teams = ops_teams.sort_values(by=['team', 'Date'])

    # Calculate league average OPS if needed
    lgAvgOPS = .671  # Replace with actual league average if provided

    # Create the plot using Plotly Express
    fig = px.line(ops_teams, x='Date', y='OPS', color='team', color_discrete_map=team_color_map,
                line_shape='spline', labels={'OPS': 'OPS', 'Date': 'Fecha'}, title="Rolling OPS por equipo")

    # Modify the hover mode to show all values on vertical line
    fig.update_traces(mode='lines+markers')  # Shows points on the line
    # fig.update_layout(hovermode='x unified')
    plot_bgcolor='lightgray',  # Set plot area background color to gray
    paper_bgcolor='lightgray',

    # Customize the layout for better readability and aspect ratio
    fig.update_layout(
        width=1700,
        height=800,
        plot_bgcolor='lightgray',  # Set plot area background color to gray
        paper_bgcolor='lightgray',  # Set the overall background color to gray
        font=dict(color='black'),
        legend=dict(
            title=dict(text='Equipos', font=dict(size=14, color='black')),
            font=dict(size=12, color='black'),
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.95
        ),
        title=dict(font=dict(size=18, color='black')),
        yaxis_title='OPS',
        yaxis_tickformat='.3f'
    )

    # Add the league average line
    fig.add_hline(y=lgAvgOPS, line_dash="dash",
                annotation_text="OPS promedio de la Liga",
                annotation_position="bottom right",
                annotation_font_size=12, 
                annotation_font_color = 'black')

    # Improve the y-axis readability and range
    fig.update_yaxes(tickformat='.3f', title_font=dict(size=14, color='black'), tickfont=dict(color='black'), range=[ops_teams['OPS'].min() - 0.05, ops_teams['OPS'].max() + 0.05])
    fig.update_xaxes(title_font=dict(size=14, color='black'), tickfont=dict(color='black'), showgrid=False, gridwidth=1, gridcolor='white')


    # Soften the gridlines
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

elif view_selection == "Leaderboard":
    @st.cache_data
    def load_standard_stats():
        standard_stats_files = glob.glob(os.path.join('stats_data', 'df_standard_stats_*.csv'))
        standard_stats_df_list = [pd.read_csv(file) for file in standard_stats_files]
        return pd.concat(standard_stats_df_list, ignore_index=True)

    @st.cache_data
    def load_advanced_stats():
        advanced_stats_files = glob.glob(os.path.join('stats_data', 'df_advanced_stats_*.csv'))
        advanced_stats_df_list = [pd.read_csv(file) for file in advanced_stats_files]
        return pd.concat(advanced_stats_df_list, ignore_index=True)

    # Load the data
    standard_stats_df = load_standard_stats()
    advanced_stats_df = load_advanced_stats()

    # Merge data on player ID and season, with suffixes to avoid conflicts
    merged_df = pd.merge(
        standard_stats_df,
        advanced_stats_df,
        on=['player_id', 'season'],
        how='outer',
        suffixes=('', '_adv')
    )

    # Separate the "2 Teams" entries
    two_teams_df = merged_df[merged_df['team'] == '2 Teams']
    individual_teams_df = merged_df[merged_df['team'] != '2 Teams']

    # Keep only unique players by dropping individual team entries if "2 Teams" exists
    merged_df = pd.concat([two_teams_df, individual_teams_df]).drop_duplicates(subset=['player_id', 'season'], keep='first')

    # Convert season to integer for consistent filtering
    merged_df['season'] = merged_df['season'].astype(int)

    # List of columns to display (excluding 'season' for the final display)
    display_columns = [
        'Name', 'team', 'G', 'PA', 'AB', 'H', 'RBI', 'SB', '2B', '3B', 'HR', 'R',
        'TB', 'HBP', 'GIDP','SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS', 'K%', 'BB%', 'BABIP'
    ]

    # Set up Streamlit layout
    st.title("LMP Batting Leaderboard")
    st.divider()

    # Layout: Select Year, Minimum AB, and Qualified Players toggle
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1.5, 1, 1])
    with col1:
        available_years = merged_df['season'].unique()
        selected_year = st.selectbox("Year", sorted(available_years, reverse=True))
    with col3:
        # Filter data based on the selected year
        filtered_df = merged_df[merged_df['season'] == selected_year]
        max_ab = filtered_df['AB'].max()
        min_ab = st.slider("Minimum AB", min_value=0, max_value=int(max_ab), value=0)

    # Calculate the qualified player threshold for the selected year
    max_games = filtered_df['G'].max()
    pa_threshold = int(max_games * 2.72)

    with col5:
        # Toggle for "All Players" and "Qualified Players" using radio buttons
        player_filter = st.radio("Player Filter", ["All Players", "Qualified Players"], horizontal=True)

    # Apply the PA filter if "Qualified Players" is selected
    if player_filter == "Qualified Players":
        filtered_df = filtered_df[filtered_df['PA'] >= pa_threshold]

    # Apply the AB filter and drop the season column
    filtered_df = filtered_df[filtered_df['AB'] >= min_ab]
    filtered_df = filtered_df[display_columns]  # Drop season for display

    # Sort by AVG in descending order
    filtered_df = filtered_df.sort_values(by='AVG', ascending=False)

    # Format columns for specific decimal places
    format_dict = {
        'AVG': '{:.3f}',
        'OBP': '{:.3f}',
        'SLG': '{:.3f}',
        'OPS': '{:.3f}',
        'BABIP': '{:.3f}',
        'K%': '{:.1f}',
        'BB%': '{:.1f}'
    }
    filtered_df = filtered_df.style.format(format_dict)

    # Display filtered leaderboard with 20 players per page
    st.dataframe(filtered_df, height=600, use_container_width=True, hide_index=True)
