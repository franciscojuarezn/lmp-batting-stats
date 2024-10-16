import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
players_2023_df = pd.read_csv('players_data_2023.csv')
players_2024_df = pd.read_csv('players_data_2024.csv')
standard_stats_df = pd.read_csv('df_standard_stats.csv')
advanced_stats_df = pd.read_csv('df_advanced_stats.csv')

# Load hit trajectory data
hit_trajectory_2023 = pd.read_csv('hit_trajectory_lmp_2023.csv')
hit_trajectory_2024 = pd.read_csv('hit_trajectory_lmp_2024.csv')

# Combine the hit data
hit_trajectory_df = pd.concat([hit_trajectory_2023, hit_trajectory_2024], ignore_index=True)

# Combine the players data for 2023 and 2024
players_df = pd.concat([players_2023_df, players_2024_df])

# Connect to the SQLite database to get the headshot URLs
conn = sqlite3.connect('player_headshots_lmp.db')

# Load player IDs and headshot URLs from the database
headshots_df = pd.read_sql_query("SELECT playerId, headshot_url FROM player_headshots_lmp", conn)

# Close the database connection
conn.close()

# Ensure 'playerId' and 'id' are of the same type
headshots_df['playerId'] = headshots_df['playerId'].astype(int)

# Merge headshot URLs with players data
players_df = pd.merge(players_df, headshots_df, left_on='id', right_on='playerId', how='left')

# **NEW CODE: Convert 'player_id' in stats DataFrames to string**
# Ensure 'player_id' in stats DataFrames is of type integer
standard_stats_df['player_id'] = standard_stats_df['player_id'].astype(int)
advanced_stats_df['player_id'] = advanced_stats_df['player_id'].astype(int)


# Streamlit app setup
st.set_page_config(page_title="LMP Batting Stats", layout="wide")

# Title
st.title("LMP Batting Stats")
st.divider()
# Filter to exclude players whose position is 'P' (Pitcher)
non_pitchers_df = players_df[players_df['POS'] != 'P']
# Drop duplicates based on 'id' to ensure each player appears only once
non_pitchers_df_unique = non_pitchers_df.drop_duplicates(subset=['id'])
# Dropdown for selecting a batter (non-pitchers only)
batter = st.selectbox("Select a Batter", non_pitchers_df_unique['fullName'])

# Get player data for the selected batter
player_data = non_pitchers_df[non_pitchers_df['fullName'] == batter].iloc[0]

# Display player information in three columns
st.subheader("Player Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**Full Name:** {player_data['fullName']}")
    st.write(f"**Position:** {player_data['POS']}")
    st.write(f"**B/T:** {player_data['B/T']}")

with col2:
    st.write(f"**Birthdate:** {player_data['birthDate']}")
    st.write(f"**Birthplace:** {player_data['Birthplace']}")

with col3:
    # Check if headshot_url exists and display the image
    if pd.notna(player_data['headshot_url']):
        st.image(player_data['headshot_url'], width=150)
    else:
        st.write("No headshot available.")

# st.divider()

# Filter stats for selected player (can have multiple rows if player has stats for multiple seasons/teams)
standard_stats = standard_stats_df[standard_stats_df['player_id'] == player_data['id']]
advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]

# Convert 'season' column to string in both dataframes
standard_stats['season'] = standard_stats['season'].astype(str)
advanced_stats['season'] = advanced_stats['season'].astype(str)

# Select specific columns and order for standard stats
standard_columns = ['season', 'team', 'Name', 'POS', 'G', 'PA', 'AB', 'H', 'RBI', '2B', '3B', 'HR', 'TB', 'HBP', 'SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS']
standard_stats_filtered = standard_stats[standard_columns].copy()

# Convert 'season' to integer for proper sorting
standard_stats_filtered['season'] = standard_stats_filtered['season'].astype(int)

# Sort by 'season' in descending order
standard_stats_filtered = standard_stats_filtered.sort_values('season', ascending=False)

# Format numeric columns in standard stats to three decimal places
standard_stats_formatted = standard_stats_filtered.style.format({
    'AVG': '{:.3f}',
    'OBP': '{:.3f}',
    'SLG': '{:.3f}',
    'OPS': '{:.3f}'
})

# Standard stats table
st.subheader("Standard Stats", divider='gray')
st.dataframe(standard_stats_formatted, hide_index=True, use_container_width=True)

# --- Advanced Stats ---

# Select specific columns and order for advanced stats
advanced_columns = ['season', 'team', 'BABIP', 'K%', 'BB%', 'HR/PA', 'BB/K', 'HR/FB%', 'SwStr%', 'Whiff%', 'FB%', 'GB%', 'LD%', 'PopUp%']
advanced_stats_filtered = advanced_stats[advanced_columns].copy()

# Convert 'season' to integer for proper sorting
advanced_stats_filtered['season'] = advanced_stats_filtered['season'].astype(int)

# Sort by 'season' in descending order
advanced_stats_filtered = advanced_stats_filtered.sort_values('season', ascending=False)

# Format numeric columns in advanced stats
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
})

# Advanced stats table
st.subheader("Advanced Stats & Batted Ball", divider='gray')
st.dataframe(advanced_stats_formatted, hide_index=True, use_container_width=True)
