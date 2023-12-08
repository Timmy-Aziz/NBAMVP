# Import Statements
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Reading CSVs
advanced = pd.read_csv('Advanced.csv')
all_star_selections = pd.read_csv('All-Star Selections.csv')
end_of_season_teams_voting = pd.read_csv('End of Season Teams (Voting).csv')
end_of_season_teams = pd.read_csv('End of Season Teams.csv')
opponent_stats_per_hundred = pd.read_csv('Opponent Stats Per 100 Poss.csv')
opponent_stats_per_game = pd.read_csv('Opponent Stats Per Game.csv')
opponent_totals = pd.read_csv('Opponent Totals.csv')
per_thirty_six = pd.read_csv('Per 100 Poss.csv')
player_award_shares = pd.read_csv('Player Award Shares.csv')
player_career_info = pd.read_csv('Player Career Info.csv')
player_per_game = pd.read_csv('Player Per Game.csv')
player_play_by_play = pd.read_csv('Player Play By Play.csv')
player_season_info = pd.read_csv('Player Season Info.csv')
player_shooting = pd.read_csv('Player Shooting.csv')
player_totals= pd.read_csv('Player Totals.csv')
player_data = pd.read_csv('player_data.csv')
team_abbrev = pd.read_csv('Team Abbrev.csv')
team_stats_per_hundred = pd.read_csv('Team Stats Per 100 Poss.csv')
team_stats_per_game = pd.read_csv('Team Stats Per Game.csv')
team_summaries = pd.read_csv('Team Summaries.csv')
team_totals = pd.read_csv('Team Totals.csv')


# Past MVPs
nba_mvp_winners = player_award_shares[(player_award_shares['award'] == 'nba mvp') & (player_award_shares['winner'] == True)]

'''
Most important statistics for predicting NBA MVP: 
Include team record
1. PER
2. WS
3. Team Record
4. PPG
5. APG
6. RPG
7. FG%
8. 3P%
9. FT%
10. SPG / BPG
11. Usage Rate
12. Plus/Minus
'''


# Merging Player-Specific CSVs into a DataFrame
merged_player_df = pd.merge(advanced, all_star_selections, on=['player', 'season'], how='inner', suffixes=('_adv', '_all_star'))
merged_player_df = pd.merge(merged_player_df, end_of_season_teams_voting, on=['player', 'season'], how='inner', suffixes=('_merged', '_teams_voting'))
merged_player_df = pd.merge(merged_player_df, end_of_season_teams, on=['player', 'season'], how='inner', suffixes=('_merged', '_teams'))
merged_player_df = pd.merge(merged_player_df, per_thirty_six, on=['player', 'season'], how='inner', suffixes=('_merged', '_per_thirty_six'))
merged_player_df = pd.merge(merged_player_df, player_award_shares, on=['player', 'season'], how='inner', suffixes=('_merged', '_award_shares'))
merged_player_df = pd.merge(merged_player_df, player_career_info, on=['player', 'season'], how='inner', suffixes=('_merged', '_career_info'))
merged_player_df = pd.merge(merged_player_df, player_per_game, on=['player', 'season'], how='inner', suffixes=('_merged', '_per_game'))
merged_player_df = pd.merge(merged_player_df, player_play_by_play, on=['player', 'season'], how='inner', suffixes=('_merged', '_play_by_play'))
merged_player_df = pd.merge(merged_player_df, player_season_info, on=['player', 'season'], how='inner', suffixes=('_merged', '_season_info'))
merged_player_df = pd.merge(merged_player_df, player_shooting, on=['player', 'season'], how='inner', suffixes=('_merged', '_shooting'))
merged_player_df = pd.merge(merged_player_df, player_totals, on=['player', 'season'], how='inner', suffixes=('_merged', '_totals'))

# Manually rename overlapping columns
merged_player_df.rename(columns={'player_id_merged': 'player_id', 'age_merged': 'age', 'seas_id_merged': 'seas_id', 'tm_merged': 'tm'}, inplace=True)

# Assuming merged_df is your merged DataFrame
columns_to_remove = merged_player_df.filter(like='_lg').columns
merged_player_df = merged_player_df.drop(columns=columns_to_remove)

# Merging Team Related CSVs
merged_team_df = pd.merge(team_abbrev, team_stats_per_hundred, on=['team', 'season'], how='inner', suffixes=('_abbrev', '_stats_per_hundred'))
merged_team_df = pd.merge(merged_team_df, team_stats_per_game, on=['team', 'season'], how='inner', suffixes=('_merged', '_stats_per_game'))
merged_team_df = pd.merge(merged_team_df, team_summaries, on=['team', 'season'], how='inner', suffixes=('_merged', '_summaries'))
merged_team_df = pd.merge(merged_team_df, team_totals, on=['team', 'season'], how='inner', suffixes=('_merged', '_totals'))
merged_team_df = pd.merge(merged_team_df, opponent_stats_per_hundred, on=['team', 'season'], how='inner', suffixes=('_merged', '_opp_stats_per_hundred'))
merged_team_df = pd.merge(merged_team_df, opponent_stats_per_game, on=['team', 'season'], how='inner', suffixes=('_merged', '_opp_stats_per_game'))
merged_team_df = pd.merge(merged_team_df, opponent_totals, on=['team', 'season'], how='inner', suffixes=('_merged', '_opp_totals'))




print(merged_player_df.head())
