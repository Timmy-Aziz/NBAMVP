# Import Statements
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Reading CSVs
advanced = pd.read_csv('advanced.csv')
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
players = pd.read_csv('Players.csv')
team_abbrev = pd.read_csv('Team Abbrev.csv')
team_stats_per_hundred = pd.read_csv('Team Stats Per 100 Poss.csv')
team_stats_per_game = pd.read_csv('Team Stats Per Game.csv')
team_summaries = pd.read_csv('Team Summaries.csv')
team_totals = pd.read_csv('Team Totals.csv')


# Past MVPs
nba_mvp_winners = player_award_shares[(player_award_shares['award'] == 'nba mvp') & (player_award_shares['winner'] == True)]
print(nba_mvp_winners.player)

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

