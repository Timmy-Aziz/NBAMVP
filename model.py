# Import Statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


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


# Read CSV files
advanced = pd.read_csv("Advanced.csv")
all_star_selections = pd.read_csv("All-Star Selections.csv") 
end_of_season_teams_voting = pd.read_csv("End of Season Teams (Voting).csv")
end_of_season_teams = pd.read_csv("End of Season Teams.csv")
per_thirty_six = pd.read_csv("Per 36 Minutes.csv")
player_award_shares = pd.read_csv("Player Award Shares.csv")
player_per_game = pd.read_csv("Player Per Game.csv")
player_play_by_play = pd.read_csv("Player Play By Play.csv")
player_shooting = pd.read_csv("Player Shooting.csv")
player_totals = pd.read_csv("Player Totals.csv")


# Rename columns in each DataFrame to avoid conflicts
suffixes = {
    'advanced': '_adv',
    'all_star_selections': '_allstar',
    'end_of_season_teams_voting': '_voting',
    'end_of_season_teams': '_teams',
    'per_thirty_six': '_thirtysix',
    'player_award_shares': '_award',
    'player_per_game': '_pergame',
    'player_play_by_play': '_playbyplay',
    'player_shooting': '_shooting',
    'player_totals': '_totals',
}

advanced = advanced.add_suffix(suffixes['advanced'])
all_star_selections = all_star_selections.add_suffix(suffixes['all_star_selections'])
end_of_season_teams_voting = end_of_season_teams_voting.add_suffix(suffixes['end_of_season_teams_voting'])
end_of_season_teams = end_of_season_teams.add_suffix(suffixes['end_of_season_teams'])
per_thirty_six = per_thirty_six.add_suffix(suffixes['per_thirty_six'])
player_award_shares = player_award_shares.add_suffix(suffixes['player_award_shares'])
player_per_game = player_per_game.add_suffix(suffixes['player_per_game'])
player_play_by_play = player_play_by_play.add_suffix(suffixes['player_play_by_play'])
player_shooting = player_shooting.add_suffix(suffixes['player_shooting'])
player_totals = player_totals.add_suffix(suffixes['player_totals'])

# Merging Player-Specific CSVs into a DataFrame
merged_player_df = pd.merge(advanced, all_star_selections, left_on=['player_adv', 'season_adv'], right_on=['player_allstar', 'season_allstar'], how='inner')
merged_player_df = pd.merge(merged_player_df, end_of_season_teams_voting, left_on=['player_adv', 'season_adv'], right_on=['player_voting', 'season_voting'], how='inner')
merged_player_df = pd.merge(merged_player_df, end_of_season_teams, left_on=['player_adv', 'season_adv'], right_on=['player_teams', 'season_teams'], how='inner')
merged_player_df = pd.merge(merged_player_df, per_thirty_six, left_on=['player_adv', 'season_adv'], right_on=['player_thirtysix', 'season_thirtysix'], how='inner')
merged_player_df = pd.merge(merged_player_df, player_award_shares, left_on=['player_adv', 'season_adv'], right_on=['player_award', 'season_award'], how='inner')
merged_player_df = pd.merge(merged_player_df, player_per_game, left_on=['player_adv', 'season_adv'], right_on=['player_pergame', 'season_pergame'], how='inner')
merged_player_df = pd.merge(merged_player_df, player_play_by_play, left_on=['player_adv', 'season_adv'], right_on=['player_playbyplay', 'season_playbyplay'], how='inner')
merged_player_df = pd.merge(merged_player_df, player_shooting, left_on=['player_adv', 'season_adv'], right_on=['player_shooting', 'season_shooting'], how='inner')
merged_player_df = pd.merge(merged_player_df, player_totals, left_on=['player_adv', 'season_adv'], right_on=['player_totals', 'season_totals'], how='inner')

# Remove columns with 'lg' in the name
columns_to_remove = merged_player_df.filter(like='lg').columns
merged_player_df = merged_player_df.drop(columns=columns_to_remove)

# Drop Duplicates
merged_player_df = merged_player_df.drop_duplicates(subset=['player_adv', 'season_adv'])

# Filter rows for NBA MVP award and winner
nba_mvp_winners = merged_player_df[(merged_player_df['award_award'] == 'nba mvp') & (merged_player_df['winner_award'] == True)]
nba_mvp_winner_columns = ['award_award', 'winner_award']
columns_to_keep = [col for col in merged_player_df.columns if col not in nba_mvp_winner_columns]

# Use One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(merged_player_df[columns_to_keep])

# Begin with Multiple Regression. Split into train, test, and validate sets
X = merged_player_df[columns_to_keep]

# Filter rows for NBA MVP award and winner
nba_mvp_winners = merged_player_df[(merged_player_df['award_award'] == 'nba mvp') & (merged_player_df['winner_award'] == True)]
nba_mvp_winners['target'] = 1
Y = merged_player_df.merge(nba_mvp_winners[['player_adv', 'season_adv', 'target']], on=['player_adv', 'season_adv'], how='left')['target'].fillna(0)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Parameters
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [10, 20, 30, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Parameter Grid
param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

# Fit for Best Hyperparameters
rf_model = RandomForestClassifier()
rf_Grid = GridSearchCV(estimator = rf_model, param_grid = param_grid, cv = 5, verbose = 2, n_jobs = -1)
rf_Grid.fit(X_train, y_train)
rf_Grid.best_params_

# Metrics
train_acc = rf_Grid.score(X_train, y_train)
test_acc = rf_Grid.score(X_test, y_test)
print(f'Train Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')

importances = rf_Grid.best_estimator_.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Feature importances
plt.figure(figsize=(10,6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.show()
