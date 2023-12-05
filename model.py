import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

player_data = pd.read_csv('player_data.csv')
players = pd.read_csv('players.csv')
seasons_stats = pd.read_csv('Seasons_Stats.csv')

print(player_data.head())
print(players.head())
print(seasons_stats.head())
