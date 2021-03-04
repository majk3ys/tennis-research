import pandas as pd
from datetime import datetime, timedelta
from timeit import default_timer as timer
import requests, zipfile, io
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from bs4 import BeautifulSoup
from scrape import get_json, get_utr, get_current_tourney_stats
from h2h import get_rec_h2h
from home_adv import get_home_adv

# Data sources
# https://www.ultimatetennisstatistics.com/eloRatings
# http://www.tennis-data.co.uk/alldata.php



# Data wrangling and quality checks
# =================
"""
# Import ATP bet data from 2019 and 2020
atp_bet_data_2019 = pd.read_excel('atpdata/bet_data_2019.xlsx')

r = requests.get("http://www.tennis-data.co.uk/2020/2020.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
os.rename('2020.xlsx','atpdata/bet_data_2020.xlsx')
atp_bet_data_2020 = pd.read_excel('atpdata/bet_data_2020.xlsx')

atp_bet_data = atp_bet_data_2019.append(atp_bet_data_2020)
"""


"""
# Get elo, dob and age
elo_winners, elo_losers = scrape_elo()
atp_bet_data['WEloRank'] = elo_winners
atp_bet_data['LEloRank'] = elo_losers

dob_winners, dob_losers = get_dob()
atp_bet_data['WDob'] = dob_winners
atp_bet_data['LDob'] = dob_losers
atp_bet_data['WDob'] = pd.to_datetime(atp_bet_data['WDob'])
atp_bet_data['LDob'] = pd.to_datetime(atp_bet_data['LDob'])

atp_bet_data['WAge'] = round(((atp_bet_data['Date']-atp_bet_data['WDob'])/365).dt.days, 2)
atp_bet_data['LAge'] = round(((atp_bet_data['Date']-atp_bet_data['LDob'])/365).dt.days, 2)

atp_bet_data['Winner'] = atp_bet_data['Winner'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
atp_bet_data['Loser'] = atp_bet_data['Loser'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)

# Save new match dataset to Excel
atp_bet_data.to_excel('ATP_dataset_19_20.xlsx')

# Get fixed player stats
player_fixed_stats = get_player_fixed_stats()

# Save new player dataset to Excel
player_fixed_stats.to_excel('ATP_player_dataset.xlsx')
"""


def surface(fav_surf):
    """ Returns None if player has surface advantage, otherwise a string of surfaces on which they have an advantage. """
    if type(fav_surf) != str or 'All-Rounder' in fav_surf:
        return None
    surfaces = [surf.replace('(', '').replace(')', '').replace(',','') for surf in fav_surf.split() if '%' not in surf and surf not in ['Slow', 'Fast', 'Soft', 'Carpet']]
    surfaces = [surf.replace('Cl', 'Clay').replace('H', 'Hard').replace('G', 'Grass')  if len(surf) <= 2 else surf for surf in surfaces]
    return ' '.join(surfaces)


# Import data
# =================
atp_bet_data = pd.read_excel('ATP_dataset.xlsx')
player_fixed_stats = pd.read_excel('ATP_player_dataset.xlsx')
player_fixed_stats['FavSurface'] = player_fixed_stats['FavSurface'].apply(lambda x: surface(x))

# Filter only Grand Slams and completed matches
atp_bet_data = atp_bet_data[(atp_bet_data['Series']!='ATP250')&(atp_bet_data['Comment']=='Completed')]

# Drop rows with missing odds
atp_bet_data.dropna(subset=['AvgW', 'AvgL'], inplace=True)

#(atp_bet_data['Series']=='Grand Slam')

# Create features
# =================

# Impute missing age with mean
atp_bet_data['WAge'].fillna(27.5, inplace=True)
atp_bet_data['LAge'].fillna(27.5, inplace=True)

# Transform height to numerical
#player_fixed_stats['Height'] = player_fixed_stats['Height'].apply(lambda x: float(x.split()[0]) if type(x)==int else None)

# Impute missing height with mean
player_fixed_stats['Height'].fillna(round(player_fixed_stats['Height'].mean()), inplace=True)

# Randomly transform winner/loser to player1/player2
data = atp_bet_data[['Winner', 'Loser', 'WRank', 'LRank', 'AvgW', 'AvgL', 'WEloRank', 'LEloRank', 'WEloSurfRank', 'LEloSurfRank', 'WReturnEloRank', 'LReturnEloRank', 'WAge', 'LAge', 'WForm', 'LForm', 'WSetsPrev', 'LSetsPrev', 'WHomeAdv', 'LHomeAdv', 'WSurfAdv', 'LSurfAdv', 'WTourneySuccess', 'LTourneySuccess', 'Surface', 'Series', 'Court', 'Round']].sample(frac=1, random_state=2)

# Add height data - impute missing values with mean
data['WHeight'] = data['Winner'].apply(lambda x: player_fixed_stats[player_fixed_stats['Name']==x]['Height'].values[0] if len(player_fixed_stats[player_fixed_stats['Name']==x]['Height'].values) != 0 else player_fixed_stats['Height'].mean())
data['LHeight'] = data['Loser'].apply(lambda x: player_fixed_stats[player_fixed_stats['Name']==x]['Height'].values[0] if len(player_fixed_stats[player_fixed_stats['Name']==x]['Height'].values) != 0 else player_fixed_stats['Height'].mean())

# Add playing hand data - if missing, assume right-handed
data['WPlays'] = data['Winner'].apply(lambda x: player_fixed_stats[player_fixed_stats['Name']==x]['Plays'].values[0] if len(player_fixed_stats[player_fixed_stats['Name']==x]['Plays'].values) != 0 else 'Right-handed')
data['LPlays'] = data['Loser'].apply(lambda x: player_fixed_stats[player_fixed_stats['Name']==x]['Plays'].values[0] if len(player_fixed_stats[player_fixed_stats['Name']==x]['Plays'].values) != 0 else 'Right-handed')
data['WPlays'].fillna('Right-handed', inplace=True)
data['LPlays'].fillna('Right-handed', inplace=True)

# Label encode playing hand
encoder = LabelEncoder()
data['WPlays'] = encoder.fit_transform(data['WPlays'])
data['LPlays'] = encoder.fit_transform(data['LPlays'])

# Impute missing tourney success with 0.5
data['WTourneySuccess'].fillna(0.5, inplace=True)
data['LTourneySuccess'].fillna(0.5, inplace=True)

# Impute missing form with 0.5
data['WForm'].fillna(0.5, inplace=True)
data['LForm'].fillna(0.5, inplace=True)

#data.dropna(inplace=True)
"""
# Impute missing elo rank with 200
data['WEloRank'].fillna(200, inplace=True)
data['LEloRank'].fillna(200, inplace=True)

# Impute missing elo rank with 200
data['WReturnEloRank'].fillna(200, inplace=True)
data['LReturnEloRank'].fillna(200, inplace=True)

# Impute missing elo rank with 200
data['WEloSurfRank'].fillna(200, inplace=True)
data['LEloSurfRank'].fillna(200, inplace=True)
"""

# Drop rows with missing values
data.dropna(inplace=True)

# Assign player1 to winner (first half), assign player1 to loser (second half)
half = len(data)//2
player1 = data['Winner'].iloc[:half].append(data['Loser'].iloc[half:])
player2 = data['Loser'].iloc[:half].append(data['Winner'].iloc[half:])

player1_rank = data['WRank'].iloc[:half].append(data['LRank'].iloc[half:])
player2_rank = data['LRank'].iloc[:half].append(data['WRank'].iloc[half:])

player1_avg_odds = data['AvgW'].iloc[:half].append(data['AvgL'].iloc[half:])
player2_avg_odds = data['AvgL'].iloc[:half].append(data['AvgW'].iloc[half:])

player1_elo_rank = data['WEloRank'].iloc[:half].append(data['LEloRank'].iloc[half:])
player2_elo_rank = data['LEloRank'].iloc[:half].append(data['WEloRank'].iloc[half:])

player1_age = data['WAge'].iloc[:half].append(data['LAge'].iloc[half:])
player2_age = data['LAge'].iloc[:half].append(data['WAge'].iloc[half:])

player1_height = data['WHeight'].iloc[:half].append(data['LHeight'].iloc[half:])
player2_height = data['LHeight'].iloc[:half].append(data['WHeight'].iloc[half:])

player1_surf_adv = data['WSurfAdv'].iloc[:half].append(data['LSurfAdv'].iloc[half:])
player2_surf_adv = data['LSurfAdv'].iloc[:half].append(data['WSurfAdv'].iloc[half:])

player1_form = data['WForm'].iloc[:half].append(data['LForm'].iloc[half:])
player2_form = data['LForm'].iloc[:half].append(data['WForm'].iloc[half:])

player1_surf_elo = data['WEloSurfRank'].iloc[:half].append(data['LEloSurfRank'].iloc[half:])
player2_surf_elo = data['LEloSurfRank'].iloc[:half].append(data['WEloSurfRank'].iloc[half:])

player1_ret_elo = data['WReturnEloRank'].iloc[:half].append(data['LReturnEloRank'].iloc[half:])
player2_ret_elo = data['LReturnEloRank'].iloc[:half].append(data['WReturnEloRank'].iloc[half:])

player1_sets_prev = data['WSetsPrev'].iloc[:half].append(data['LSetsPrev'].iloc[half:])
player2_sets_prev = data['LSetsPrev'].iloc[:half].append(data['WSetsPrev'].iloc[half:])

player1_home_adv = data['WHomeAdv'].iloc[:half].append(data['LHomeAdv'].iloc[half:])
player2_home_adv = data['LHomeAdv'].iloc[:half].append(data['WHomeAdv'].iloc[half:])

player1_hand = data['WPlays'].iloc[:half].append(data['LPlays'].iloc[half:])
player2_hand = data['LPlays'].iloc[:half].append(data['WPlays'].iloc[half:])

player1_tourney_success = data['WTourneySuccess'].iloc[:half].append(data['LTourneySuccess'].iloc[half:])
player2_tourney_success = data['LTourneySuccess'].iloc[:half].append(data['WTourneySuccess'].iloc[half:])

player1_win = [1]*half+[0]*(len(data)-half)
data['P1Win'] = player1_win


# Assign features
# =================

"""
X_full = pd.DataFrame()
X_full['P1Rank'] = player1_rank
X_full['P2Rank'] = player2_rank
X_full['P1EloRank'] = player1_elo_rank
X_full['P2EloRank'] = player2_elo_rank
X_full['P1Age'] = player1_age
X_full['P2Age'] = player2_age
X_full['P1Height'] = player1_height
X_full['P2Height'] = player2_height
X_full['P1SurfAdv'] = player1_surf_adv
X_full['P2SurfAdv'] = player2_surf_adv
X_full['P1Form'] = player1_form
X_full['P2Form'] = player2_form
X_full['P1EloSurfRank'] = player1_surf_elo
X_full['P2EloSurfRank'] = player2_surf_elo
X_full['P1SetsPrev'] = player1_sets_prev
X_full['P2SetsPrev'] = player2_sets_prev
X_full['P1Avg'] = player1_avg_odds
X_full['P2Avg'] = player2_avg_odds
X_full['P1HomeAdv'] = player1_home_adv
X_full['P2HomeAdv'] = player2_home_adv
X_full['P1Plays'] = player1_hand
X_full['P2Plays'] = player2_hand
X_full['P1ReturnElo'] = player1_ret_elo
X_full['P2ReturnElo'] = player2_ret_elo
X_full['P1TourneySuccess'] = player1_tourney_success
X_full['P2TourneySuccess'] = player2_tourney_success

"""

X_full = pd.DataFrame()
X_full['RankDiff'] = np.log(player1_rank)-np.log(player2_rank)
X_full['EloDiff'] = np.log(player1_elo_rank)-np.log(player2_elo_rank)
X_full['OddsDiff'] = player1_avg_odds-player2_avg_odds
X_full['AgeDiff'] = player1_age-player2_age
X_full['HeightDiff'] = player1_height-player2_height
#X_full['SurfAdvDiff'] = player1_surf_adv-player2_surf_adv
X_full['FormDiff'] = player1_form-player2_form
X_full['EloSurfDiff'] = np.log(player1_surf_elo)-np.log(player2_surf_elo)
X_full['ReturnEloDiff'] = np.log(player1_ret_elo)-np.log(player2_ret_elo)
#X_full['SetsPrevDiff'] = player1_sets_prev-player2_sets_prev
X_full['HomeAdvDiff'] = player1_home_adv-player2_home_adv
X_full['PlaysDiff'] = player1_hand-player2_hand
X_full['TourneySuccessDiff'] = player1_tourney_success-player2_tourney_success


# Convert categorical columns to numerical
encoder = LabelEncoder()
#X_full['Surface'] = encoder.fit_transform(data['Surface'])
#X_full['Court'] = encoder.fit_transform(data['Court'])
X_full['Series'] = encoder.fit_transform(data['Series'])
# Label encode round
X_full['Round'] = encoder.fit_transform(data['Round'])




# 18448-18574 Aus 2020
# 7323-7448 US 2019
# 6475-6601 French 2019
# 6772-6898 Wimbledon 2019


# VALIDATION SET (20%) - 2019/2020 matches < Aus Open 2020
# =================
# Create training and validation sets
X_full.sort_index(inplace=True)
y_full = pd.Series(data['P1Win'].sort_index())
X = X_full.sample(random_state=0, frac=1)
y = y_full.sample(random_state=0, frac=1)

"""
fro = 18448
up_to = 18574
X = X_full[(X_full.index>up_to)|(X_full.index<fro)]
X_test = X_full[(fro<=X_full.index)&(X_full.index<=up_to)]

# Filter out target variable 
# y = 1 if player 1 wins, 0 otherwise
y = y_full[(y_full.index>up_to)|(y_full.index<fro)]
y_test = y_full[(fro<=y_full.index)&(y_full.index<=up_to)]

X = X.sample(random_state=0, frac=1)
y = y.sample(random_state=0, frac=1)
"""

eighty = int(len(X)*0.8)
ninety = int(len(X)*0.9)

X_train = X.iloc[:eighty]
X_valid = X.iloc[eighty: ninety]

y_train = y.iloc[:eighty]
y_valid = y.iloc[eighty: ninety]

X_test = X.iloc[ninety:]
y_test = y.iloc[ninety:]

"""
ninety = int(len(X_full)*0.9)
X_test = X_full.iloc[ninety:]
y_test = y_full.iloc[ninety:]
X_test = X_test.sample(random_state=0, frac=1)
y_test = y_test.sample(random_state=0, frac=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=False)
"""

df = X_full.copy()
df['AvgP1'] = player1_avg_odds
df['AvgP2'] = player2_avg_odds

"""
X_full.sort_index(inplace=True)
y_full = pd.Series(data['P1Win'].sort_index())

#y_full = pd.Series(data['P1Win'])
                   
X_full = X_full.sample(random_state=0, frac=1)
y_full = y_full.sample(random_state=0, frac=1)


eighty = int(len(X_full)*0.8)
ninety = int(len(X_full)*0.9)
X_train = X_full.iloc[:eighty]
X_valid = X_full.iloc[eighty: ninety]
X_test = X_full.iloc[ninety:]

y_train = y_full.iloc[:eighty]
y_valid = y_full.iloc[eighty: ninety]
y_test = y_full.iloc[ninety:]
"""



"""
# Grid search
# define models and parameters
X, y = make_blobs(n_samples=1000, centers=2, n_features=12, cluster_std=20)
model = LogisticRegression(class_weight='balanced', max_iter=100)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc', error_score=0)
grid_result = grid_search.fit(X_full, y_full)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
"""
    
    
# Define the model
#scale_weight = (len(y_train)/sum(y_train)-1)**0.5
pipe = make_pipeline(Normalizer(), LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2', C=0.01, max_iter=100))
pipe.fit(X_train, y_train)

#model = LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2', C=0.01, max_iter=100)
#model.fit(X_train, y_train)

# Get predictions for validation set
y_prob = [val[1] for val in pipe.predict_proba(X_valid)]
y_pred = [round(value) for value in y_prob]

# Evaluate the model
# Get validation accuracy score
score = accuracy_score(y_valid, y_pred)
print('Accuracy:', score)

# Get validation confusion matrix
print('Confusion matrix: ')
print(confusion_matrix(y_valid, y_pred))


# Kelly criterion
init_capital = 100
capital = 100
staked = 0
i = 0
count = 0
correct = 0

for ind, row in X_valid.iterrows():
    if y_pred[i] == 1 and 1 < y_prob[i]*df.loc[ind]['AvgP1'] and df.loc[ind]['AvgP1'] > 1.3:
        perc = ((df.loc[ind]['AvgP1']-1)*y_prob[i]-(1-y_prob[i]))/(df.loc[ind]['AvgP1']-1)
        stake = max(5, min(perc*capital, 10))
        staked += stake
        if y_valid[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            capital += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            capital -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(stake)
        print(y_prob[i])
        print(round(capital, 2))
    elif y_pred[i] == 0 and 1 < (1-y_prob[i])*df.loc[ind]['AvgP2'] and df.loc[ind]['AvgP2'] > 1.3:
        perc = ((df.loc[ind]['AvgP2']-1)*(1-y_prob[i])-y_prob[i])/(df.loc[ind]['AvgP2']-1)
        stake = max(5, min(perc*capital, 10))
        staked += stake
        if y_valid[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            capital += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            capital -= stake  
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(stake)
        print(1-y_prob[i])
        print(round(capital, 2))
    i += 1

print()
print("VALIDATION SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(capital-init_capital, 2)}")
print(f"Return: {round(100*(capital-init_capital)/(staked), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round((capital-init_capital)/count, 2)}")
print()
print()

"""
# Get profit
profit = 0
stake = 5
i = 0
count = 0
correct = 0

for ind, row in X_valid.iterrows():
    if y_pred[i] == 1 and 1 < y_prob[i]*df.loc[ind]['AvgP1'] and df.loc[ind]['AvgP1'] > 1.3:
        if y_valid[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            profit += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(round(profit, 2))
    elif y_pred[i] == 0 and 1 < (1-y_prob[i])*df.loc[ind]['AvgP2'] and df.loc[ind]['AvgP2'] > 1.3:
        if y_valid[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            profit += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            profit -= stake  
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(round(profit, 2))
    i += 1


print()
print("VALIDATION SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(profit, 2)}")
print(f"Return: {round(100*profit/(stake*count), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round(profit/count, 2)}")
"""


# TEST SET - 2020 matches >= Aus Open 2020
# =================

# Get predictions for test set
y_prob = [val[1] for val in pipe.predict_proba(X_test)]
y_pred = [round(value) for value in y_prob]

# Evaluate the model
# Get test accuracy score
score = accuracy_score(y_test, y_pred)
print('Accuracy:', score)

# Get test confusion matrix
print('Confusion matrix: ')
print(confusion_matrix(y_test, y_pred))

# Kelly criterion
init_capital = 100
capital = 100
staked = 0
i = 0
count = 0
correct = 0

for ind, row in X_test.iterrows():
    if y_pred[i] == 1 and 1 < y_prob[i]*df.loc[ind]['AvgP1'] and df.loc[ind]['AvgP1'] > 1.3:
        perc = ((df.loc[ind]['AvgP1']-1)*y_prob[i]-(1-y_prob[i]))/(df.loc[ind]['AvgP1']-1)
        stake = max(5, min(perc*capital, 10))
        staked += stake
        if y_test[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            capital += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            capital -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(stake)
        print(round(capital, 2))
    elif y_pred[i] == 0 and 1 < (1-y_prob[i])*df.loc[ind]['AvgP2'] and df.loc[ind]['AvgP2'] > 1.3:
        perc = ((df.loc[ind]['AvgP2']-1)*(1-y_prob[i])-y_prob[i])/(df.loc[ind]['AvgP2']-1)
        stake = max(5, min(perc*capital, 10))
        staked += stake
        if y_test[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            capital += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            capital -= stake  
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(stake)
        print(round(capital, 2))
    i += 1

print()
print("TEST SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(capital-init_capital, 2)}")
print(f"Return: {round(100*(capital-init_capital)/(staked), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round((capital-init_capital)/count, 2)}")



"""
# Get profit
profit = 0
stake = 5
i = 0
count = 0
correct = 0

for ind, row in X_test.iterrows():
    if y_pred[i] == 1 and 1 < y_prob[i]*df.loc[ind]['AvgP1'] and df.loc[ind]['AvgP1'] > 1.3:
        if y_test[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            profit += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(round(profit, 2))
    elif y_pred[i] == 0 and 1 < (1-y_prob[i])*df.loc[ind]['AvgP2'] and df.loc[ind]['AvgP2'] > 1.3:
        if y_test[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            profit += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(round(profit, 2))
    i += 1


print()
print("TEST SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(profit, 2)}")
print(f"Return: {round(100*profit/(stake*count), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round(profit/count, 2)}")
"""
