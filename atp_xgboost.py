import pandas as pd
from datetime import datetime, timedelta
from timeit import default_timer as timer
import requests, zipfile, io
import os

from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance, plot_tree

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from bs4 import BeautifulSoup
from scrape import get_json, get_utr, get_current_tourney_stats
from h2h import get_rec_h2h
from home_adv import get_home_adv

# Data sources
# https://github.com/JeffSackmann/tennis_atp
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

# Get elo rankings before each tournament (2019-2020)
def closest_past_date(date_list, curr_date):
    """ Returns index corresponding to the rankings date closest (but prior) to the current date. """
    diff = date_list-curr_date
    return diff[(diff < pd.to_timedelta(0))].idxmax()

def get_elo(elo, row):
    """ Returns elo ranking of winner and loser, given dataframe of elo rankings and current match data. 
    If elo ranking not listed, return ranking of 0. """
    player_name, opp_name = row['Winner'], row['Loser'] 
    if player_name[-1] == ' ':
        player_name = player_name[:-1]
    player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-2], player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    if opp_name[-1] == ' ':
        opp_name = opp_name[:-1]
    opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-2], opp_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    
    try:
        player = elo[(elo['name'].str.split().str[-1]==player_surname) & (elo['name'].str.contains(player_first_name))]
        player_elo_rank = player['rank'].iloc[0]
    except:
        player_elo_rank = None
    
    try:
        opp = elo[(elo['name'].str.split().str[-1]==opp_surname) & (elo['name'].str.contains(opp_first_name))]
        opp_elo_rank = opp['rank'].iloc[0]
    except:
        opp_elo_rank = None
    
    return (player_elo_rank, opp_elo_rank)
    
    
def scrape_elo():
    """ Returns list of winner and loser elo rankings, prior to each match played. """
    
    # Get dataframe of elo ranking dates
    elo_dates = pd.DataFrame()
    for year in [2018, 2019, 2020]:
        dates = pd.read_json(f"https://www.ultimatetennisstatistics.com/seasonRankingDates?rankType=ELO_RANK&season={year}")
        elo_dates = elo_dates.append(dates)
    elo_dates.columns=['Date']
    elo_dates.sort_values('Date', inplace=True)
    elo_dates.reset_index(inplace=True, drop=True)
    elo_dates = pd.to_datetime(elo_dates['Date']).dt.date
    
    # Get dataframe of tournament dates
    tourney_dates = atp_bet_data['Date'].dt.date
    
    # Get dates corresponding to elo ranking before tournament
    date_indices = tourney_dates.apply(lambda x: closest_past_date(elo_dates, x))
    dates_before = elo_dates[date_indices]
    dates_before = [date.strftime('%d-%m-%Y') for date in dates_before]
    
    # Get elo rankings before tournament
    elo_winners = []
    elo_losers = []
    date = dates_before[0]
    elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date={date}&_=1578626200145")
    elo_winner, elo_loser = get_elo(elo, atp_bet_data.iloc[0])
    elo_winners.append(elo_winner)
    elo_losers.append(elo_loser)
    
    for i in range(1, len(dates_before)):
        # If date different to previous, get new elo rankings
        if dates_before[i] != dates_before[i-1]:
            date = dates_before[i]
            elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date={date}&_=1578626200145")

        elo_winner, elo_loser = get_elo(elo, atp_bet_data.iloc[i])
        elo_winners.append(elo_winner)
        elo_losers.append(elo_loser)
    
    return elo_winners, elo_losers


def get_dob():
    """ Returns list of winner and loser date of birth, corresponding to each match played. """
    atp_rankings = pd.read_html('http://tennisabstract.com/reports/atpRankings.html')[-1]
    dob_winners = []
    dob_losers = []
    for ind, row in atp_bet_data.iterrows():
        player_name, opp_name = row['Winner'], row['Loser'] 
        if player_name[-1] == ' ':
            player_name = player_name[:-1]
        player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-2], player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
        if opp_name[-1] == ' ':
            opp_name = opp_name[:-1]
        opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-2], opp_name.replace('-',' ').replace(',','').split(' ')[-1][0]
        
        try:
            player = atp_rankings[(atp_rankings['Player'].str.split().str[-1]==player_surname) & (atp_rankings['Player'].str.contains(player_first_name))]
            player_dob = player['Birthdate'].iloc[0]
        except:
            player_dob = None
        
        try:
            opp = atp_rankings[(atp_rankings['Player'].str.split().str[-1]==opp_surname) & (atp_rankings['Player'].str.contains(opp_first_name))]
            opp_dob = opp['Birthdate'].iloc[0]
        except:
            opp_dob = None
        dob_winners.append(player_dob)
        dob_losers.append(opp_dob)
    
    return dob_winners, dob_losers


def get_player_ids():
    """ Returns dataframe of player names and ids. """
    elo = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date=&_=1578626200145")
    player_ids = []
    
    for ind, row in atp_bet_data.iterrows():
        player_name = row['Winner']
        if player_name[-1] == ' ':
            player_name = player_name[:-1]
        player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-2], player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
        try:
            player = elo[(elo['name'].str.split().str[-1]==player_surname) & (elo['name'].str.contains(player_first_name))]
            player_id = player['playerId'].iloc[0]
        except:
            player_id = None
    
        player_ids.append([player_name, player_id])
    
    player_id = pd.DataFrame(player_ids, columns=['Name', 'PlayerId']).drop_duplicates()
    return player_id


def get_player_fixed_stats():
    """ Returns dataframe of fixed player statistics: player name, id, country, height, favourite surface. """
    countries = []
    heights = []
    fav_surfaces = []
    player_fixed_stats = get_player_ids()
    for ind, row in player_fixed_stats.iterrows():
        try:
            player_id = int(row['PlayerId'])
            x=pd.read_html(f'https://www.ultimatetennisstatistics.com/playerProfileTab?playerId={player_id}')[0]
            x.set_index(0, drop=True, inplace=True)
            try:
                country = x.loc['Country'].iloc[0]
            except:
                country = None
            try:
                height = x.loc['Height'].iloc[0]
            except:
                height = None
            try:
                fav_surface = x.loc['Favorite Surface'].iloc[0]
            except:
                fav_surface = None
        except:
            country = None
            height = None
            fav_surface = None
        countries.append(country)
        heights.append(height)
        fav_surfaces.append(fav_surface)
    
    player_fixed_stats['Country'] = countries
    player_fixed_stats['Height'] = heights
    player_fixed_stats['FavSurface'] = fav_surfaces

    return player_fixed_stats
    


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

# Import data
# =================
atp_bet_data = pd.read_excel('ATP_dataset_17_18_19_20.xlsx')
player_fixed_stats = pd.read_excel('ATP_player_dataset.xlsx')



# Data cleaning and manipulation
# =================
# Impute missing elo with mean
#atp_bet_data['WEloRank'].fillna(63, inplace=True)
#atp_bet_data['LEloRank'].fillna(63, inplace=True)

# Impute missing age with mean
atp_bet_data['WAge'].fillna(round(atp_bet_data['WAge'].mean()), inplace=True)
atp_bet_data['LAge'].fillna(round(atp_bet_data['LAge'].mean()), inplace=True)

# Transform height to numerical
player_fixed_stats['Height'] = player_fixed_stats['Height'].apply(lambda x: float(x.split()[0]) if type(x)==str else None)

# Impute missing height with mean
player_fixed_stats['Height'].fillna(round(player_fixed_stats['Height'].mean()), inplace=True)


# Randomly transform winner/loser to player1/player2
data = atp_bet_data[['Winner', 'Loser', 'WRank', 'LRank', 'AvgW', 'AvgL', 'WEloRank', 'LEloRank', 'Surface', 'WAge', 'LAge']].sample(frac=1, random_state=2)

# Add height data
data['WHeight'] = data['Winner'].apply(lambda x: player_fixed_stats[player_fixed_stats['Name']==x]['Height'].iloc[0])
data['LHeight'] = data['Loser'].apply(lambda x: player_fixed_stats[player_fixed_stats['Name']==x]['Height'].values[0] if len(player_fixed_stats[player_fixed_stats['Name']==x]['Height'].values) != 0 else None)

# Impute missing loser height with mean
data['LHeight'].fillna(round(player_fixed_stats['Height'].mean()), inplace=True)

# Drop rows with missing (elo) values
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


player1_win = [1]*half+[0]*half
data['P1Win'] = player1_win


# Create features
# =================
#features = ['P1Rank', 'P2Rank', 'AvgP1', 'AvgP2', 'P1EloRank', 'P2EloRank']
X_full = pd.DataFrame()
X_full['P1Rank'] = player1_rank
X_full['P2Rank'] = player2_rank
X_full['AvgP1'] = player1_avg_odds
X_full['AvgP2'] = player2_avg_odds
X_full['P1EloRank'] = player1_elo_rank
X_full['P2EloRank'] = player2_elo_rank
X_full['P1Age'] = player1_age
X_full['P2Age'] = player2_age
X_full['P1Height'] = player1_height
X_full['P2Height'] = player2_height


"""
X_full['RankDiff'] = player1_rank-player2_rank
X_full['EloDiff'] = player1_elo_rank-player2_elo_rank
X_full['AgeDiff'] = player1_age-player2_age
X_full['HeightDiff'] = player1_height-player2_height
X_full['AvgP1'] = player1_avg_odds
X_full['AvgP2'] = player2_avg_odds
"""

X_full.sort_index(inplace=True)
X = X_full[X_full.index<7961]
X_test = X_full[X_full.index>=7961]

# Convert categorical columns to numerical 
# Label encode surface
#encoder = LabelEncoder()
#X_full['Surface'] = encoder.fit_transform(data['Surface'])


# Filter out target variable 
# y = 1 if player 1 wins, 0 otherwise
y_full = pd.Series(data['P1Win'].sort_index())
y = y_full[y_full.index<7961]
y_test = y_full[y_full.index>=7961]





# VALIDATION SET (20%) - 2019/2020 matches < Aus Open 2020
# =================

# Create training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=2, shuffle=True)

# Define the model
#scale_weight = (len(y_train)/sum(y_train)-1)**0.5
model = XGBClassifier(n_estimators=1000, learning_rate=0.01, random_state=0, objective=
model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_valid, y_valid)], eval_metric=['mae'], verbose=False)

# Get predictions for validation set
y_prob = model.predict(X_valid)
y_pred = [round(value) for value in y_prob]

# Evaluate the model
# Get validation accuracy score
score = accuracy_score(y_valid, y_pred)
print('Accuracy:', score)

# Get validation confusion matrix
print('Confusion matrix: ')
print(confusion_matrix(y_valid, y_pred))


# Get profit
profit = 0
stake = 5
i = 0
count = 0
correct = 0

for ind, row in X_valid.iterrows():
    if y_pred[i] == 1 and 1 < y_prob[i]*row['AvgP1'] < 1.1:
        if y_valid[ind] == 1:
            player1_odds = row['AvgP1']
            profit += (player1_odds-1)*stake
            correct += 1
        else:
            profit -= stake
        count += 1
        print(round(profit, 2))
    elif y_pred[i] == 0 and 1 < (1-y_prob[i])*row['AvgP2'] < 1.1:
        if y_valid[ind] == 0:
            player2_odds = row['AvgP2']
            profit += (player2_odds-1)*stake
            correct += 1
        else:
            profit -= stake  
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



# TEST SET - 2020 matches >= Aus Open 2020
# =================

# Get predictions for validation set
y_prob = model.predict(X_test)
y_pred = [round(value) for value in y_prob]

# Evaluate the model
# Get validation accuracy score
score = accuracy_score(y_test, y_pred)
print('Accuracy:', score)

# Get validation confusion matrix
print('Confusion matrix: ')
print(confusion_matrix(y_test, y_pred))


# Get profit
profit = 0
stake = 5
i = 0
count = 0
correct = 0

for ind, row in X_test.iterrows():
    if y_pred[i] == 1 and 1 < y_prob[i]*row['AvgP1'] < 1.1:
        if y_test[ind] == 1:
            player1_odds = row['AvgP1']
            profit += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(round(profit, 2))
    elif y_pred[i] == 0 and 1 < (1-y_prob[i])*row['AvgP2'] < 1.1:
        if y_test[ind] == 0:
            player2_odds = row['AvgP2']
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

