import pandas as pd
import requests, zipfile, io
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from atp_helpers2 import scrape_elo

# Download latest ATP match dataset
r = requests.get("http://www.tennis-data.co.uk/2020/2020.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
os.rename('2020.xlsx','atpdata/bet_data_2020.xlsx')
atp_bet_data_2020 = pd.read_excel('atpdata/bet_data_2020.xlsx')

# Get previous ATP dataset and player dataset
atp_bet_data = pd.read_excel('ATP_dataset.xlsx')
player_fixed_stats = pd.read_excel('ATP_player_dataset.xlsx')

# Update features for new matches
last_date = atp_bet_data.iloc[-1]['Date']
#last_date = atp_bet_data.iloc[-50]['Date']

new_data = atp_bet_data_2020[atp_bet_data_2020['Date']>last_date]

# Get elo rankings
elo_winners, elo_losers = scrape_elo('overall', new_data)
ret_elo_winners, ret_elo_losers = scrape_elo('return', new_data)
hard_elo_winners, hard_elo_losers = scrape_elo('HARD_ELO_RANK', new_data)
clay_elo_winners, clay_elo_losers = scrape_elo('CLAY_ELO_RANK', new_data)
grass_elo_winners, grass_elo_losers = scrape_elo('GRASS_ELO_RANK', new_data)

new_data['WEloRank'] = elo_winners
new_data['LEloRank'] = elo_losers
new_data['WEloHardRank'] = hard_elo_winners
new_data['LEloHardRank'] = hard_elo_losers
new_data['WEloClayRank'] = clay_elo_winners
new_data['LEloClayRank'] = clay_elo_losers
new_data['WEloGrassRank'] = grass_elo_winners
new_data['LEloGrassRank'] = grass_elo_losers
new_data['WReturnEloRank'] = ret_elo_winners
new_data['LReturnEloRank'] = ret_elo_losers

# Get surface elo ranking
surf_elo_winner = []
surf_elo_loser = []

# Get country where tournament is played
geolocator = Nominatim(user_agent="tennis_loc")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

countries = []
for loc in new_data['Location'].unique():
    location = geocode(loc)
    country = location.address.split(', ')[-1]
    countries.append((loc, country))
    
x = pd.DataFrame(countries)
countries = x.set_index(0, drop=True)

new_data['Country'] = new_data['Location'].apply(lambda x: countries.loc[x][1])


# Get home advantage
home_adv_winner = []
home_adv_loser = []

for ind, row in new_data.iterrows():
    if row['Surface'] == 'Hard':
        elo_winner = row['WEloHardRank']
        elo_loser = row['LEloHardRank']
    elif row['Surface'] == 'Grass':
        elo_winner = row['WEloGrassRank']
        elo_loser = row['LEloGrassRank']
    elif row['Surface'] == 'Clay':
        elo_winner = row['WEloClayRank']
        elo_loser = row['LEloClayRank']
    surf_elo_winner.append(elo_winner)
    surf_elo_loser.append(elo_loser)
    
    try:
        if player_fixed_stats[player_fixed_stats['Name']==row['Winner']]['Country'].iloc[0] == row['Country']:
            home_adv_winner.append(1)
        else:
            home_adv_winner.append(0)
    except:
        home_adv_winner.append(0)
    try:
        if player_fixed_stats[player_fixed_stats['Name']==row['Loser']]['Country'].iloc[0] == row['Country']:
            home_adv_loser.append(1)
        else:
            home_adv_loser.append(0)
    except:
        home_adv_loser.append(0)

"""   
last_id = 1
last_tourney = df.iloc[0]['Tournament']
new_ids = []
for ind, row in df.iterrows():
    if row['Tournament'] != last_tourney:
        last_id += 1
        last_tourney = row['Tournament']
    new_ids.append(last_id)
df['TournamentId'] = new_ids
"""   
 
# Update tournament ids
last_tourney = atp_bet_data.iloc[-1]['Tournament']
last_id = atp_bet_data.iloc[-1]['TournamentId']
#last_tourney = atp_bet_data.iloc[-50]['Tournament']
new_ids = []
for ind, row in new_data.iterrows():
    if row['Tournament'] != last_tourney:
        last_id += 1
        last_tourney = row['Tournament']
    new_ids.append(last_id)
new_data['TournamentId'] = new_ids


# Get recent form - win percentage over previous 2 tournaments prior to current tournament
df = atp_bet_data.append(new_data)

winner_df = pd.DataFrame(df['Winner'].values, columns=['Name'])
winner_df['Id'] = df['TournamentId'].values
winner_df['Win'] = 1

loser_df = pd.DataFrame(df['Loser'].values, columns=['Name'])
loser_df['Id'] = df['TournamentId'].values
loser_df['Win'] = 0

match_hist = winner_df.append(loser_df).sort_index(kind='mergesort') # stable

match_wins = match_hist.groupby(['Name','Id']).sum()
matches_played = match_hist.groupby(['Name','Id']).count()

form_winner = []
form_loser = []

for ind, row in new_data.iterrows():
     year = row['TournamentId']
     wins_winner = (match_wins.loc[row['Winner']].loc[:year-1].iloc[-2:].sum()/matches_played.loc[row['Winner']].loc[:year-1].iloc[-2:].sum()).iloc[0]
     form_winner.append(wins_winner)
     wins_loser = (match_wins.loc[row['Loser']].loc[:year-1].iloc[-2:].sum()/matches_played.loc[row['Loser']].loc[:year-1].iloc[-2:].sum()).iloc[0]
     form_loser.append(wins_loser)

new_data['WEloSurfRank'] = surf_elo_winner
new_data['LEloSurfRank'] = surf_elo_loser

new_data['WForm'] = form_winner
new_data['LForm'] = form_loser
new_data['WForm'].fillna(0.5, inplace=True)
new_data['LForm'].fillna(0.5, inplace=True)

new_data['WHomeAdv'] = home_adv_winner
new_data['LHomeAdv'] = home_adv_loser


# Get age
winner_ages = []
loser_ages = []
for ind, row in new_data.iterrows():
    try:
        winner_dob = player_fixed_stats[player_fixed_stats['Name']==row['Winner']]['DateOfBirth'].iloc[0]
        winner_ages.append(round(((row['Date']- winner_dob)/365).days, 2))
    except:
        print(row['Winner'])
        winner_ages.append(None)
    try:
        loser_dob = player_fixed_stats[player_fixed_stats['Name']==row['Loser']]['DateOfBirth'].iloc[0]
        loser_ages.append(round(((row['Date']- loser_dob)/365).days, 2))
    except:
        print(row['Loser'])
        loser_ages.append(None)
new_data['WAge'] = winner_ages
new_data['LAge'] = loser_ages



# Save updated features to excel spreadsheet
new_data.to_excel('atpdata/bet_data_2020.xlsx')


# Merge two datasets and save to excel
final = atp_bet_data.append(new_data)
final.reset_index(drop=True, inplace=True)
#final.to_excel('ATP_dataset.xlsx')
final.to_excel('ATP_dataset.xlsx')


