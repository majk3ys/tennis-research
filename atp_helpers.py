import pandas as pd
import requests, zipfile, io
import os

from scrape import get_json
from geopy.geocoders import Nominatim


# Get elo rankings before each tournament (2019-2020)
def closest_past_date(date_list, curr_date):
    """ Returns index corresponding to the rankings date closest (but prior) to the current date. """
    diff = date_list-curr_date
    return diff[(diff < pd.to_timedelta(0))].idxmax()

def get_elo(elo, row):
    """ Returns elo ranking of winner and loser, given dataframe of elo rankings and current match data. 
    If elo ranking not listed, return None. """
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

    
def scrape_elo(elo_type, atp_bet_data):
    """ Returns list of winner and loser elo rankings, prior to each match played. """
    
    # Get dataframe of elo ranking dates
    elo_dates = pd.DataFrame()
    for year in [2019, 2020]:
        dates = pd.read_json(f"https://www.ultimatetennisstatistics.com/seasonRankingDates?rankType=ELO_RANK&season={year}")
        elo_dates = elo_dates.append(dates)
    elo_dates.columns = ['Date']
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
    
    if elo_type == 'return':
        elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=RETURN_GAME_ELO_RANK&season=&date={date}&_=1578626200145")
    elif elo_type == 'overall':
        elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&sort%5Brank%5D=asc&searchPhrase=&rankType=ELO_RANK&season=2013&date={date}&_=1596435645168")
    else:
        elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&sort%5Brank%5D=asc&searchPhrase=&rankType={elo_type}&season=&date={date}&_=1596435645168")
    elo_winner, elo_loser = get_elo(elo, atp_bet_data.iloc[0])
    elo_winners.append(elo_winner)
    elo_losers.append(elo_loser)
    
    for i in range(1, len(dates_before)):
        # If date different to previous, get new elo rankings
        if dates_before[i] != dates_before[i-1]:
            date = dates_before[i]
            if elo_type == 'return':
                elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=RETURN_GAME_ELO_RANK&season=&date={date}&_=1578626200145")
            elif elo_type == 'overall':
                 elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&sort%5Brank%5D=asc&searchPhrase=&rankType=ELO_RANK&season=2013&date={date}&_=1596435645168")
            else:
                elo = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&sort%5Brank%5D=asc&searchPhrase=&rankType={surface}&season=&date={date}&_=1596435645168")
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
    
def surface(fav_surf):
    """ Returns None if player has surface advantage, otherwise a string of surfaces on which they have an advantage. """
    if type(fav_surf) != str or 'All-Rounder' in fav_surf:
        return None
    surfaces = [surf.replace('(', '').replace(')', '').replace(',','') for surf in fav_surf.split() if '%' not in surf and surf not in ['Slow', 'Fast', 'Soft', 'Carpet']]
    surfaces = [surf.replace('Cl', 'Clay').replace('H', 'Hard').replace('G', 'Grass')  if len(surf) <= 2 else surf for surf in surfaces]
    return ' '.join(surfaces)



from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
geolocator = Nominatim(user_agent="tennis_loc")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def get_country(tourney_loc):
    location = geolocator.geocode(tourney_loc)
    country = location.address.split()[-1]
    return country

location = geolocator.geocode(atp_bet_data['Location'].iloc[0])
country = location.address.split()[-1]
countries = [country]
for ind, row in atp_bet_data.iloc[1:].iterrows():
    if row['Location'] != location:
        location = geolocator.geocode(row['Location'])
        country = location.address.split()[-1]
    countries.append(country)
atp_bet_data['Country'] = countries

countries = []
for loc in atp_bet_data['Location'].unique():
    location = geocode(loc)
    country = location.address.split(', ')[-1]
    countries.append((loc, country))
    
x=pd.DataFrame(countries)
countries = x.set_index(0, drop=True)
atp_bet_data['Country'] = atp_bet_data['Location'].apply(lambda x: countries.loc[x][1])



"""
# Get elo, dob and age
elo_winners, elo_losers = scrape_elo()
atp_bet_data['WEloHardRank'] = elo_winners
atp_bet_data['LEloHardRank'] = elo_losers

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
atp_bet_data.to_excel('ATP_dataset_13_14.xlsx')

# Get fixed player stats
player_fixed_stats = get_player_fixed_stats()

# Save new player dataset to Excel
player_fixed_stats.to_excel('ATP_player_dataset.xlsx')
"""



"""
# One hot encode surface advantage (1 if player has surface advantage, 0 otherwise)
winner_surf_adv, loser_surf_adv = [], []
for ind, row in data.iterrows():
    try:
        if row['Surface'] in player_fixed_stats[player_fixed_stats['Name']==row['Winner']]['FavSurface'].iloc[0]:
            winner_surf_adv.append(1)
        else:
            winner_surf_adv.append(0)
    except:
        winner_surf_adv.append(0)
    try:
        if row['Surface'] in player_fixed_stats[player_fixed_stats['Name']==row['Loser']]['FavSurface'].iloc[0]:
            loser_surf_adv.append(1)
        else:
            loser_surf_adv.append(0)
    except:
        loser_surf_adv.append(0)

data['WSurfAdv'] = winner_surf_adv
data['LSurfAdv'] = loser_surf_adv
"""

"""
ages = []
for ind, row in player_fixed_stats.iterrows():
    try:
        player_id = int(row['PlayerId'])
        r=requests.get(f'https://www.ultimatetennisstatistics.com/playerProfileTab?playerId={player_id}')
        x=pd.read_html(r.text)[0]
        x.set_index(0, drop=True, inplace=True)
        try:
            age = x.loc['Age'].iloc[0]
        except:
            age = None

    except:
        age = None
    ages.append(age)

player_fixed_stats['Age'] = ages
"""

winner_ages = []
loser_ages = []
for ind, row in atp_bet_data.iterrows():
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
atp_bet_data['WAge']=winner_ages
atp_bet_data['LAge']=loser_ages



# Get tournament success -  win percentage over previous 5 years
winner_df = pd.DataFrame(atp_bet_data['Winner'].values, columns=['Name'])
winner_df['Tournament'] = atp_bet_data['Tournament'].values
winner_df['Id'] = atp_bet_data['TournamentId']
winner_df['Win'] = 1

loser_df = pd.DataFrame(atp_bet_data['Loser'].values, columns=['Name'])
loser_df['Tournament'] = atp_bet_data['Tournament'].values
loser_df['Id'] = atp_bet_data['TournamentId']
loser_df['Win'] = 0

match_hist = winner_df.append(loser_df).sort_index(kind='mergesort') # stable

tourney_wins_each_year = match_hist.groupby(['Name','Tournament','Id']).sum()
tourney_matches_each_year = match_hist.groupby(['Name','Tournament','Id']).count()

tourney_success_winner = []
tourney_success_loser = []

for ind, row in atp_bet_data.iterrows():
     year = row['TournamentId']
     wins_winner = (tourney_wins_each_year.loc[row['Winner'], row['Tournament'], :year-1].iloc[-5:].sum()/tourney_matches_each_year.loc[row['Winner'], row['Tournament'], :year-1].iloc[-5:].sum()).iloc[0]
     tourney_success_winner.append(wins_winner)
     wins_loser = (tourney_wins_each_year.loc[row['Loser'], row['Tournament'], :year-1].iloc[-5:].sum()/tourney_matches_each_year.loc[row['Loser'], row['Tournament'], :year-1].iloc[-5:].sum()).iloc[0]
     tourney_success_loser.append(wins_loser)
atp_bet_data['WTourneySuccess'] = tourney_success_winner
atp_bet_data['LTourneySuccess'] = tourney_success_loser



# Label encode surface advantage (1 if player has surface advantage, 0 otherwise)
winner_surf_adv, loser_surf_adv = [], []
for ind, row in atp_bet_data.iterrows():
    try:
        if row['Surface'] in player_fixed_stats[player_fixed_stats['Name']==row['Winner']]['FavSurface'].iloc[0]:
            winner_surf_adv.append(1)
        else:
            winner_surf_adv.append(0)
    except:
        winner_surf_adv.append(0)
    try:
        if row['Surface'] in player_fixed_stats[player_fixed_stats['Name']==row['Loser']]['FavSurface'].iloc[0]:
            loser_surf_adv.append(1)
        else:
            loser_surf_adv.append(0)
    except:
        loser_surf_adv.append(0)
atp_bet_data['WSurfAdv'] = winner_surf_adv
atp_bet_data['LSurfAdv'] = loser_surf_adv
  

# Get recent form - win percentage over previous 2 tournaments prior to current tournament
winner_df = pd.DataFrame(atp_bet_data['Winner'].values, columns=['Name'])
winner_df['Id'] = atp_bet_data['TournamentId'].values
winner_df['Win'] = 1

loser_df = pd.DataFrame(atp_bet_data['Loser'].values, columns=['Name'])
loser_df['Id'] = atp_bet_data['TournamentId'].values
loser_df['Win'] = 0

match_hist = winner_df.append(loser_df).sort_index(kind='mergesort') # stable

match_wins = match_hist.groupby(['Name','Id']).sum()
matches_played = match_hist.groupby(['Name','Id']).count()

form_winner = []
form_loser = []

for ind, row in atp_bet_data.iterrows():
     year = row['TournamentId']
     wins_winner = (match_wins.loc[row['Winner']].loc[:year-1].iloc[-2:].sum()/matches_played.loc[row['Winner']].loc[:year-1].iloc[-2:].sum()).iloc[0]
     form_winner.append(wins_winner)
     wins_loser = (match_wins.loc[row['Loser']].loc[:year-1].iloc[-2:].sum()/matches_played.loc[row['Loser']].loc[:year-1].iloc[-2:].sum()).iloc[0]
     form_loser.append(wins_loser)
atp_bet_data['WForm'] = form_winner
atp_bet_data['LForm'] = form_loser

