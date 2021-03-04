import pandas as pd
import numpy as np
from tennis_odds_scrape import scrape_tennis_odds
from scrape import get_json
from datetime import datetime, timedelta
from dateutil.tz import tzlocal

# Get fixed player statistics
player_fixed_stats = pd.read_excel('ATP_player_dataset.xlsx')


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

# Get current UTC time, today's date and tomorrow's date
curr_utc = datetime.utcnow()
today = curr_utc.date()
time = curr_utc.time()
tom = today+timedelta(days=1)

# Scrape odds
odds_today = scrape_tennis_odds(today)

# Filter upcoming matches and convert back to local time
odds_today['Time'] = odds_today['Time'].apply(lambda x: datetime.strptime(x + ' ' + today.strftime('%Y%m%d'), '%H:%M %Y%m%d'))
try:
    odds_tom = scrape_tennis_odds(tom)
    odds_tom['Time'] = odds_tom['Time'].apply(lambda x: datetime.strptime(x + ' ' + tom.strftime('%Y%m%d'), '%H:%M %Y%m%d'))
    odds = odds_today[odds_today['Time']>curr_utc].append(odds_tom)
except:
    odds = odds_today[odds_today['Time']>curr_utc]
odds['Time'] = odds['Time'].apply(lambda x: x.to_pydatetime() + x.to_pydatetime().astimezone(tzlocal()).utcoffset())

odds = odds[(odds['P1 Odds'] != '-')&(odds['P2 Odds'] != '-')]
odds['P1 Odds'] = odds['P1 Odds'].astype('float')
odds['P2 Odds'] = odds['P2 Odds'].astype('float')
odds.reset_index(drop=True, inplace=True)

# Filter ATP singles matches
odds = odds[(odds['Tournament'].str.contains('ATP'))&(odds['P1'].apply(lambda x: '/' not in x))]


def get_elo(elo, player_first_name, player_surname, opp_first_name, opp_surname):
    """
    if player_name[-1] == ' ':
        player_name = player_name[:-1]
    player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-2], player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    if opp_name[-1] == ' ':
        opp_name = opp_name[:-1]
    opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-2], opp_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    """
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


def get_features(player_surname, player_first_name, opp_surname, opp_first_name, player1_odds, player2_odds, player1_stats, player2_stats, surface, tourney):
    elo_curr = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&sort%5Brank%5D=asc&searchPhrase=&rankType=ELO_RANK&season=&date=&_=1595985630050")
    player1_elo, player2_elo = get_elo(elo_curr, player_first_name, player_surname, opp_first_name, opp_surname)
    try:
        elo_diff = np.log(player1_elo)-np.log(player2_elo)
    except:
        return None
    
    odds_diff = player1_odds-player2_odds
    
    player1_age = round(((datetime.today()-player1_stats['DateOfBirth'])/365).days,2)
    player2_age = round(((datetime.today()-player2_stats['DateOfBirth'])/365).days,2)
    age_diff = player1_age-player2_age
    
    player1_height = player1_stats['Height']
    player2_height = player2_stats['Height']
    if np.isnan(player1_height):
         player1_height = player_fixed_stats['Height'].mean()
    if np.isnan(player2_height):
         player2_height = player_fixed_stats['Height'].mean()
    height_diff = player1_height-player2_height
        
    try:
        player1_form = (match_wins.loc[player1].iloc[-2:].sum()/matches_played.loc[player1].iloc[-2:].sum()).iloc[0]
    except:
        player1_form = 0.5
    try:
        player2_form = (match_wins.loc[player2].iloc[-2:].sum()/matches_played.loc[player2].iloc[-2:].sum()).iloc[0]
    except:
        player2_form = 0.5
    if player1_form is None:
       player1_form = 0.5
    if player2_form is None:
       player2_form = 0.5
    form_diff = player1_form-player2_form
    
    # If surface unkown, assume hard
    if surface is None:
        surface = 'Hard'
    elo_surf_curr = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&sort%5Brank%5D=asc&searchPhrase=&rankType={surface.upper()}_ELO_RANK&season=&date=&_=1595987779542")
    player1_surf_elo, player2_surf_elo = get_elo(elo_surf_curr, player_first_name, player_surname, opp_first_name, opp_surname)
    try:
        surf_elo_diff = np.log(player1_surf_elo)-np.log(player2_surf_elo)
    except:
        return None
        
    elo_ret_curr = get_json(f"https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=RETURN_GAME_ELO_RANK&season=&date=&_=15786262001450")
    player1_ret_elo, player2_ret_elo = get_elo(elo_ret_curr, player_first_name, player_surname, opp_first_name, opp_surname)
    try:
        ret_elo_diff = np.log(player1_ret_elo)-np.log(player2_ret_elo)
    except:
        return None
    
    if player1_stats['Country'] in tourney:
        player1_home_adv = 1
    else:
        player1_home_adv = 0
    if player2_stats['Country'] in tourney:
        player2_home_adv = 1
    else:
        player2_home_adv = 0
    home_adv_diff = player1_home_adv-player2_home_adv
    
    if player1_stats['Plays'] == 'Right-handed':
        player1_hand = 1
    else:
        player1_hand = 0
    if player2_stats['Plays'] == 'Right-handed':
        player2_hand = 1
    else:
        player2_hand = 0
    plays_diff = player1_hand-player2_hand
    
    return elo_diff, odds_diff, age_diff, height_diff, form_diff, surf_elo_diff, ret_elo_diff, home_adv_diff, plays_diff


    
# Get feature set for valid matches
X_pred = pd.DataFrame(columns=range(len(X_full.columns)))

inds = []
for ind, row in odds.iterrows():
    player_name, opp_name = row['P1'], row['P2']
    if player_name[-1] == ' ':
        player_name = player_name[:-1]
    player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-2], player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    if opp_name[-1] == ' ':
        opp_name = opp_name[:-1]
    opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-2], opp_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    
    player1_stats = player_fixed_stats[(player_fixed_stats['Name'].str.contains(player_surname)) & (player_fixed_stats['Name'].str.contains(player_first_name))]
    player2_stats = player_fixed_stats[(player_fixed_stats['Name'].str.contains(opp_surname)) & (player_fixed_stats['Name'].str.contains(opp_first_name))]

    if len(player1_stats) > 0 and len(player2_stats) > 0:
        features = get_features(player_surname, player_first_name, opp_surname, opp_first_name, row['P1 Odds'], row['P2 Odds'], player1_stats.iloc[0], player2_stats.iloc[0], row['Surface'], row['Tournament'])
        if features is not None:
            X_pred = X_pred.append(pd.Series(features), ignore_index=True)
            inds.append(ind)
        else: # at least one player has missing elo
            pass
    else: # at least one player not in database
        pass

X_pred.columns = X_full.columns


# Get predictions
y_prob = [val[1] for val in pipe.predict_proba(X_pred)]
y_pred = [round(value) for value in y_prob]

stake = 5
profit = 0
i = 0
for ind, row in odds.loc[inds].iterrows():
    player_name, opp_name = row['P1'], row['P2']
    if player_name[-1] == ' ':
        player_name = player_name[:-1]
    player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-2], player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    if opp_name[-1] == ' ':
        opp_name = opp_name[:-1]
    opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-2], opp_name.replace('-',' ').replace(',','').split(' ')[-1][0]
    
    player1_stats = player_fixed_stats[(player_fixed_stats['Name'].str.contains(player_surname)) & (player_fixed_stats['Name'].str.contains(player_first_name))]
    player2_stats = player_fixed_stats[(player_fixed_stats['Name'].str.contains(opp_surname)) & (player_fixed_stats['Name'].str.contains(opp_first_name))]
    
    if len(player1_stats) > 0 and len(player2_stats) > 0:
        if y_pred[i] == 1 and 0 < y_prob[i]*row['P1 Odds'] and row['P1 Odds'] > 1:
            player1_odds = row['P1 Odds']
            profit += (player1_odds-1)*stake
            print(f"{row['P1']} to beat {row['P2']}")
            print(f"Bet value: {round(y_prob[i]*player1_odds, 4)}. Prob: {round(y_prob[i], 4)}")
            print(f"Stake ${stake} to win ${round((player1_odds-1)*stake, 2)}")
            print()
        elif y_pred[i] == 0 and 0 < (1-y_prob[i])*row['P2 Odds'] and row['P2 Odds'] > 1:
            player2_odds = row['P2 Odds']
            profit += (player2_odds-1)*stake
            print(f"{row['P2']} to beat {row['P1']}")
            print(f"Bet value: {round((1-y_prob[i])*player2_odds, 4)}. Prob: {round(1-y_prob[i], 4)}")
            print(f"Stake ${stake} to win ${round((player2_odds-1)*stake, 2)}")
            print()
        i += 1
    
