import pandas as pd
from datetime import datetime, timedelta
from timeit import default_timer as timer
import requests, zipfile, io
import os


from bs4 import BeautifulSoup
from scrape import get_json
from h2h import get_rec_h2h

# Data sources
# https://github.com/JeffSackmann/tennis_atp
# https://www.ultimatetennisstatistics.com/eloRatings
# http://www.tennis-data.co.uk/alldata.php


# elo = pd.read_csv('atpdata/test_elo_ratings.csv')
# elo.loc[elo['name'].str.contains('Wawrinka'),'name']='Stanislas Wawrinka'

match_hist = pd.read_csv('atp_master/ATP.csv')
"""
players = pd.read_csv('atpdata/current_elo_ratings.csv')
def get_num_matches(players):
    players['won'], players['lost'], players['matches'] = 0, 0, 0
    for index, row in players.iterrows():a
        try:
            # players.at[index,'won'] += match_hist['winner_name'].value_counts()[row['name']]
            # players.at[index,'lost'] += match_hist['loser_name'].value_counts()[row['name']]
            players.at[index,'won'] += match_hist[match_hist['tourney_date']<='20200106']['winner_name'].value_counts()[row['name']]
            players.at[index,'lost'] += match_hist[match_hist['tourney_date']<='20200106']['loser_name'].value_counts()[row['name']]
        except:
            pass
    players['matches'] = players['won'] + players['lost']
    # players.to_csv('atpdata/elo_ratings_current.csv')
    players.to_csv('atpdata/current_elo_ratings.csv')
get_num_matches(players)
"""


atp_bet_data_2019 = pd.read_excel('atpdata/bet_data_2019.xlsx')

r = requests.get("http://www.tennis-data.co.uk/2020/2020.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
os.rename('2020.xlsx','atpdata/bet_data_2020.xlsx')
atp_bet_data_2020 = pd.read_excel('atpdata/bet_data_2020.xlsx')

atp_bet_data = atp_bet_data_2019.append(atp_bet_data_2020)

#ausopen_bet_data_2019 = atp_bet_data_2019[atp_bet_data_2019['Tournament'].str.contains('Australian')]
ausopen_bet_data_2019 = atp_bet_data[atp_bet_data['Date']>'2019-01-01']
#ausopen_2019_matches = match_hist[(match_hist['tourney_date']>'20190113')&(match_hist['tourney_date']<='20190131')]
#ausopen_2019_matches = match_hist[match_hist['tourney_id']=='2019-580']

"""ausopen_2019_matches = atp_bet_data[(atp_bet_data['Date']>datetime.strptime('20200101','%Y%m%d'))&(atp_bet_data['Date']<=datetime.today())]
ausopen_2019_matches.index=range(len(ausopen_2019_matches))"""




curr_date = datetime.today()
#curr_date = datetime.strptime('2019-06-10', '%Y-%m-%d')
from_date = (curr_date-timedelta(days=90)).strftime('%d-%m-%Y')
to_date = curr_date.strftime('%d-%m-%Y')


time_filter = (curr_date-timedelta(days=60))
ausopen_2019_matches = atp_bet_data[(atp_bet_data['Date']>time_filter)&(atp_bet_data['Date']<curr_date)]
ausopen_2019_matches.index=range(len(ausopen_2019_matches))

elo = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date=&_=1578626200145")
elo.loc[elo['name'].str.contains('Wawrinka'),'name']='Stanislas Wawrinka'

from_date = (time_filter-timedelta(days=90)).strftime('%d-%m-%Y')
to_date = time_filter.strftime('%d-%m-%Y')
#win_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate={0}&toDate={1}&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305719".format(from_date, to_date))
win_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=20&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=&fromDate={0}&toDate={1}&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1579235798414".format(from_date, to_date))
from_date = (curr_date-timedelta(days=365)).strftime('%d-%m-%Y')
from_date = (time_filter-timedelta(days=365)).strftime('%d-%m-%Y')

rankings = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=RANK&season=&date=&_=1578782369780")
grass_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate={0}&toDate={1}&level=&bestOf=&surface=G&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305720".format(from_date, to_date))
hard_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate={0}&toDate={1}&level=&bestOf=&surface=H&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305721".format(from_date, to_date))
clay_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate={0}&toDate={1}&level=&bestOf=&surface=C&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305722".format(from_date, to_date))


def reg_model(elo_prob, surf_rec, form, opp_hist):
    
    # Regression model features
    # 1. Elo rating (0.5)
    # 2. Win/loss record on surface type in last year (0.1)
    # 3. Form - win/loss record in last 3 months (0.1)
    # 4. Win/loss record against that opponent in last year (0.3)
    # (0.5*elo_prob, 0.1*surf_rec, 0.1*form, 0.3*opp_hist)
    return (0.4*elo_prob + 0.1*surf_rec + 0.2*form + 0.3*opp_hist)


def elo_pwin(elo, player_name, opp_name, curr_date, surface, bet=False):    
    if bet:
        player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-1], player_name[0]
        player = elo[(elo['name'].str.split().str[-1]==player_surname) & (elo['name'].str.contains(player_first_name))]
        # opp_surname, opp_first_name = opp_name.replace(',','').split(' ')[-1], opp_name.replace(',','').replace('-',' ').split(' ')[0][0]
        opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-1], opp_name[0]
        opp = elo[(elo['name'].str.split().str[-1]==opp_surname) & (elo['name'].str.contains(opp_first_name))]
    else:
        if player_name[-1] == ' ':
            player_name = player_name[:-1]
        player_surname, player_first_name = ' '.join(player_name.replace('-',' ').replace(',','').split(' ')[:-1]), player_name.replace('-',' ').replace(',','').split(' ')[-1][0]
        player = elo[(elo['name'].str.split().str[-1]==player_surname) & (elo['name'].str.contains(player_first_name))]
        if opp_name[-1] == ' ':
            opp_name = opp_name[:-1]
        opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[0], opp_name.replace('-',' ').replace(',','').split(' ')[-1][0]
        opp = elo[(elo['name'].str.split().str[-1]==opp_surname) & (elo['name'].str.contains(opp_first_name))]

    # if player is not in elo_ratings (top 181 ranked), then predict opponent to win, and vice versa
    if len(player) == 0:
        pwin = 0
    elif len(opp) == 0:
        pwin = 1
    else:
        elo_player = player['points'].values[0]
        elo_opp = opp['points'].values[0]
         
        # Probability of player winning
        elo_prob = 1/(1+10**((elo_opp-elo_player)/400))

        # Form over last 3 months
        # tourney_date = match_hist[match_hist['tourney_id']=='2019-580'].iloc[0]['tourney_date']
        # tourney_date = datetime.strptime(str(tourney_date), '%Y%m%d')
        # time_filter = (tourney_date-timedelta(days=60)).strftime('%Y%m%d')


        win = win_perc[(win_perc['name'].str.split().str[-1]==player_surname)&(win_perc['name'].str.contains(player_first_name))]
        if win.empty:
            form = 0
        else:
            form = win['value'].iloc[0]
            form = float(form[:-1])/100

        # Surface win % over last year
        if surface == 'Grass':
            grass = grass_perc[(grass_perc['name'].str.split().str[-1]==player_surname)&(grass_perc['name'].str.contains(player_first_name))]
            if grass.empty:
                surf_rec = 0
            else:
                surf_rec = grass['value'].iloc[0]
                surf_rec = float(surf_rec[:-1])/100
        elif surface == 'Hard':
            hard = hard_perc[(hard_perc['name'].str.split().str[-1]==player_surname)&(hard_perc['name'].str.contains(player_first_name))]
            if hard.empty:
                surf_rec = 0
            else:
                surf_rec = hard['value'].iloc[0]
                surf_rec = float(surf_rec[:-1])/100
        else:
            clay = clay_perc[(clay_perc['name'].str.split().str[-1]==player_surname)&(clay_perc['name'].str.contains(player_first_name))]
            if clay.empty:
                surf_rec = 0
            else:
                surf_rec = clay['value'].iloc[0]
                surf_rec = float(surf_rec[:-1])/100


        ranking_player = rankings[(rankings['name'].str.split().str[-1]==player_surname)&(rankings['name'].str.contains(player_first_name))]
        ranking_opp = rankings[(rankings['name'].str.split().str[-1]==opp_surname)&(rankings['name'].str.contains(opp_first_name))]
        
        # Head-to-head win % over last year
        if ranking_player.empty or ranking_opp.empty:
            hth_perc = 0
        else:
            player_id = ranking_player['playerId'].iloc[0]
            opp_id = ranking_opp['playerId'].iloc[0]
            hth_perc = get_rec_h2h(player_id, opp_id, (curr_date-timedelta(days=365)).strftime('%d-%m-%Y'), (curr_date-timedelta(days=4)).strftime('%d-%m-%Y'), surface)
    
        # Get probability of player winning from regression model
        
        pwin = reg_model(elo_prob, surf_rec, form, hth_perc) 
        """
        if player['rank'].values[0] - opp['rank'].values[0] >= 20:
            pwin += 0.2
        elif opp['rank'].values[0] - player['rank'].values[0] >= 20:
            pwin -= 0.2"""
    return pwin, elo




# Algorithm backtesting 
# Aus Open 2019 (tourney_id = '2019-580'): 78.74% accuracy for 0.7 elo, 0.1 surf 0.1 form, 0.1 hth
# French Open 2018 (tourney_id = '2019-520': 79.53% accuracy for 0.7 elo, 0.1 surf, 0.05 form, 0.05 hth
# Wimbledon 2018 (tourney_id = '2019-540': 77.17% accuracy for 0.6 elo, 0.3 surf, 0.1 form, 0.1 hth
# US Open 2018 (tourney_id = '2019-560': 77.17% accuracy for 0.7 elo, 0.1 surf 0.1 form, 0.1 hth
"""
# NB: Brisbane International (tourney_id = '2019-580')
start = timer()

num_correct_pred = 0
profit = 0
stake = 5
total_staked = 0
total_loss = 0
win, loss = 0, 0
for index, row in ausopen_2019_matches.iterrows():
    player_name = row['Loser']
    opp_name = row['Winner']
    
    bet_data_match = ausopen_bet_data_2019[(ausopen_bet_data_2019['Winner'].apply(lambda x: x.split(' ')[0] in opp_name)) & (ausopen_bet_data_2019['Loser'].apply(lambda x: x.split(' ')[0] in player_name))]
    odds_player = bet_data_match['MaxL']
    odds_opp = bet_data_match['MaxW']
    
    curr_date = row['Date']
    prob_player_win, elo = elo_pwin(elo, player_name, opp_name, curr_date, 'Hard')
    prob_opp_win, elo = elo_pwin(elo, opp_name, player_name, curr_date, 'Hard')
    
    # Adjust probabilities to sum to 1
    try:
        prob_player_win, prob_opp_win = prob_player_win/(prob_player_win + prob_opp_win), prob_opp_win/(prob_player_win + prob_opp_win)
    except:
        pass
    
    # update_elo(elo, opp_name, player_name)
    
    # Probability of winning based on elo rating
    # elo_prob, elo = predict_winner_elo(elo, player_name, opp_name)
    
    # Surface options: 'Hard', 'Clay', 'Grass'
    # surface = match_hist[match_hist['tourney_id']=='2019-580'].iloc[0]['surface']
    # surf_tot = match_hist[(match_hist['winner_name']==player_name)&(match_hist['surface']==surface) | (match_hist['loser_name']==player_name)&(match_hist['surface']==surface)]
    
    # if surf_tot.empty:
        # surf_rec = 0
    # else:
        # surf_won = surf_tot[surf_tot['winner_name']==player_name]
        # surf_rec = len(surf_won)/len(surf_tot)
    
    # Form over last 3 months
    # tourney_date = match_hist[match_hist['tourney_id']=='2019-580'].iloc[0]['tourney_date']
    # tourney_date = datetime.strptime(str(tourney_date), '%Y%m%d')
    # time_filter = (tourney_date-timedelta(days=60)).strftime('%Y%m%d')
    # last_3_mths = match_hist[(match_hist['winner_name']==player_name)&(match_hist['tourney_date']>=int(time_filter)) | (match_hist['loser_name']==player_name)&(match_hist['tourney_date']>=int(time_filter))]
    
    # if last_3_mths.empty:
        # form = 0
    # else:
        # last_3_mths_won = len(last_3_mths[last_3_mths['winner_name']==player_name])
        # form = last_3_mths_won/len(last_3_mths)

    
    if prob_player_win >= prob_opp_win:
        pred = player_name
    else:
        pred = opp_name
                
    # Correctly predict opp to win
    if pred == opp_name:
        num_correct_pred += 1
        if not odds_opp.empty:
            val_opp_win = round(odds_opp.iloc[0]*prob_opp_win, 2)
            if 1.3 <= val_opp_win <= 2 and odds_opp.iloc[0] >= 1.5:
                profit += odds_opp.iloc[0]*stake - stake
                total_staked += stake
                win += 1
                print(opp_name, player_name)
    
    # Incorrectly predict player to win
    else:
        if not odds_player.empty:
            val_player_win = round(odds_player.iloc[0]*prob_player_win, 2)
            if 1.3 <= val_player_win <= 2 and odds_player.iloc[0] >= 1.5:
                profit -= stake
                total_staked += stake
                total_loss -= stake
                loss += 1
                print(opp_name, player_name)
    #print(prob_opp_win, 1/odds_opp.iloc[0])
    #print(prob_player_win, 1/odds_player.iloc[0])
    print(profit)
    
print("Accuracy: {:.2%}".format(num_correct_pred/len(ausopen_2019_matches)))

print('Profit: ${0}'.format(round(profit, 2)))
print('Value of lost bets: ${0}'.format(round(total_loss, 2)))

try:
    print('Total staked: ${0}'.format(round(total_staked, 2)))
    print('Return: {0}%'.format(round(profit/total_staked*100, 2)))
    print('Correct bets: {0}%'.format(round(win/(win+loss)*100), 2))

except:
    pass

print()
end = timer()
print(end - start)
"""