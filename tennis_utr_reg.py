import pandas as pd
from datetime import datetime, timedelta
from timeit import default_timer as timer
import requests, zipfile, io
import os

from scrape import get_json, get_utr

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
elo = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date=&_=1578626200145")
elo.loc[elo['name'].str.contains('Wawrinka'),'name']='Stanislas Wawrinka'

utr = get_utr("https://agw-prod.myutr.com/v2/player/top?gender=M&tags=Pro")
utr = utr.sort_values('threeMonthRating', ascending=False)
utr = utr.rename(columns={'displayName': 'name', 'threeMonthRating': 'points'})


atp_bet_data_2019 = pd.read_excel('atpdata/bet_data_2019.xlsx')

r = requests.get("http://www.tennis-data.co.uk/2020/2020.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
os.rename('2020.xlsx','atpdata/bet_data_2020.xlsx')
atp_bet_data_2020 = pd.read_excel('atpdata/bet_data_2020.xlsx')

atp_bet_data = atp_bet_data_2019.append(atp_bet_data_2020)

#ausopen_bet_data_2019 = atp_bet_data_2019[atp_bet_data_2019['Tournament'].str.contains('Australian')]
ausopen_bet_data_2019 = atp_bet_data[atp_bet_data['Date']>'20190113']
#ausopen_2019_matches = match_hist[(match_hist['tourney_date']>'20190113')&(match_hist['tourney_date']<='20190131')]
#ausopen_2019_matches = match_hist[match_hist['tourney_id']=='2019-580']

ausopen_2019_matches = atp_bet_data[(atp_bet_data['Date']>datetime.strptime('20200101','%Y%m%d'))&(atp_bet_data['Date']<=datetime.today())]
ausopen_2019_matches.index=range(len(ausopen_2019_matches))




def reg_model(elo_prob, utr_prob, surf_rec, form, opp_hist):
    
    # Regression model features
    # 1. Elo rating (0.5)
    # 2. Win/loss record on surface type in last year (0.1)
    # 3. Form - win/loss record in last 3 months (0.1)
    # 4. Win/loss record against that opponent in last year (0.3)
    #(0.4*elo_prob, 0.1*surf_rec, 0.2*form, 0.3*opp_hist)
    return (0.35*elo_prob + 0.25*utr_prob + 0.15*surf_rec + 0.15*form + 0.1*opp_hist)


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

        time_filter = (curr_date-timedelta(days=90)).strftime('%Y%m%d')
        
        player_matches = match_hist[((match_hist['winner_name'].str.split().str[-1]==player_surname)&(match_hist['winner_name'].str.contains(player_first_name))) | ((match_hist['loser_name'].str.split().str[-1]==player_surname) & (match_hist['loser_name'].str.contains(player_first_name)))]
        last_3_mths = player_matches[(time_filter<=player_matches['tourney_date'])&(player_matches['tourney_date']<curr_date.strftime('%Y%m%d'))]
        
        if last_3_mths.empty:
            form = 0
        else:
            last_3_mths_won = len(last_3_mths[(last_3_mths['winner_name'].str.split().str[-1]==player_surname) & (last_3_mths['winner_name'].str.contains(player_first_name))])
            form = last_3_mths_won/len(last_3_mths)
   
        time_filter = (curr_date-timedelta(days=365)).strftime('%Y%m%d')
        last_year = player_matches[(time_filter<=player_matches['tourney_date'])&(player_matches['tourney_date']<curr_date.strftime('%Y%m%d'))]
        
        # Surface win % over last year
        # Surface options: 'Hard', 'Clay', 'Grass'
        #surface = match_hist[match_hist['tourney_id']=='2019-580'].iloc[0]['surface']
        surf_tot = last_year[last_year['surface']==surface]
        
        if surf_tot.empty:
            surf_rec = 0
        else:
            surf_won = surf_tot[(surf_tot['winner_name'].str.split().str[-1]==player_surname)&(surf_tot['winner_name'].str.contains(player_first_name))]
            surf_rec = len(surf_won)/len(surf_tot)    
        
        
        # Head-to-head win % over last year
        # hth = last_year[((last_year['winner_name'].str.split().str[-1]==player_surname)&(last_year['loser_name'].str.split().str[-1]==opp_surname) | (last_year['loser_name'].str.split().str[-1]==player_surname)&(last_year['winner_name'].str.split().str[-1]==opp_surname))]
        
        # Head-to-head win % over last year on same surface
        hth = surf_tot[((surf_tot['winner_name'].str.split().str[-1]==player_surname)&(surf_tot['loser_name'].str.split().str[-1]==opp_surname) | (surf_tot['loser_name'].str.split().str[-1]==player_surname)&(surf_tot['winner_name'].str.split().str[-1]==opp_surname))]
        
        if hth.empty:
            hth_perc = 0
        else:
            hth_won = hth[(hth['winner_name'].str.split().str[-1]==player_surname)&(hth['winner_name'].str.contains(player_first_name))]
            hth_perc = len(hth_won)/len(hth)
        
        
        # UTR ranking
        if bet:
            player = utr[(utr['name'].str.split().str[-1]==player_surname) & (utr['name'].str.contains(player_first_name))]
            opp = utr[(utr['name'].str.split().str[-1]==opp_surname) & (utr['name'].str.contains(opp_first_name))]
        else:
            opp = utr[(elo['name'].str.split().str[-1]==opp_surname) & (utr['name'].str.contains(opp_first_name))]

        
        # Get probability of player winning from regression model
        
        if len(player) == 0:
            utr_prob = 0
        elif len(opp) == 0:
            utr_prob = 1
        else:
            utr_player = player['points'].values[0]
            utr_opp = opp['points'].values[0]
         
            # Probability of player winning
            utr_prob = 1/(1+10**((utr_opp-utr_player)/400))
        
        pwin = reg_model(elo_prob, utr_prob, surf_rec, form, hth_perc) 

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
            if 1.1 <= val_opp_win <= 2 and odds_opp.iloc[0] >= 1.5:
                profit += odds_opp.iloc[0]*stake - stake
                total_staked += stake
                win += 1
                print('WIN: Correctly bet {0} to beat {1}'.format(opp_name, player_name))
                print(val_opp_win)
        #print('My prob: {0}, market prob: {1}'.format(prob_opp_win, 1/odds_opp.iloc[0]))
    
    # Incorrectly predict player to win
    else:
        if not odds_player.empty:
            val_player_win = round(odds_player.iloc[0]*prob_player_win, 2)
            if 1.1 <= val_player_win <= 2 and odds_player.iloc[0] >= 1.5:
                profit -= stake
                total_staked += stake
                total_loss -= stake
                loss += 1
                print('LOSS: Incorrectly bet {0} to beat {1}'.format(player_name, opp_name))
                print(val_player_win)
        #print('My prob: {0}, market prob: {1}'.format(prob_player_win, 1/odds_player.iloc[0]))
    
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