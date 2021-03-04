import pandas as pd
from datetime import datetime, timedelta
from timeit import default_timer as timer

from scrape import get_json
from h2h import get_h2h

# Data sources
# https://github.com/JeffSackmann/tennis_atp
# https://www.ultimatetennisstatistics.com/eloRatings
# http://www.tennis-data.co.uk/alldata.php


# elo = pd.read_csv('atpdata/test_elo_ratings.csv')
# elo.loc[elo['name'].str.contains('Wawrinka'),'name']='Stanislas Wawrinka'

match_hist = pd.read_csv('atp_master/ATP.csv')
"""
players = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date=&_=1578783179362")
def get_num_matches(players):
    players['won'], players['lost'], players['matches'] = 0, 0, 0
    for index, row in players.iterrows():
        try:
            # players.at[index,'won'] += match_hist['winner_name'].value_counts()[row['name']]
            # players.at[index,'lost'] += match_hist['loser_name'].value_counts()[row['name']]
            players.at[index,'won'] += match_hist[match_hist['tourney_date']<='20200106']['winner_name'].value_counts()[row['name']]
            players.at[index,'lost'] += match_hist[match_hist['tourney_date']<='20200106']['loser_name'].value_counts()[row['name']]
        except:
            pass
    players['matches'] = players['won'] + players['lost']
    # players.to_csv('atpdata/elo_ratings_current.csv')
    players.to_csv('atpdata/current2_elo_ratings.csv')
get_num_matches(players)
"""
elo = pd.read_csv('atpdata/test3_elo_ratings.csv')
elo.loc[elo['name'].str.contains('Wawrinka'),'name']='Stanislas Wawrinka'

aces = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=100&sort%5Bvalue%5D=desc&searchPhrase=&category=acePct&season=-1&fromDate=&toDate=&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578621981846")
first_serve = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=firstServePct&season=-1&fromDate=&toDate=&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578641519820")
upsets_scored = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=upsetsScoredPct&season=&fromDate=&toDate=&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578626448297")
upsets_against = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=upsetsAgainstPct&season=&fromDate=&toDate=&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578626448300")
# test_elo = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=ELO_RANK&season=&date=&_=1578626200145")
rankings = get_json("https://www.ultimatetennisstatistics.com/rankingsTableTable?current=1&rowCount=-1&searchPhrase=&rankType=RANK&season=&date=&_=1578782369780")
win_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate=&toDate=&level=&bestOf=&surface=&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305719")

grass_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate=&toDate=&level=&bestOf=&surface=G&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305720")
hard_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate=&toDate=&level=&bestOf=&surface=H&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305721")
clay_perc = get_json("https://www.ultimatetennisstatistics.com/statsLeadersTable?current=1&rowCount=-1&sort%5Bvalue%5D=desc&searchPhrase=&category=matchesWonPct&season=-1&fromDate=&toDate=&level=&bestOf=&surface=C&indoor=&speed=&round=&result=&tournamentId=&opponent=&countryId=&minEntries=&_=1578783305722")


atp_bet_data_2019 = pd.read_excel('atpdata/bet_data_2019.xlsx')
ausopen_bet_data_2019 = atp_bet_data_2019[atp_bet_data_2019['Tournament'].str.contains('French')]


def reg_model(elo_prob, surf_rec, form, ace_perc, first_serve_perc, upsets):
    
    # Regression model features
    # 1. Elo rating (0.5)
    # 2. Win/loss record on surface type in last year (0.1)
    # 3. Form - win/loss record in last 3 months (0.1)
    # 4. Win/loss record against that opponent in last year (0.3)
    # (0.5*elo_prob, 0.1*surf_rec, 0.1*form, 0.3*opp_hist)
    #0.4*elo_prob + 0.1*surf_rec + 0.1*form + 0.3*opp_hist + 0.0*ace_perc + 0.0*upsets
    #print(0.4*elo_prob, 0.2*surf_rec, 0.3*form, 0.1*opp_hist, 0.1*ace_perc, 0.2*first_serve_perc, 1*upsets)
    return (0.4*elo_prob + 0.2*surf_rec + 0.2*form + 0.1*ace_perc + 0.2*first_serve_perc + 1*upsets)


def elo_pwin(elo, player_name, opp_name, curr_date, surface):    
    """player_surname, player_first_let = player_name.split(' ')[-1], player_name.split(' ')[0][0]
    player = elo[(elo['name'].str.contains(player_surname)) & (elo['name'].str[0]==player_first_let)]
    opp_surname, opp_first_let = opp_name.split(' ')[-1], opp_name.split(' ')[0][0]
    opp = elo[(elo['name'].str.contains(opp_surname)) & (elo['name'].str[0]==opp_first_let)]"""
    # player_surname, player_first_name = player_name.replace(',','').split(' ')[-1], player_name.replace(',','').replace('-',' ').split(' ')[0][0]
    player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-1], player_name[0]
    player = elo[(elo['name'].str.split().str[-1]==player_surname) & (elo['name'].str.contains(player_first_name))]
    # opp_surname, opp_first_name = opp_name.replace(',','').split(' ')[-1], opp_name.replace(',','').replace('-',' ').split(' ')[0][0]
    opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-1], opp_name[0]
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

        """time_filter = (curr_date-timedelta(days=90)).strftime('%Y%m%d')
        
        player_matches = match_hist[((match_hist['winner_name'].str.split().str[-1]==player_surname)&(match_hist['winner_name'].str.contains(player_first_name))) | ((match_hist['loser_name'].str.split().str[-1]==player_surname) & (match_hist['loser_name'].str.contains(player_first_name)))]
        last_3_mths = player_matches[(time_filter<=player_matches['tourney_date'])&(player_matches['tourney_date']<curr_date.strftime('%Y%m%d'))]
        
        if last_3_mths.empty:
            form = 0
        else:
            last_3_mths_won = len(last_3_mths[(last_3_mths['winner_name'].str.split().str[-1]==player_surname) & (last_3_mths['winner_name'].str.contains(player_first_name))])
            form = last_3_mths_won/len(last_3_mths)
        """
        
        win = win_perc[(win_perc['name'].str.split().str[-1]==player_surname)&(win_perc['name'].str.contains(player_first_name))]
        if win.empty:
            form = 0
        else:
            form = win['value'].iloc[0]
            form = float(form[:-1])/100
            
        """
        time_filter = (curr_date-timedelta(days=365)).strftime('%Y%m%d')
        
        last_year = player_matches[(time_filter<=player_matches['tourney_date'])&(player_matches['tourney_date']<curr_date.strftime('%Y%m%d'))]

        # Surface win % over last year
        # Surface options: 'Hard', 'Clay', 'Grass'
        surface = match_hist[match_hist['tourney_id']=='2019-540'].iloc[0]['surface']
        surf_tot = last_year[last_year['surface']==surface]
        
        if surf_tot.empty:
            surf_rec = 0
        else:
            surf_won = surf_tot[(surf_tot['winner_name'].str.split().str[-1]==player_surname)&(surf_tot['winner_name'].str.contains(player_first_name))]
            surf_rec = len(surf_won)/len(surf_tot)    
        """
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
        

        # Ace % in past 2 years
        ace = aces[(aces['name'].str.split().str[-1]==player_surname)&(aces['name'].str.contains(player_first_name))]
        
        if ace.empty:
            ace_perc = 0
        else:
            ace_perc = ace['value'].iloc[0]
            ace_perc = float(ace_perc[:-1])/100
            
        # First serve % in past 2 years
        first_serves = first_serve[(first_serve['name'].str.split().str[-1]==player_surname)&(first_serve['name'].str.contains(player_first_name))]
        
        if first_serves.empty:
            first_serve_perc = 0
        else:
            first_serve_perc = first_serves['value'].iloc[0]
            first_serve_perc = float(first_serve_perc[:-1])/100
        
        # Upset scored % in past 2 years
        if elo_player > elo_opp:
            upset_score = upsets_scored[(upsets_scored['name'].str.split().str[-1]==player_surname)&(upsets_scored['name'].str.contains(player_first_name))]
            if upset_score.empty:
                upsets = 0
            else:
                upsets = upset_score['value'].iloc[0]
                upsets = float(upsets[:-1])/100
        else:
            # upsets = 0
            upset_against = upsets_against[(upsets_against['name'].str.split().str[-1]==player_surname)&(upsets_against['name'].str.contains(player_first_name))]
            if upset_against.empty:
                upsets = 0
            else:
                upsets = upset_against['value'].iloc[0]
                upsets = -float(upsets[:-1])/100
                
                
        
        # Get probability of player winning from regression model
        pwin = reg_model(elo_prob, surf_rec, form, ace_perc, first_serve_perc, upsets)
        
        # ATP ranking
        ranking_player = rankings[(rankings['name'].str.split().str[-1]==player_surname)&(rankings['name'].str.contains(player_first_name))]
        ranking_opp = rankings[(rankings['name'].str.split().str[-1]==opp_surname)&(rankings['name'].str.contains(opp_first_name))]

        """if ranking_player.empty:
            rank_player = 0
        else:
            rank_player = ranking_player.index+1
        if ranking_opp.empty:
            rank_opp = 0
        else:
            rank_opp = ranking_opp.index+1
        
        if rank_player - rank_opp >= 20:
            pwin += 0.2
        elif rank_opp - rank_player >= 20:
            pwin -= 0.2"""
        
        # Head-to-head %
        if ranking_player.empty or ranking_opp.empty:
            hth_perc = 0
        else:
            player_id = ranking_player['playerId'].iloc[0]
            opp_id = ranking_opp['playerId'].iloc[0]
            hth_perc = get_h2h(player_id, opp_id)
            
        #hth_perc = 0
        pwin += 0.1*hth_perc
        
    return pwin, elo



# Update win/loss and no. matches played, provided player has an elo_rating
# outcome: lose = 0 or win = 1
def update_elo(elo, winner_name, loser_name):

    elo_prob, elo, player, opp = elo_pwin(elo, winner_name, loser_name)
    
    # K-factor
    c = 250
    matches = player['matches']
    offset = 5
    shape = 0.4
    K = c/((matches+offset)**shape)
    
    if len(player) > 0:
        elo.loc[elo['Unnamed: 0'].values==player['Unnamed: 0'].values,'points'] += int(round(K*(0-elo_prob)))
        elo.loc[elo['Unnamed: 0'].values==player['Unnamed: 0'].values,'won'] += 1
        elo.loc[elo['Unnamed: 0'].values==player['Unnamed: 0'].values,'matches'] += 1
    if len(opp) > 0:
        elo.loc[elo['Unnamed: 0'].values==opp['Unnamed: 0'].values,'points'] += int(round(K*(1-elo_prob)))
        elo.loc[elo['Unnamed: 0'].values==opp['Unnamed: 0'].values,'lost'] += 1
        elo.loc[elo['Unnamed: 0'].values==opp['Unnamed: 0'].values,'matches'] += 1
    
    # Save updated elo ratings
    elo.to_csv('atpdata/elo_ratings_updated.csv')
    return elo




# Algorithm backtesting 
# Aus Open 2019 (tourney_id = '2019-580'): 78.74% accuracy for 0.7 elo, 0.1 surf 0.1 form, 0.1 hth
# French Open 2018 (tourney_id = '2019-520': 79.53% accuracy for 0.7 elo, 0.1 surf, 0.05 form, 0.05 hth
# Wimbledon 2018 (tourney_id = '2019-540': 77.17% accuracy for 0.6 elo, 0.3 surf, 0.1 form, 0.1 hth
# US Open 2018 (tourney_id = '2019-560': 77.17% accuracy for 0.7 elo, 0.1 surf 0.1 form, 0.1 hth
"""
# NB: Brisbane International (tourney_id = '2019-M020')
start = timer()

num_correct_pred = 0
profit = 0
stake = 5
total_staked = 0
total_loss = 0
win, loss = 0, 0
ausopen_2019_matches = match_hist[match_hist['tourney_id']=='2019-520']
surface = 'Hard'

for index, row in ausopen_2019_matches.iterrows():
    player_name = row['loser_name']
    opp_name = row['winner_name']
    
    bet_data_match = ausopen_bet_data_2019[(ausopen_bet_data_2019['Winner'].apply(lambda x: x.split(' ')[0] in opp_name)) & (ausopen_bet_data_2019['Loser'].apply(lambda x: x.split(' ')[0] in player_name))]
    odds_player = bet_data_match['MaxL']
    odds_opp = bet_data_match['MaxW']
    
    curr_date = datetime.strptime(row['tourney_date'],'%Y%m%d')
    prob_player_win, elo = elo_pwin(elo, player_name, opp_name, curr_date, surface)
    prob_opp_win, elo = elo_pwin(elo, opp_name, player_name, curr_date, surface)
    
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
            print(val_opp_win)
            if 2 <= val_opp_win <= 3:
                profit += odds_opp.iloc[0]*stake - stake
                total_staked += stake
                win += 1
                print(opp_name, player_name)
    
    # Incorrectly predict player to win
    else:
        if not odds_player.empty:
            val_player_win = round(odds_player.iloc[0]*prob_player_win, 2)
            print(val_player_win)
            if 2 <= val_player_win <= 3:
                profit -= stake
                total_staked += stake
                total_loss -= stake
                loss += 1
                
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