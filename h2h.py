import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup



def get_h2h(player_id, opp_id):
    url = "https://www.ultimatetennisstatistics.com/h2hProfiles?playerId1={0}&playerId2={1}".format(str(player_id), str(opp_id))
    
    response = requests.get(url)
    
    page = response.text
    soup = BeautifulSoup(page, 'lxml')
    soup.prettify()
    
    table = soup.find(class_='table')
    
    hth_wins = table.find_all('tr')[1].text.split()
    player1_wins = int(hth_wins[0])
    player2_wins = int(hth_wins[2])
    try:
        player1_perc = player1_wins/(player1_wins + player2_wins)
    except:
        player1_perc = 0
    #player2_perc = 1 - player1_perc
    
    return player1_perc
    
"""
player_name = 'Rafael Nadal'
opp_name = 'Novak Djokovic'

player_surname, player_first_name = player_name.replace('-',' ').replace(',','').split(' ')[-1], player_name[0]
opp_surname, opp_first_name = opp_name.replace('-',' ').replace(',','').split(' ')[-1], opp_name[0]

ranking_player = rankings[(rankings['name'].str.split().str[-1]==player_surname)&(rankings['name'].str.contains(player_first_name))]
ranking_opp = rankings[(rankings['name'].str.split().str[-1]==opp_surname)&(rankings['name'].str.contains(opp_first_name))]

player_id = ranking_player['playerId'].iloc[0]
opp_id = ranking_opp['playerId'].iloc[0]
"""




def get_rec_h2h(player_id, opp_id, from_date, to_date, surface):
    url = "https://www.ultimatetennisstatistics.com/h2h?playerId1={0}&playerId2={1}&fromDate={2}&toDate={3}&surface={4}".format(str(player_id), str(opp_id), from_date, to_date, surface[0])
    response = requests.get(url)
    
    page = response.text
    soup = BeautifulSoup(page, 'lxml')
    soup.prettify()
    
    hth_wins = soup.text.split()[0][1:-1].split(',')
    player1_wins = int(hth_wins[0])
    player2_wins = int(hth_wins[1])
    
    try:
        player1_perc = player1_wins/(player1_wins + player2_wins)
    except:
        player1_perc = 0
    #player2_perc = 1 - player1_perc
    
    return player1_perc

