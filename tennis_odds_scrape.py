import pandas as pd
import requests
import urllib
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options


def get_surface(string):
    if 'hard' in string.lower():
        return 'Hard'
    elif 'grass' in string.lower():
        return 'Grass'
    elif 'clay' in string.lower():
        return 'Clay'
    else:
        return None
    
def scrape_tennis_odds(date):
    # Today's matches
    url = f"https://www.oddsportal.com/matches/tennis/"+date.strftime('%Y%m%d')
    
    options = Options()
    options.headless=True
    browser = webdriver.Firefox(options=options, executable_path='/usr/local/bin/geckodriver')
    browser.get(url)
    table = pd.read_html(browser.page_source)[0]
    browser.quit()

    #table.drop(columns=table.columns[[1,2,3,6]], inplace=True)
    new = pd.DataFrame(data=[table.columns])
    table.columns = list(range(len(table.columns)))
    tennis_odds = new.append(table, ignore_index=True)
    
    tourney = None
    tourney_series = []
    
    for ind, row in tennis_odds.iterrows():
        if type(row[0]) == str and len(row[0]) > 7:
            tourney = row[0]
            tourney_series.append(None)
        else:
            tourney_series.append(tourney)

    tennis_odds['Tournament'] = tourney_series
    tennis_odds.drop(columns=[2,3,6], inplace=True)
    tennis_odds.dropna(inplace=True)
    
    # Filter upcoming men's matches
    tennis_odds = tennis_odds[(tennis_odds[0].str.contains(':'))&((tennis_odds['Tournament'].str.contains('Men'))|(tennis_odds['Tournament'].str.contains('ATP')))]
    
    # Reformat columns and index
    tennis_odds['P1'] = tennis_odds[1].apply(lambda x: x.split(' - ')[0])
    tennis_odds['P1'] = tennis_odds['P1'].apply(lambda x: ' '.join(x.split()[:-1]) + ' ' + x.split()[-1].replace('-', '.'))
    tennis_odds['P2'] = tennis_odds[1].apply(lambda x: x.split(' - ')[-1])
    tennis_odds['P2'] = tennis_odds['P2'].apply(lambda x: ' '.join(x.split()[:-1]) + ' ' + x.split()[-1].replace('-', '.'))
    tennis_odds['Surface'] = tennis_odds['Tournament'].apply(lambda x: get_surface(x))
    tennis_odds.drop(columns=[1], inplace=True)
    tennis_odds.rename(columns={0: 'Time', 4: 'P1 Odds', 5: 'P2 Odds'}, inplace=True)
    tennis_odds.reset_index(inplace=True, drop=True)
    
    return tennis_odds




