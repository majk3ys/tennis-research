import json
import pandas as pd
import requests


from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import csv


url = "https://www.oddschecker.com/au/tennis/challenger-tour"

def scrape_odds(url):
    response = requests.get(url)
    
    page = response.text
    soup = BeautifulSoup(page, 'lxml')
    soup.prettify()
    
    rows = soup.find_all(class_="_3f0k2k")
    
    df = pd.DataFrame(columns=['Home Team', 'Away Team', 'Home Odds', 'Away Odds'])
    
    for row in rows:
        teams = [team.text.split() for team in row.find_all(class_="_2tehgH")]
        home, away = ' '.join(teams[0]), ' '.join(teams[1])
        odds = [odd.text.split() for odd in row.find_all(class_="_1NtPy1")]
        home_odds, away_odds = float(odds[0][0]), float(odds[-1][0])

        df = df.append({'Home Team': home, 'Away Team': away, 'Home Odds': home_odds, 'Away Odds': away_odds}, ignore_index=True)
    
    return df

odds = scrape_odds(url)
