# Source: https://betfair-datascientists.github.io/api/apiPythontutorial/#sending-requests-to-the-api

# Import libraries
import betfairlightweight
import pandas as pd
import numpy as np
import os
import datetime
import json

#from ausopen_elo_model import elo_pwin
from tennis_model import elo_pwin



from betfairlightweight import filters
from datetime import datetime, timedelta
from timeit import default_timer as timer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn import metrics



start = timer()


# Login to API Client
# =================

# Change this certs path to wherever you're storing your certificates
certs_path = "/Users/17macm/Other/Programming/certs"

# Change these login details to your own
my_username = "mackey.melina@gmail.com"
my_password = "Fr311701!"
my_app_key = "LFpPp46vq2zDrYmR"

trading = betfairlightweight.APIClient(username=my_username,
                                       password=my_password,
                                       app_key=my_app_key,
                                       certs=certs_path)

trading.login()



"""
# Grab all event type ids. This will return a list which we will iterate over to print out the id and the name of the sport
event_types = trading.betting.list_event_types()

sport_ids = pd.DataFrame({
    'Sport': [event_type_object.event_type.name for event_type_object in event_types],
    'ID': [event_type_object.event_type.id for event_type_object in event_types]
}).set_index('Sport').sort_index()

sport_ids
"""



# Get upcoming tennis events
# ==================

# Get a datetime object in a week and convert to string
datetime_in_a_month = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%dT%TZ")

# Create a tennis filter
tennis_filter = betfairlightweight.filters.market_filter(
    event_type_ids=[2], # Tennis's event type id is 2
    competition_ids=[12279276],
    market_start_time={
        'to': datetime_in_a_month,
    })


"""
# Get a list of competitions for tennis
competitions = trading.betting.list_competitions(
    filter=tennis_filter
)

# Iterate over the competitions and create a dataframe of competitions and competition ids
tennis_competitions = pd.DataFrame({
    'Competition': [competition_object.competition.name for competition_object in competitions],
    'ID': [competition_object.competition.id for competition_object in competitions]
})
"""

# Get a list of events for tennis
events = trading.betting.list_events(
    filter=tennis_filter
)

# Iterate over the events and create a dataframe of events and event ids
tennis_events = pd.DataFrame({
    'Event': [event_object.event.name for event_object in events],
    'Event ID': [event_object.event.id for event_object in events],
    'Event Venue': [event_object.event.venue for event_object in events],
    'Country Code': [event_object.event.country_code for event_object in events],
    'Time Zone': [event_object.event.time_zone for event_object in events],
    'Open Date': [event_object.event.open_date for event_object in events],
    'Market Count': [event_object.market_count for event_object in events]
})


"""# Get market types
# ===============
# Define a market filter (market_id = 1.154254141 i.e. winner)
market_types_filter = betfairlightweight.filters.market_filter(event_ids=['29616806', '29617509'])

# Request market types
market_types = trading.betting.list_market_types(
        filter=market_types_filter
)

# Create a DataFrame of market types
market_types_ausopen = pd.DataFrame({
    'Market Type': [market_type_object.market_type for market_type_object in market_types],
})"""


# Get market catalogues
# ===============
market_catalogue_filter = betfairlightweight.filters.market_filter(event_ids=list(tennis_events['Event ID']), in_play_only=False, market_type_codes=['MATCH_ODDS'])

market_catalogues = trading.betting.list_market_catalogue(
    filter=market_catalogue_filter,
    max_results='100',
    sort='FIRST_TO_START',
    market_projection=['RUNNER_DESCRIPTION']
)

# Create a DataFrame for each market catalogue
market_cat_ausopen = pd.DataFrame({
    'Runners': [[runner.runner_name for runner in market_cat_object.runners] for market_cat_object in market_catalogues],
    'Market Name': [market_cat_object.market_name for market_cat_object in market_catalogues],
    'Market ID': [market_cat_object.market_id for market_cat_object in market_catalogues],
    'Total Matched': [market_cat_object.total_matched for market_cat_object in market_catalogues],
    'Selection ID - Player': [player_id[0] for player_id in [[runner.selection_id for runner in market_cat_object.runners] for market_cat_object in market_catalogues]],
    'Selection ID - Opp': [player_id[1] for player_id in [[runner.selection_id for runner in market_cat_object.runners] for market_cat_object in market_catalogues]]
})



# Get market books
# ===============
# Create a price filter. Get all traded and offer data
price_filter = betfairlightweight.filters.price_projection(
    price_data=['EX_BEST_OFFERS']
)

# Request market books
market_books = trading.betting.list_market_book(
    market_ids=[market_cat_object.market_id for market_cat_object in market_catalogues],
    price_projection=price_filter
)

def process_runner_books(runner_books):
    '''
    This function processes the runner books and returns a DataFrame with the best back/lay prices + vol for each runner
    :param runner_books:
    :return:
    '''
    best_back_prices_player = [runner_book[0].ex.available_to_back[0].price
                        if runner_book[0].ex.available_to_back
                        else 1.01
                        for runner_book
                        in runner_books]
    
    best_back_sizes_player = [runner_book[0].ex.available_to_back[0].size
                       if runner_book[0].ex.available_to_back
                       else 1.01
                       for runner_book
                       in runner_books]

    selection_ids_player = [runner_book[0].selection_id for runner_book in runner_books]
    last_prices_traded_player = [runner_book[0].last_price_traded for runner_book in runner_books]
    statuses_player = [runner_book[0].status for runner_book in runner_books]

    best_back_prices_opp = [runner_book[1].ex.available_to_back[0].price
                        if runner_book[1].ex.available_to_back
                        else 1.01
                        for runner_book
                        in runner_books]

    best_back_sizes_opp = [runner_book[1].ex.available_to_back[0].size
                       if runner_book[1].ex.available_to_back
                       else 1.01
                       for runner_book
                       in runner_books]
    
    selection_ids_opp = [runner_book[1].selection_id for runner_book in runner_books]
    last_prices_traded_opp = [runner_book[1].last_price_traded for runner_book in runner_books]
    statuses_opp = [runner_book[1].status for runner_book in runner_books]


    df = pd.DataFrame({
        'Selection ID - Player': selection_ids_player,
        'Best Back Price - Player': best_back_prices_player,
        'Best Back Size - Player': best_back_sizes_player,
        'Last Price Traded - Player': last_prices_traded_player,
        'Status - Player': statuses_player,
        'Selection ID - Opp': selection_ids_opp,
        'Best Back Price - Opp': best_back_prices_opp,
        'Best Back Size - Opp': best_back_sizes_opp,
        'Last Price Traded - Opp': last_prices_traded_opp,
        'Status - Opp': statuses_opp
    }, columns=[
        'Selection ID - Player',
        'Best Back Price - Player',
        'Best Back Size - Player',
        'Last Price Traded - Player',
        'Status - Player',
        'Selection ID - Opp',
        'Best Back Price - Opp',
        'Best Back Size - Opp',
        'Last Price Traded - Opp',
        'Status - Opp'
    ])
    return df

runner_books = [market_book.runners for market_book in market_books]
df = market_cat_ausopen.merge(process_runner_books(runner_books))


elo = pd.read_csv('atpdata/current_elo_ratings.csv')
elo.loc[elo['name'].str.contains('Wawrinka'),'name']='Stanislas Wawrinka'


# Filter out value bets: https://www.bettingkingdom.co.uk/blog/calculating-value-and-finding-your-edge-for-profitable-tennis-betting
# market_prob = 1/odds
# value = my_prob/market_prob = my_prob/(1/odds) = my_prob*odds
# If value > 1.2, then bet


stake = 5
for index, row in df.iterrows():
    player_name = row['Runners'][0]
    opp_name = row['Runners'][1]
    prob_player_win, elo = elo_pwin(elo, player_name, opp_name, datetime.today(), 'Hard')
    prob_opp_win, elo = elo_pwin(elo, opp_name, player_name, datetime.today(), 'Hard')

    if prob_player_win != 0:
        val_player = round(row['Best Back Price - Player']*prob_player_win, 2)
        if 0 <= val_player <= 3 and prob_player_win > prob_opp_win:
            print("Bet {0} to beat {1}. Value of bet is {2}.".format(player_name, opp_name, val_player))
            # exp_val = prob_player_win*row['Best Back Price - Player']-1
            # stake = round(profit/exp_val,2)
            profit = round(row['Best Back Price - Player']*stake - stake, 2)
            print("Stake ${0} to win ${1}".format(stake, profit))
            
            # print(prob_player_win, 1/row['Best Back Price - Player'])
            # print(prob_opp_win)
    if prob_opp_win != 0:
        val_opp = round(row['Best Back Price - Opp']*prob_opp_win, 2)
        if 0 <= val_opp <= 3 and prob_opp_win > prob_player_win:
            print("Bet {0} to beat {1}. Value of bet is {2}.".format(opp_name, player_name, val_opp))
            # exp_val = prob_opp_win*row['Best Back Price - Opp']-1
            # stake = round(profit/exp_val,2)
            profit = round(row['Best Back Price - Opp']*stake - stake, 2)
            print("Stake ${0} to win ${1}".format(stake, profit))
            
            # print(prob_opp_win, 1/row['Best Back Price - Opp'])
            # print(prob_player_win)
end = timer()
print(end - start)