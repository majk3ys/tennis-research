from geotext import GeoText


"""from geopy import Nominatim

geolocator = Nominatim(user_agent="my-application")
location = geolocator.geocode("Pune")

import geograpy
"""

def get_home_adv(player_country, tourney_city):
    tourney_city = 'Belgrade'
    country = list(GeoText(tourney_city).country_mentions)[0]
    
    # filter by country code
    if country in [x.upper() for x in player_country]:
        return True
    else:
        return False
        
