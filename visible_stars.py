from skyfield.api import load
from skyfield.api import Topos

ts = load.timescale()
planets = load('de421.bsp')
earth, *_ = planets['earth']


toronto = earth + Topos(latitude_degrees=43.70, longitude_degrees=-79.42)

current_time = ts.now()

stars = load('hipparcos.csv')

astrometric = (planets + stars).at(current_time).observe(toronto)
alt, az, d = astrometric.apparent().altaz()

visible_stars = [star for star, altitude in zip(stars, alt.degrees) if altitude > 0]

print(visible_stars)

