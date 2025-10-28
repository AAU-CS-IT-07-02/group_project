"""
Retrieve and export the historical weather data for a specific locatiom, in this case AAU Build location.

This script is using the Meteostat API to download hourly temperature data, between two time periods. The data is saved as a CSV file for further analysis.

Requirements:
    meteostat (install using pip install meteostat)

Modules:
    datetime: Used for defining the time range.
    meteostat: Provides weather data retrieval functionality.

Output:
    output_file: CSV with hourly weather data with timestamps.

"""

from datetime import datetime
from meteostat import Point, Hourly
import time

# change as needed
output_file = "aalborg_weather_hourly_test2.csv"

start_time = time.time()
location = Point(57.014768, 9.974116)
start = datetime(2024, 9, 7)
end = datetime(2025, 5, 11)
data = Hourly(location, start, end)
data = data.fetch()
temp_data = data[['temp']]
temp_data.to_csv(output_file)
print("--- %s seconds ---" % (time.time() - start_time))