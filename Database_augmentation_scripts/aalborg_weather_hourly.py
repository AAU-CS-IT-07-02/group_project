from datetime import datetime
from meteostat import Point, Hourly
import time
start_time = time.time()

location = Point(57.014768, 9.974116)  # AAU Build
start = datetime(2024, 9, 7)
end = datetime(2025, 5, 11)
# we specify the location, start and end date of the dataset
data = Hourly(location, start, end)
data = data.fetch()
# we extract the weather data hourly ( we cannot extract it minute-by-minute )
temp_data = data[['temp']]
# we want to extract only the temperature from the API
temp_data.to_csv("aalborg_weather_hourly_test2.csv")
print("--- %s seconds ---" % (time.time() - start_time))