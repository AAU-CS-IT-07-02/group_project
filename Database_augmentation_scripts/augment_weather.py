"""
Augment sensor data with corresponding historical weather information.

This script merges time-series data from indoor room sensors with hourly
weather data (temperature) using a nearest backward timestamp match.
The merged dataset is then exported to a CSV file for further analysis.

Input Files:
    sensor_file: CSV with time-stamped sensor data.
    weather_file: CSV with hourly weather data with timestamps.

Output:
    output_fule: Merged dataset combining room sensor and weather data.
"""

import pandas as pd
import time

# change as needed
sensor_file = "TMV23_2025_02_28_Rooms_100_memeff_2024_2__2024_6.csv"
weather_file = "aalborg_weather_hourly.csv"
output_file = "TMV23_2025_02_28_Rooms_100_memeff_2024_2__2024_6_augmented_weather.csv"

start_time = time.time()
sensor = pd.read_csv(sensor_file, parse_dates=['time'])
temp = pd.read_csv(weather_file, parse_dates=['time'])
merged = pd.merge_asof(sensor, temp, on='time', direction='backward')
merged.to_csv(output_file, index=False)
print("--- %s seconds ---" % (time.time() - start_time))