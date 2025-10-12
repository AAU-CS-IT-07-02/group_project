import pandas as pd
import time
start_time = time.time()

senzori = pd.read_csv("TMV23_2025_02_28_Rooms_100_memeff_2024_2__2024_6.csv", parse_dates=['time'])
# we read the dataset , using each date value  as datetime
temp = pd.read_csv("aalborg_weather_hourly.csv", parse_dates=['time'])
# we read the weather dataset, using each date value  as datetime
merged = pd.merge_asof(senzori, temp, on='time', direction='backward')
# we merge the two datasets, each row will get the last known hourly temperature from the weather dataset, using the direction backwards
merged.to_csv("TMV23_2025_02_28_Rooms_100_memeff_2024_2__2024_6_augmented_weather.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))