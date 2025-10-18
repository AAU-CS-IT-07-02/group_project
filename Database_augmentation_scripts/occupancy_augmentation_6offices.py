"""
This script is designed to process a raw sensor data CSV file (`dataset__2023_02_27__2023_12_31.csv`) and enrich it with occupancy data.

It generates a new file, `dataset_with_occupancy.csv`, which includes all the original data plus six new columns (one for each room, A-F). These new columns (`RoomA_is_occupied`, etc.) contain a binary value: `1` if the room is considered occupied and `0` if it is considered unoccupied.

The occupancy status is determined by a specific set of rules based on CO2 levels and the time of day.
"""
import pandas as pd
import sys

input_csv_file = "dataset__2023_02_27__2023_12_31.csv"
output_csv_file = "dataset_with_occupancy.csv"
col_time = "timestamp"

rooms = ['A', 'B', 'C', 'D', 'E', 'F']
co2_columns = [f"Room{letter}:Sensor__CO2" for letter in rooms]
occupancy_columns = [f"Room{letter}_is_occupied" for letter in rooms]

# occupancy rules
CO2_THRESHOLD = 470
START_HOUR = 7
END_HOUR = 22
TIMEZONE = 'Europe/Copenhagen'

try:
    df = pd.read_csv(input_csv_file, sep=';')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    all_check_cols = [col_time] + co2_columns
    missing = [c for c in all_check_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df[col_time] = pd.to_datetime(df[col_time], errors='coerce', utc=True)
    df.dropna(subset=[col_time], inplace=True)
    
    for col in co2_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.set_index(col_time, inplace=True)
    df.sort_index(inplace=True)

    # convert time zone
    try:
        df.index = df.index.tz_convert(TIMEZONE)
    except TypeError:
        df.index = df.index.tz_localize('UTC').tz_convert(TIMEZONE)
        
    # occupancy logic
    cond_hour = (df.index.hour >= START_HOUR) & (df.index.hour < END_HOUR)
    
    for col_co2, col_occ in zip(co2_columns, occupancy_columns):
        cond_co2 = (df[col_co2] > CO2_THRESHOLD)
        df[col_occ] = (cond_co2 & cond_hour).astype(int)
    
    df.reset_index(inplace=True)
    df[col_time] = df[col_time].dt.strftime("%Y-%m-%d %H:%M:%S")   
    df.to_csv(output_csv_file, sep=';', index=False, encoding='utf-8')
    

except FileNotFoundError:
    print(f"The file '{input_csv_file}' was not found.")
    sys.exit(1)
except KeyError as e:
    print(f"A column was not found. {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)