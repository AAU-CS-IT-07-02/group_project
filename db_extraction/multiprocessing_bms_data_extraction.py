# -*- coding: utf-8 -*-
"""
Created on Sep 22 2025

Based on original script by GQ05XY (Nov 1 2022)
Modified to use multithreading for faster data extraction
"""

import requests
import pandas as pd
import datetime as dt
import math
import json
import csv
import os
import glob
import time
import concurrent.futures
from tqdm import tqdm  # For progress bars
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Start timing the script
script_start_time = time.time()

pd.set_option('display.max_colwidth', None)

# API endpoints for the DB server
TRENDDATA_NAME = "https://bms-api.build.aau.dk/api/v1/trenddata"
METADATA_NAME = "https://bms-api.build.aau.dk/api/v1/metadata"

# Set the username for the DB server (if you do not have one ask Simon)
username = os.getenv('BD_API_USER')
# Set the password for the DB server (if you do not have one ask Simon)
password = os.getenv('BD_APU_PASSWORD')

print(f"Username: {username}")
print(f"Password: {password}")

# Set the maximum number of concurrent API requests
# Adjust this based on API rate limits and your network capacity
MAX_WORKERS = 8

# Function to fetch trend data for a specific time interval
def fetch_trend_data_for_interval(starttime, endtime, externallogid, source_df, logmap_var_loc, logmap_var_name):
    """
    Fetch trend data for a specific time interval.
    
    Args:
        starttime: Start time for data extraction
        endtime: End time for data extraction
        externallogid: List of external log IDs to fetch
        source_df: DataFrame containing source information
        logmap_var_loc: Column name for variable location
        logmap_var_name: Column name for variable name
    
    Returns:
        DataFrame containing the trend data for the interval
    """
    PARAMS = {'starttime': starttime, 'endtime': endtime, 'externallogid': externallogid}
    try:
        print(f'Fetching data for interval: {starttime} to {endtime}')
        trend_data = requests.get(TRENDDATA_NAME, params=PARAMS, auth=(username, password))
        trend_data_text = trend_data.text
        trend_data_df = pd.read_json(trend_data_text, orient='records')
        
        if trend_data_df.empty:
            print(f"No data found for interval {starttime} to {endtime}")
            return pd.DataFrame()
        
        # Process data similar to the original script
        temp_externallogid_1 = source_df['externallogid'].explode()
        temp_externallogid_1 = temp_externallogid_1[~temp_externallogid_1.index.duplicated(keep='first')]
        local_source_df = source_df.copy()
        local_source_df['externallogid'] = temp_externallogid_1
        
        dfs = []
        for id_, id_df in trend_data_df.groupby('externallogid'):
            temp_source = (local_source_df[local_source_df['externallogid']==id_][logmap_var_loc]).tolist()
            temp_name = (local_source_df[local_source_df['externallogid']==id_][logmap_var_name]).tolist()
            
            if not temp_source or not temp_name:
                continue
                
            temp_source_name = '/'.join(temp_source + temp_name)
            
            temp_s = pd.Series([temp_source_name] * len(id_df))
            temp_df = pd.concat([id_df.reset_index(drop=True),
                                temp_s.reset_index(drop=True)],
                                ignore_index=True, axis=1)
            
            dfs.append(temp_df)
        
        if not dfs:
            return pd.DataFrame()
            
        result_df = pd.concat(dfs, axis=0, ignore_index=True)
        col_names = ['externallogid', 'timestamp', 'timestamp_tzinfo', 'value', 'source']
        result_df.columns = col_names
        source_column = result_df.pop('source')
        result_df.insert(1, 'source', source_column)
        
        return result_df
        
    except Exception as e:
        print(f"Error fetching data for interval {starttime} to {endtime}: {e}")
        return pd.DataFrame()

# Main process
if __name__ == '__main__':
    # Set the path where the different file(s) to run are located
    files_to_run = glob.glob("C:/software/AAU/aau_group_project_7/db_extraction/log_maps/Log_map_TMV23_2025_02_28_Rooms_100.xlsx")

    # Set the path for where the data file should be saved to
    save_location = "C:/software/AAU/aau_group_project_7/db_extraction/SAVED_LOGS"

    # Set the timestep size
    timestep = dt.timedelta(hours=5)

    # In the final dataframe, when dropping NA values, should the row contain NA for all variables, or just for any variables before dropping the row?
    na_drop_setting = 'all'     # either 'any' or 'all'

    filenames = []
    for i in files_to_run:
        filenames.append(i.split("\\")[-1].split(".")[0].split("Log_map_")[-1])

    print("Files to run: ", files_to_run)
    
    for j in range(len(files_to_run)):
        # Set the name of the data file as a string, e.g. 'test'
        save_file_name = filenames[j]
        
        # Start and endtime for the dataextraction 
        start_year = 2024
        start_month = 1
        start_day = 1
        start_hour = 4
        end_year = 2024
        end_month = 2
        end_day = 1
        end_hour = 4
        
        starttime = dt.datetime(start_year, start_month, start_day, start_hour) 
        endtime = dt.datetime(end_year, end_month, end_day, end_hour) 
        
        # if identifier is 3, the source_df is pulled from the excel sheet logmap (for many inputs)
        source_logmap = files_to_run[j]
        logmap_sheet = 'log_map'
        logmap_columns = 'A:D'
        logmap_var_loc = 'Log_variable_location'
        logmap_var_name = 'Logged_variable_name'
        
        print(f"Reading logmap from {source_logmap}")
        source_df = pd.read_excel(io=source_logmap, sheet_name=logmap_sheet, header=0, usecols=logmap_columns)
        
        # Create the list of externallogid (same as original script)
        source = []
        for i in range(0, len(source_df)):
            temp = '/'.join([str(source_df[logmap_var_loc][i]), str(source_df[logmap_var_name][i])])
            source.append(temp)
            
        print("Fetching metadata...")
        trend_meta = requests.get(url=METADATA_NAME, auth=(username, password))
        trend_meta_text = trend_meta.text
        print("Metadata status code: ", trend_meta.status_code)
        
        trend_meta_df = pd.read_json(trend_meta_text, orient='records')
        print("Metadata columns: ", trend_meta_df.columns)
        
        externallogid = []
        temp_externallogid = []
        for i in range(0, len(source_df)):
            externallogid.append(trend_meta_df.externallogid[source[i] == trend_meta_df.source].tolist())
            temp_externallogid.append(trend_meta_df.externallogid[source[i] == trend_meta_df.source].tolist())
            if externallogid[-1] == []:
                print(' '.join(['source_df index', str(source_df.index[i]), 'failed']))
                externallogid.pop()
                
        source_df_insert_point = len(source_df.columns)
        source_df.insert(source_df_insert_point, 'externallogid', temp_externallogid)
        print("Metadata processing completed")
        
        # Generate time intervals
        time_intervals = []
        current_time = starttime
        while current_time < endtime:
            time_intervals.append((current_time, current_time + timestep))
            current_time += timestep
            
        print(f"Total time intervals to process: {len(time_intervals)}")
        
        # Use ThreadPoolExecutor to fetch data for each time interval in parallel
        all_trend_data_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a dictionary mapping futures to time intervals for better tracking
            future_to_interval = {
                executor.submit(
                    fetch_trend_data_for_interval, 
                    interval[0],  # starttime 
                    interval[1],  # endtime
                    externallogid, 
                    source_df, 
                    logmap_var_loc, 
                    logmap_var_name
                ): interval for interval in time_intervals
            }
            
            # Process the results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_interval), 
                              total=len(future_to_interval), 
                              desc="Processing time intervals"):
                interval = future_to_interval[future]
                try:
                    trend_data_df = future.result()
                    if not trend_data_df.empty:
                        all_trend_data_dfs.append(trend_data_df)
                except Exception as e:
                    print(f"Error processing interval {interval}: {e}")
        
        # Combine all dataframes
        if all_trend_data_dfs:
            temp_trend_data_df = pd.concat(all_trend_data_dfs, ignore_index=True)
            print(f"Combined data shape: {temp_trend_data_df.shape}")
        else:
            print("No data was collected. Exiting.")
            continue
            
        print("All data extracted")
        # Convert the temp_trend_data_df file into a useful df
        unique_source = temp_trend_data_df['source'].unique()
        print(f"Found {len(unique_source)} unique sources")
        
        final_trend_data_df = pd.DataFrame()
        for source_id in tqdm(unique_source, desc="Processing sources"):
            temp_final_trend_data_df = pd.DataFrame()
            temp_final_trend_data_df[source_id] = temp_trend_data_df['value'][temp_trend_data_df['source']==source_id]
            temp_final_trend_data_df['time'] = temp_trend_data_df['timestamp'][temp_trend_data_df['source']==source_id]
            temp_final_trend_data_df['time'] = temp_final_trend_data_df['time'].apply(lambda ts: ts.replace(second=0))
            temp_final_trend_data_df.set_index('time',inplace=True)
            
            if len(final_trend_data_df) == 0:
                final_trend_data_df = temp_final_trend_data_df
            else:
                final_trend_data_df = final_trend_data_df.join(temp_final_trend_data_df, how='outer')
            
            final_trend_data_df.dropna(how=na_drop_setting, inplace=True)
            final_trend_data_df = final_trend_data_df[~final_trend_data_df.index.duplicated(keep='first')]
            
        final_trend_data_df.sort_index(inplace=True)
        
        # Save the file with the desired name, location and filetype
        save_file = ''.join([save_location, '/', save_file_name, '_mp_', 
                           str(start_year), '_', str(start_month), '__', 
                           str(end_year), '_', str(end_month), '.csv'])
        
        print(f"Saving data to {save_file}")
        final_trend_data_df.to_csv(save_file)
        print(f"Data saved successfully! Final dataframe shape: {final_trend_data_df.shape}")
        
        # Calculate and print timing statistics
        script_end_time = time.time()
        elapsed_seconds = script_end_time - script_start_time
        elapsed_minutes = elapsed_seconds / 60
        print(f"\nScript execution time: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)")