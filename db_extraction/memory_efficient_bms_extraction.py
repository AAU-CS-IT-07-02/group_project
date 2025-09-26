# -*- coding: utf-8 -*-
"""
Created on Sep 22 2025

Based on original script by GQ05XY (Nov 1 2022)
Modified to use multithreading for faster data extraction
Memory-efficient version that uses temp files instead of keeping everything in RAM
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
import tempfile
import shutil
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
password = os.getenv('BD_API_PASSWORD')

print(f"Username: {username}")
print(f"Password: {password}")

# Set the maximum number of concurrent API requests
# Adjust this based on API rate limits and your network capacity
MAX_WORKERS = 8

# Create a temporary directory for chunk storage
def create_temp_dir():
    """Create a temporary directory for storing data chunks"""
    temp_dir = tempfile.mkdtemp(prefix="bms_data_")
    print(f"Created temporary directory: {temp_dir}")
    return temp_dir

# Function to fetch trend data for a specific time interval and save to temp file
def fetch_trend_data_for_interval(starttime, endtime, externallogid, source_df, 
                                 logmap_var_loc, logmap_var_name, temp_dir, interval_index):
    """
    Fetch trend data for a specific time interval and save to a temporary file.
    
    Args:
        starttime: Start time for data extraction
        endtime: End time for data extraction
        externallogid: List of external log IDs to fetch
        source_df: DataFrame containing source information
        logmap_var_loc: Column name for variable location
        logmap_var_name: Column name for variable name
        temp_dir: Directory to save the temporary chunk file
        interval_index: Index of this interval (for unique filename)
    
    Returns:
        Path to the saved temp file, or None if no data was found
    """
    PARAMS = {'starttime': starttime, 'endtime': endtime, 'externallogid': externallogid}
    try:
        print(f'Fetching data for interval: {starttime} to {endtime}')
        trend_data = requests.get(TRENDDATA_NAME, params=PARAMS, auth=(username, password))
        trend_data_text = trend_data.text
        trend_data_df = pd.read_json(trend_data_text, orient='records')
        
        if trend_data_df.empty:
            print(f"No data found for interval {starttime} to {endtime}")
            return None
        
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
            return None
            
        result_df = pd.concat(dfs, axis=0, ignore_index=True)
        col_names = ['externallogid', 'timestamp', 'timestamp_tzinfo', 'value', 'source']
        result_df.columns = col_names
        source_column = result_df.pop('source')
        result_df.insert(1, 'source', source_column)
        
        # Save the result to a temporary file
        temp_file_path = os.path.join(temp_dir, f"chunk_{interval_index}.parquet")
        result_df.to_parquet(temp_file_path)
        
        # Free up memory
        del result_df
        del dfs
        
        return temp_file_path
        
    except Exception as e:
        print(f"Error fetching data for interval {starttime} to {endtime}: {e}")
        return None

# Function to process chunked data and create the final DataFrame
def process_chunked_data(temp_files, na_drop_setting):
    """
    Process the chunked data files to create the final DataFrame
    
    Args:
        temp_files: List of paths to temporary parquet files
        na_drop_setting: Setting for dropping NA values ('all' or 'any')
    
    Returns:
        The final processed DataFrame
    """
    print("Processing chunked data files...")
    
    # First, collect all unique sources across all chunks
    all_sources = set()
    for file_path in tqdm(temp_files, desc="Scanning for unique sources"):
        if file_path is None:
            continue
        chunk_df = pd.read_parquet(file_path)
        all_sources.update(chunk_df['source'].unique())
    
    print(f"Found {len(all_sources)} unique sources")
    
    # Process one source at a time to reduce memory usage
    final_trend_data_df = pd.DataFrame()
    
    for source_id in tqdm(all_sources, desc="Processing sources"):
        source_data = []
        
        # Extract data for this source from each chunk
        for file_path in temp_files:
            if file_path is None:
                continue
                
            chunk_df = pd.read_parquet(file_path)
            source_chunk = chunk_df[chunk_df['source'] == source_id]
            
            if not source_chunk.empty:
                temp_df = pd.DataFrame()
                temp_df[source_id] = source_chunk['value']
                temp_df['time'] = source_chunk['timestamp']
                temp_df['time'] = temp_df['time'].apply(lambda ts: ts.replace(second=0))
                temp_df.set_index('time', inplace=True)
                source_data.append(temp_df)
                
            # Free memory
            del chunk_df
        
        # Combine chunks for this source
        if source_data:
            source_df = pd.concat(source_data, axis=0)
            source_df = source_df[~source_df.index.duplicated(keep='first')]
            
            if len(final_trend_data_df) == 0:
                final_trend_data_df = source_df
            else:
                final_trend_data_df = final_trend_data_df.join(source_df, how='outer')
            
            # Free memory
            del source_df
            del source_data
    
    # Drop NA values and sort the final result
    if not final_trend_data_df.empty:
        final_trend_data_df.dropna(how=na_drop_setting, inplace=True)
        final_trend_data_df = final_trend_data_df[~final_trend_data_df.index.duplicated(keep='first')]
        final_trend_data_df.sort_index(inplace=True)
    
    return final_trend_data_df

# Main process
if __name__ == '__main__':
    # Set the path where the different file(s) to run are located
    files_to_run = glob.glob("./log_maps/Log_map_TMV23_2025_02_28_MIN.xlsx")

    # Set the path for where the data file should be saved to
    save_location = "./SAVED_LOGS"

    # Set the timestep size
    timestep = dt.timedelta(hours=10)

    # In the final dataframe, when dropping NA values, should the row contain NA for all variables, or just for any variables before dropping the row?
    na_drop_setting = 'all'     # either 'any' or 'all'

    filenames = []
    for i in files_to_run:
        filenames.append(i.split("\\")[-1].split(".")[0].split("Log_map_")[-1])

    print("Files to run: ", files_to_run)
    
    for j in range(len(files_to_run)):
        # Create temporary directory for this file processing
        temp_dir = create_temp_dir()
        
        try:
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
            # and save each chunk to a temporary file
            temp_files = [None] * len(time_intervals)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_index = {}
                
                for idx, interval in enumerate(time_intervals):
                    future = executor.submit(
                        fetch_trend_data_for_interval, 
                        interval[0],  # starttime 
                        interval[1],  # endtime
                        externallogid, 
                        source_df, 
                        logmap_var_loc, 
                        logmap_var_name,
                        temp_dir,
                        idx
                    )
                    future_to_index[future] = idx
                
                # Process the results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                                  total=len(future_to_index), 
                                  desc="Downloading data chunks"):
                    idx = future_to_index[future]
                    try:
                        temp_file_path = future.result()
                        temp_files[idx] = temp_file_path
                    except Exception as e:
                        print(f"Error processing interval {time_intervals[idx]}: {e}")
            
            # Check if we have any valid data
            valid_temp_files = [f for f in temp_files if f is not None]
            if not valid_temp_files:
                print("No data was collected. Exiting.")
                continue
                
            print(f"Successfully downloaded {len(valid_temp_files)} data chunks")
                
            # Process the chunked data
            final_trend_data_df = process_chunked_data(valid_temp_files, na_drop_setting)
            
            if final_trend_data_df.empty:
                print("No data after processing. Exiting.")
                continue
                
            print(f"Final dataframe shape: {final_trend_data_df.shape}")
            
            # Save the file with the desired name, location and filetype
            save_file = ''.join([save_location, '/', save_file_name, '_memeff_', 
                               str(start_year), '_', str(start_month), '__', 
                               str(end_year), '_', str(end_month), '.csv'])
            
            print(f"Saving data to {save_file}")
            final_trend_data_df.to_csv(save_file)
            print(f"Data saved successfully!")
            
            # Calculate and print timing statistics
            script_end_time = time.time()
            elapsed_seconds = script_end_time - script_start_time
            elapsed_minutes = elapsed_seconds / 60
            print(f"\nScript execution time: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)")
        
        finally:
            # Clean up the temporary directory
            print(f"Cleaning up temporary files in {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)