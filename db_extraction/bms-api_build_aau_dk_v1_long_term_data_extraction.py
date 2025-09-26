# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:10:44 2022

@author: GQ05XY
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
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Start timing the script
script_start_time = time.time()

pd.set_option('display.max_colwidth', None)
###############################
# API endpoints for the DB server
TRENDDATA_NAME = "https://bms-api.build.aau.dk/api/v1/trenddata"
METADATA_NAME = "https://bms-api.build.aau.dk/api/v1/metadata"

# Set the username for the DB server (if you do not have one ask Simon)
username = os.getenv('BD_API_USER')
# Set the password for the DB server (if you do not have one ask Simon)
password = os.getenv('BD_APU_PASSWORD')

print(f"Username: {username}")
print(f"Password: {password}")

# Set the path where the different file(s) to run are located
# files_to_run = glob.glob("log_maps/Log_map_TMV23_2025_02_28_MIN.xlsx") #TODO
files_to_run = glob.glob("./log_maps/Log_map_TMV23_2025_02_28_MIN.xlsx")


# Set the path for where the data file should be saved to as a string, e.g. 'C:/Users/GQ05XY/Aalborg Universitet/SATO - General/TMV23/Data_from_BMS/Data_dumb_from_DB'
save_location = "./SAVED_LOGS"

# Set the timestep size
timestep = dt.timedelta(hours=1) #TODO

# In the final dataframe, when dropping NA values, should the row contain NA for all variables, or just for any variables before dropping the row?
na_drop_setting = 'all'     #TODO either 'any' or 'all'


filenames = []
for i in files_to_run:
    filenames.append(i.split("\\")[-1].split(".")[0].split("Log_map_")[-1])

print("Files to run: ", files_to_run)
for j in range(len(files_to_run)):
    # Set the name of the data file as a string, e.g. 'test'
    save_file_name = filenames[j]
    
    ###############################
    # Start and endtime for the dataextraction 
    start_year = 2024       #TODO
    start_month = 1         #TODO
    start_day = 1           #TODO
    start_hour = 0          #TODO
    end_year = 2024         #TODO
    end_month = 1           #TODO
    end_day = 1             #TODO
    end_hour = 6            #TODO
    
    starttime = dt.datetime(start_year, start_month, start_day, start_hour) 
    endtime = starttime + timestep
    final_endtime = dt.datetime(end_year, end_month, end_day, end_hour) 
    
    
    temp_trend_data_df = pd.DataFrame()
    while starttime < final_endtime:
        print('Now running file '+filenames[j]+' at timestep '+str(starttime))
        ###############################
        # if identifier is 3, the source_df is pulled from the excel sheet logmap (for many inputs)
        # Fill in the path to the logmap as a string, e.g. 'C:/Users/GQ05XY/Aalborg Universitet/SATO - General/TMV23/Data_from_BMS/Log_map_TMV23.xlsx'
        source_logmap = files_to_run[j]
        # Fill in the name of the sheet containing the logmap as a string, e.g. 'log_map'
        logmap_sheet = 'log_map' #TODO
        # Fill in the columns that should be imported from the logmap file as a string, e.g. 'A:D'
        logmap_columns = 'A:D' #TODO
        # Fill in the name of the column containing the location of the log variable as a string, e.g. 'Log_variable_location'
        logmap_var_loc = 'Log_variable_location' #TODO
        # Fill in the name of the column containing the name of the log variable as a string, e.g. 'Logged_variable_name'
        logmap_var_name = 'Logged_variable_name' #TODO

        
        #########################################################################################################################################################################
        ##                                                                  Script below                                                                                       ##
        ##                                                                  Do not touch!                                                                                      ##
        #########################################################################################################################################################################
        # create the list of externallogid
        
        
            
        
        source_df = pd.read_excel(io=source_logmap,sheet_name=logmap_sheet,header=0,usecols=logmap_columns)     # create the source from the logmap, with the specified naming and location
        source = []
        for i in range(0,len(source_df)):
            temp = '/'.join([str(source_df[logmap_var_loc][i]),str(source_df[logmap_var_name][i])])             # create a full path with variable name for each variable
            source.append(temp)                                                                                 # append the full path of each variable to this list, to get a full list of variable paths
        trend_meta = requests.get(url=METADATA_NAME, auth=(username, password))                                                            # extract the trend_meta data as a dataframe
        trend_meta_text = trend_meta.text
        print("Trend meta: ", trend_meta)
        print("Trend meta text: ", trend_meta_text[:500])  # Print first 500 characters to check format
        print("Trend meta status code: ", trend_meta.status_code)
        trend_meta_df = pd.read_json(trend_meta_text, orient = 'records')
        print("trend_meta_df columns: ", trend_meta_df.columns)
        externallogid=[]
        temp_externallogid=[]
        for i in range(0,len(source_df)):
            externallogid.append(trend_meta_df.externallogid[source[i] == trend_meta_df.source].tolist())       # create a list of all valid externallogid
            temp_externallogid.append(trend_meta_df.externallogid[source[i] == trend_meta_df.source].tolist())  # create a list of all externallogid
            if externallogid[-1] == []:
                print(' '.join(['source_df index',str(source_df.index[i]),'failed']))                           # Print the index corresponding to source_df for each variable, which did not have a valid externallogid
                externallogid.pop()                                                                             # remove the invalid externallogid from the list of all valid externallogid
        source_df_insert_point = len(source_df.columns)
        source_df.insert(source_df_insert_point,'externallogid',temp_externallogid)                                                  # append the list of all externallogid to the source_df for an easy overview of which externallogid are missing
        print("Meta data done")
        del temp
        del temp_externallogid
        del source
        del i
        
        
        
        # extract the trend_data
        PARAMS = {'starttime':starttime, 'endtime':endtime, 'externallogid':externallogid}
        try:
            trend_data = requests.get(TRENDDATA_NAME, params=PARAMS, auth=(username, password))                                                    # extract the trend_data for each externallogid in the timespan between starttime and endtime
            trend_data_text = trend_data.text
            trend_data_df = pd.read_json(trend_data_text,orient = 'records') 
            
            print("Data extracted")
            
            
            temp_externallogid_1 = source_df['externallogid'].explode()
            temp_externallogid_1 = temp_externallogid_1[~temp_externallogid_1.index.duplicated(keep='first')]
            source_df['externallogid'] = temp_externallogid_1
            
            dfs = []
            for id_, id_df in trend_data_df.groupby('externallogid'):
                temp_source = (source_df[source_df['externallogid']==id_]['Log_variable_location']).tolist()
                temp_name   = (source_df[source_df['externallogid']==id_]['Logged_variable_name']).tolist()
                
                temp_source_name = '/'.join(temp_source + temp_name)
                
                temp_s = pd.Series([temp_source_name] * len(id_df))
                temp_df = pd.concat([id_df.reset_index(drop=True),
                                     temp_s.reset_index(drop=True)],
                                    ignore_index=True, axis=1)
                
                dfs.append(temp_df)
            
            
            trend_data_df = pd.concat(dfs, axis=0, ignore_index=True)
            col_names = ['externallogid', 'timestamp', 'timestamp_tzinfo', 'value', 'source']
            trend_data_df.columns = col_names
            source_column = trend_data_df.pop('source')
            trend_data_df.insert(1,'source',source_column)
            print("Source added")
            
            if len(temp_trend_data_df) == 0:
                temp_trend_data_df = trend_data_df.copy()
            else:
                temp_trend_data_df = pd.concat([temp_trend_data_df,trend_data_df], ignore_index=True)
            
            print("temp_trend_data_df finished")
        except Exception as e:
            print("An error occurred: ", e)
            pass
        

    
        starttime += timestep                                     # create the startime indicator for the data extraction
        endtime = starttime + timestep                                                                              # create the endtime indicator for the data extraction
    
    
    print("All data extracted")
    # Convert the temp_trend_data_df file into a useful df
    unique_source = temp_trend_data_df['source'].unique()
    
    final_trend_data_df = pd.DataFrame()
    for source_id in unique_source:
        temp_final_trend_data_df = pd.DataFrame()
        temp_final_trend_data_df[source_id] = temp_trend_data_df['value'][temp_trend_data_df['source']==source_id]
        temp_final_trend_data_df['time'] = temp_trend_data_df['timestamp'][temp_trend_data_df['source']==source_id]
        temp_final_trend_data_df['time'] = temp_final_trend_data_df['time'].apply(lambda ts: ts.replace(second=0))
        temp_final_trend_data_df.set_index('time',inplace=True)
        if len(final_trend_data_df) == 0:
            final_trend_data_df = temp_final_trend_data_df
        else:
            final_trend_data_df = final_trend_data_df.join(temp_final_trend_data_df,how='outer')
        
        final_trend_data_df.dropna(how=na_drop_setting,inplace=True)
        final_trend_data_df = final_trend_data_df[~final_trend_data_df.index.duplicated(keep='first')]
        print(source_id)
        print(final_trend_data_df.size)
        print(len(final_trend_data_df))
    final_trend_data_df.sort_index(inplace=True)
    


    # save the file with the desired name, location and filetype
    save_file = ''.join([save_location,'/',save_file_name,'_',str(start_year),'_',str(start_month),'__',str(end_year),'_',str(end_month),'.csv'])
    
    final_trend_data_df.to_csv(save_file)
    
    # Calculate and print timing statistics
    script_end_time = time.time()
    elapsed_seconds = script_end_time - script_start_time
    elapsed_minutes = elapsed_seconds / 60
    print(f"\nScript execution time: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)")
