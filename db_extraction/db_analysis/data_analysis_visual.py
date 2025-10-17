import pandas as pd
import os
import re
import matplotlib 


matplotlib.use('Agg') 


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(os.path.dirname(script_dir))


data_folder = os.path.join(project_root, 'Database')
output_folder = os.path.join(project_root, 'output')
output_graph_folder = os.path.join(output_folder, 'graphs')


map_file_path = os.path.join(data_folder, 'aba.xlsx')

raw_csv_files = [
    os.path.join(data_folder, 'TMV23_2025_02_28_BASE_1K_mp_2024_1__2024_2.csv'),
    os.path.join(data_folder, 'TMV23_2025_02_28_BASE_1K_mp_2024_4__2024_6.csv'),
    os.path.join(data_folder, 'TMV23_2025_02_28_BASE_1K_mp_2024_6__2024_8.csv'),
    os.path.join(data_folder, 'TMV23_2025_02_28_BASE_1K_mp_2024_8__2024_10.csv'),
    os.path.join(data_folder, 'TMV23_2025_02_28_BASE_1K_mp_2024_10__2025_1.csv')
]


output_minute_data_csv = os.path.join(output_folder, "specific_rooms_transposed_data.csv")
output_hourly_csv = os.path.join(output_folder, 'hourly_averaged_transposed_data.csv')




def create_minute_level_report(rooms_to_process: list):
    
    print(f"\n--- PART 1: Creating Minute-by-Minute Report for {len(rooms_to_process)} rooms ---")
    
    try:
        
        print(f"Steps 1-4: Reading local map file '{os.path.basename(map_file_path)}'...")
        df_map = pd.read_excel(map_file_path)
        
        df_map['Room_Clean'] = df_map['Room'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        clean_room_list = [re.escape(str(r).strip()) for r in rooms_to_process]
        pattern = '|'.join(clean_room_list)
        map_filtered = df_map[df_map['Room_Clean'].str.contains(pattern, na=False, regex=True)].copy()

        if map_filtered.empty:
            print("\nFATAL ERROR: No matching rooms were found in 'aba.xlsx'. Exiting.")
            return None

        print("\n--- The following entries from aba.xlsx matched your room list: ---")
        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(map_filtered[['Room', 'Logged_variable_name', 'Log_variable_location']])
        
        all_csv_headers = set()
        print("\nReading sensor names from local CSV files...")
        for file_path in raw_csv_files:
            if os.path.exists(file_path):
                all_csv_headers.update(pd.read_csv(file_path, nrows=0, encoding='utf-8').columns)
            else:
                print(f"  - Warning: File not found: {os.path.basename(file_path)}")
        
        required_locations = map_filtered['Log_variable_location'].dropna().unique()
        match_data = []
        for location in required_locations:
            best_match = next((header for header in all_csv_headers if str(header).startswith(str(location))), None)
            if best_match:
                match_data.append({'Log_variable_location': location, 'Matched_Sensor_ID': best_match})

        df_matches = pd.DataFrame(match_data)
        required_sensor_ids = df_matches['Matched_Sensor_ID'].tolist()

        if not required_sensor_ids:
            print("\nFATAL ERROR: No matching sensors found in local CSV files. Exiting.")
            return None

        
        print("\nStep 5: Extracting data from local CSV files...")
        all_room_data = []
        for file_path in raw_csv_files:
            if not os.path.exists(file_path):
                continue
            
            print(f"  - Reading from: {os.path.basename(file_path)}")
            header = pd.read_csv(file_path, nrows=0).columns.tolist()
            cols_to_read = ['time'] + [col for col in header if col in required_sensor_ids]
            if len(cols_to_read) > 1:
                df_chunk = pd.read_csv(file_path, usecols=cols_to_read)
                all_room_data.append(df_chunk)

        
        print("\nSteps 6-8: Combining, transposing, adding details, and saving...")
        df_combined = pd.concat(all_room_data, ignore_index=True)
        df_combined = df_combined.groupby('time').mean(numeric_only=True).reset_index()
        df_transposed = df_combined.set_index('time').T
        df_transposed.index.name = 'Sensor_ID'
        df_transposed.reset_index(inplace=True)
        df_transposed['Log_variable_location'] = df_transposed['Sensor_ID'].str.rsplit('/', n=1).str[0]
        df_map_subset = df_map[['Room', 'Logged_variable_name', 'Log_variable_location']].drop_duplicates()
        df_final = pd.merge(df_transposed, df_map_subset, on='Log_variable_location', how='left')
        df_final = df_final[['Room', 'Logged_variable_name', 'Sensor_ID'] + [col for col in df_final.columns if col not in ['Room', 'Logged_variable_name', 'Sensor_ID', 'Log_variable_location']]]
        
        df_final.to_csv(output_minute_data_csv, index=False)
        print(f"\nSUCCESS: Minute-by-minute data saved to:\n'{output_minute_data_csv}'")
        return df_final

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: File not found. Please make sure '{os.path.basename(e.filename)}' is in the 'Database' folder.")
        return None
    except Exception as e:
        print(f"\nAn error occurred during Part 1: {e}")
        return None

def analyze_and_visualize(df_wide: pd.DataFrame):
    
    print("\n--- PART 2: Calculating Hourly Averages and Generating Graphs ---")
    if df_wide is None or df_wide.empty:
        print("Skipping Part 2 because Part 1 did not produce any data.")
        return

    try:
        
        id_vars = ['Room', 'Logged_variable_name', 'Sensor_ID']
        time_cols = [col for col in df_wide.columns if col not in id_vars]
        df_long = df_wide.melt(id_vars=id_vars, value_vars=time_cols, var_name='time', value_name='value')
        df_long['time'] = pd.to_datetime(df_long['time'])
        df_long.dropna(subset=['value'], inplace=True)

        
        print("Calculating hourly average for each sensor...")
        
        
        df_hourly = df_long.set_index('time').groupby(['Room', 'Logged_variable_name', 'Sensor_ID']).resample('h', include_groups=False).mean(numeric_only=True)
        df_hourly = df_hourly.reset_index()

    
        df_hourly_wide = df_hourly.pivot_table(index=['Room', 'Logged_variable_name', 'Sensor_ID'], columns='time', values='value').reset_index()
        df_hourly_wide.to_csv(output_hourly_csv, index=False)
        print(f"\nSUCCESS: Hourly averaged data saved to:\n'{output_hourly_csv}'")
        
        
        print("\n--- Graph Date Range Selection ---")
        print("Please enter a date range for the graphs in DD-MM-YYYY format (e.g., 01-03-2024).")
        print("Press Enter at both prompts to graph the entire date range.")
        
        start_date_str = input("Enter start date (DD-MM-YYYY): ").strip()
        end_date_str = input("Enter end date (DD-MM-YYYY): ").strip()
        
        start_date, end_date = None, None
        try:
            if start_date_str:
                start_date = pd.to_datetime(start_date_str, format='%d-%m-%Y')
            if end_date_str:
                end_date = pd.to_datetime(end_date_str, format='%d-%m-%Y').replace(hour=23, minute=59, second=59)
        except ValueError:
            print("Invalid date format. Defaulting to the full date range.")
        
        df_filtered_hourly = df_hourly.copy()
        if start_date or end_date:
            print(f"Filtering data from {start_date_str or 'the beginning'} to {end_date_str or 'the end'}...")
            if start_date:
                df_filtered_hourly = df_filtered_hourly[df_filtered_hourly['time'] >= start_date]
            if end_date:
                df_filtered_hourly = df_filtered_hourly[df_filtered_hourly['time'] <= end_date]
        
        
        print("\nGenerating a large graph for each individual sensor...")
        os.makedirs(output_graph_folder, exist_ok=True)
        
        unique_sensors = df_filtered_hourly['Sensor_ID'].unique()
        sns.set_style("whitegrid")
        
        for sensor_id in unique_sensors:
            df_sensor = df_filtered_hourly[df_filtered_hourly['Sensor_ID'] == sensor_id]
            if df_sensor.empty: continue

            room_name = df_sensor['Room'].iloc[0]
            sensor_name = df_sensor['Logged_variable_name'].iloc[0]
            
            print(f"  - Creating graph for Sensor: {sensor_name} in Room {room_name}...")
            
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.lineplot(data=df_sensor, x='time', y='value', marker='o', markersize=5, ax=ax)
            
            plt.title(f'Hourly Data for: {sensor_name}\n(Room: {room_name})', fontsize=20)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Sensor Value', fontsize=14)
            
            time_span_days = (df_sensor['time'].max() - df_sensor['time'].min()).days
            
            if time_span_days <= 3:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))
            elif time_span_days <= 14:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            else:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            fig.autofmt_xdate(rotation=45, ha='right')
            plt.tight_layout()
            
            safe_sensor_name = re.sub(r'[^\w\.-]', '_', str(sensor_name))
            safe_room_name = re.sub(r'[^\w\.-]', '_', str(room_name))
            output_path = os.path.join(output_graph_folder, f"Room_{safe_room_name}_{safe_sensor_name}.png")
            
            plt.savefig(output_path)
            plt.close()
            print(f"    Saved graph to local folder: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during Part 2: {e}")


if __name__ == "__main__":
    
    
    os.makedirs(output_graph_folder, exist_ok=True)
    
    user_rooms = []
    print("--- AAU Building Data Processor ---")
    print(f"Reading data from: {data_folder}")
    print(f"Saving output to: {output_folder}\n")
    print("Please enter the room numbers you want to process.")
    print("You can enter up to 20 rooms. Press Enter on an empty line when you are finished.")
    
    while len(user_rooms) < 20:
        prompt = f"Enter room number {len(user_rooms) + 1} (or press Enter to finish): "
        room_input = input(prompt)
        if not room_input:
            break
        user_rooms.append(room_input.strip())

    if user_rooms:
        minute_data_df = create_minute_level_report(user_rooms)
        
        if minute_data_df is not None and not minute_data_df.empty:
            analyze_and_visualize(minute_data_df)
        
        print("\n-------------------------------------")
        print("All tasks are complete.")
        print(f"Your files and graphs are saved in: {output_folder}")
        print("-------------------------------------")
    else:
        print("\nNo room numbers were entered. Exiting script.")