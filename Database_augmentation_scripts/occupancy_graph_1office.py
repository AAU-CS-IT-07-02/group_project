import pandas as pd
import matplotlib.pyplot as plt
import sys

input_csv_file = "dataset_with_occupancy.csv"
ROOM_TO_TEST = 'A'
start_time_str = "2023-10-01 00:00:00" 
end_time_str   = "2023-10-21 23:00:00" 

col_time = "timestamp"
col_co2_to_plot = f"Room{ROOM_TO_TEST}:Sensor__CO2"
col_occ_to_plot = f"Room{ROOM_TO_TEST}_is_occupied"

try:
    df = pd.read_csv(input_csv_file, sep=';')
    
    if col_co2_to_plot not in df.columns or col_occ_to_plot not in df.columns:
        raise KeyError(f"Data for  '{ROOM_TO_TEST}' was not found")
        
    df[col_co2_to_plot] = pd.to_numeric(df[col_co2_to_plot], errors='coerce')
    df[col_occ_to_plot] = pd.to_numeric(df[col_occ_to_plot], errors='coerce').fillna(0).astype(int)
    
    df.set_index(col_time, inplace=True)
    df.index = pd.to_datetime(df.index, errors='coerce')

    df = df[df.index.notna()]
    
    start_time = pd.to_datetime(start_time_str)
    end_time   = pd.to_datetime(end_time_str)
    
    df_interval = df.loc[start_time:end_time].copy()

    # graph
    fig, ax1 = plt.subplots(figsize=(15, 7))

    ax1.plot(df_interval.index, df_interval[col_co2_to_plot], color='dodgerblue', label=f'Room {ROOM_TO_TEST} CO₂')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('CO₂ (ppm)', color='dodgerblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.axhline(y=500, color='red', linestyle='--', linewidth=1.5, label='470 ppm threshold')
    
    ax2 = ax1.twinx()
    ax2.plot(df_interval.index, df_interval[col_occ_to_plot], color='orange', label=f'Occupancy (1/0)', drawstyle='steps-post', linewidth=2)
    ax2.set_ylabel('Occupancy (1=Pornit, 0=Oprit)', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(-0.1, 1.1) 
    
    plt.title(f'Occupancy for {ROOM_TO_TEST} ({start_time.date()} - {end_time.date()})', fontsize=16)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9)) 
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

except FileNotFoundError:
    print(f"File '{input_csv_file}' was not found.")
    sys.exit(1)
except KeyError as e:
    print(f"Column not found {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)