import pandas as pd
import matplotlib
# Set a non-GUI backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse
from datetime import datetime

# --- SETTINGS ---
# -----------------------------------------------------------------------------
# 1. Filenames
#    The script now uses a relative path to find the data file.
#    This assumes the script is in '.../db_analysis/' and the data
#    is in '.../AAU-BUILD-sensor.actuator/6roomsOffice/'.
input_filename = '../../AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy.csv'
# -----------------------------------------------------------------------------


def load_and_prepare_data(input_file):
    """
    Loads and preprocesses the dataset from the CSV file.
    - Sets a proper datetime index.
    - Handles file not found and other loading errors.
    """
    print(f"--- Loading data from '{input_file}' ---")
    try:
        # Use comma as delimiter, as seen in the new file format
        df = pd.read_csv(input_file, delimiter=',')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Use the correct 'timestamp' column name from your sample file
        if 'timestamp' in df.columns:
            timestamp_col = 'timestamp'
        elif 'Date/Time' in df.columns: # Keep as fallback
             timestamp_col = 'Date/Time'
        else:
            print("ERROR: Could not find a 'timestamp' or 'Date/Time' column in the CSV.")
            return None

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df.set_index(timestamp_col, inplace=True)

        print("Data loaded and preprocessed successfully.")
        return df
    except FileNotFoundError:
        print(f"ERROR: The file '{input_file}' was not found.")
        print("Please check that the path in the script is correct and you are running it from the 'db_analysis' folder.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None


def generate_files(df, analysis_choices, start_str, end_str):
    """
    Saves a filtered (untransposed) CSV and creates time-series graphs
    for the selected data.

    Args:
        df (pd.DataFrame): The pre-loaded and prepared DataFrame.
        analysis_choices (list): A list of entities to analyze (e.g., ['A', 'C', 'VENTILATION']).
        start_str (str): The start date for the graph ('DD-MM-YYYY').
        end_str (str): The end date for the graph ('DD-MM-YYYY').
    """
    # --- 1. Filter Data and Identify Columns ---
    print(f"\n--- Preparing data for analysis ---")
    start_aware = pd.to_datetime(start_str, dayfirst=True, utc=True)
    end_aware = pd.to_datetime(end_str, dayfirst=True, utc=True) + pd.Timedelta(days=1)
    df_filtered = df.loc[start_aware:end_aware]

    duration_days = (end_aware - start_aware).days

    # Find all columns that match any of the user's choices
    columns_to_plot = set() # Use a set to avoid duplicate columns
    for choice in analysis_choices:
        if choice == 'VENTILATION':
            vent_cols = {col for col in df.columns if 'ventilation' in col.lower()}
            columns_to_plot.update(vent_cols)
        elif choice == 'HEATING':
            heat_cols = {col for col in df.columns if col.startswith('Heating:')}
            columns_to_plot.update(heat_cols)
        else: # It's a room letter
            room_prefix = f'Room{choice}:'
            room_cols = {col for col in df.columns if col.startswith(room_prefix)}
            columns_to_plot.update(room_cols)

    if not columns_to_plot:
        print(f"ERROR: No columns found for your choices '{', '.join(analysis_choices)}'. Please check your input.")
        return

    # Convert set to a sorted list for consistent order
    columns_to_plot = sorted(list(columns_to_plot))


    # --- 2. Save the Filtered (Untransposed) Data ---
    df_to_save = df_filtered[columns_to_plot]
    safe_start = start_str.replace('-', '')
    safe_end = end_str.replace('-', '')
    choice_str = "+".join(analysis_choices)
    output_csv_name = f"filtered_data_{choice_str}_from_{safe_start}_to_{safe_end}.csv"
    try:
        df_to_save.to_csv(output_csv_name)
        print(f"\nSuccessfully saved filtered data to '{output_csv_name}'")
    except Exception as e:
        print(f"\nCould not save filtered data file. Error: {e}")

    # --- 3. Create and Save Graphs ---
    output_folder = 'graphs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"\nGraphs will be saved in the '{output_folder}' folder.")

    for column in columns_to_plot:
        # Updated resample from 'H' to 'h' to avoid FutureWarning
        df_hourly = df_filtered[[column]].resample('h').mean()

        if df_hourly[column].isnull().all():
            print(f"Skipping '{column}' - no data in the selected time frame.")
            continue

        print(f"Creating graph for '{column}'...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(df_hourly.index, df_hourly[column], label=column, marker='o', linestyle='-', markersize=4)

        title = f'Hourly Average for {column}\n({start_str} to {end_str})'
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date and Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, which='both')

        if duration_days <= 3:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%d-%b-%Y'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            ax.xaxis.set_minor_locator(None)

        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()

        # Clean filename for saving
        safe_filename = column.replace(':', '_').replace('__', '_').replace('/', '_')
        graph_filename = f"{safe_filename}.png"
        graph_path = os.path.join(output_folder, graph_filename)
        try:
            plt.savefig(graph_path, dpi=300)
            print(f"  -> Saved to '{graph_path}'")
        except Exception as e:
            print(f"  -> Could not save the graph. Error: {e}")
        plt.close(fig)


def main():
    """
    Main function to parse command-line arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze building sensor data for specific rooms or for ventilation.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "start_date",
        help="The start date for the analysis in DD-MM-YYYY format."
    )
    parser.add_argument(
        "end_date",
        help="The end date for the analysis in DD-MM-YYYY format."
    )
    parser.add_argument(
        "choices",
        help="One or more entities to analyze. Choose from:\n"
             "A, B, C, D, E, F - for a specific room\n"
             "VENTILATION - for all ventilation-related data\n"
             "HEATING - for all heating-related data",
        nargs='+', # This allows for one or more arguments
        choices=['A', 'B', 'C', 'D', 'E', 'F', 'VENTILATION', 'HEATING', 'a', 'b', 'c', 'd', 'e', 'f', 'ventilation', 'heating']
    )

    # If run with no arguments, print help and example usage
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\nExample Usage:")
        print("  python sroanalysis.py 01-04-2023 02-04-2023 A")
        print("  python sroanalysis.py 15-05-2023 15-05-2023 B HEATING VENTILATION")
        sys.exit(1)

    args = parser.parse_args()

    # Process arguments
    start_date_input = args.start_date
    end_date_input = args.end_date
    # Convert all choices to uppercase
    choice_inputs = [choice.upper() for choice in args.choices]


    df = load_and_prepare_data(input_filename)
    if df is None:
        return  # Stop if data loading failed

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    # --- Validate Dates ---
    try:
        dt_start = datetime.strptime(start_date_input, '%d-%m-%Y').date()
        dt_end = datetime.strptime(end_date_input, '%d-%m-%Y').date()

        if not (min_date <= dt_start <= max_date):
            print(f"Error: Start date {start_date_input} is outside the available data range ({min_date.strftime('%d-%m-%Y')} to {max_date.strftime('%d-%m-%Y')}).")
            sys.exit(1)

        if dt_end < dt_start:
            print("Error: End date cannot be before the start date.")
            sys.exit(1)

        if dt_end > max_date:
            print(f"Error: End date {end_date_input} is outside the available data range ({min_date.strftime('%d-%m-%Y')} to {max_date.strftime('%d-%m-%Y')}).")
            sys.exit(1)

    except ValueError:
        print("Invalid date format. Please use DD-MM-YYYY.")
        sys.exit(1)

    print(f"\n--- Running analysis for '{'+'.join(choice_inputs)}' from {start_date_input} to {end_date_input} ---")
    generate_files(df, choice_inputs, start_date_input, end_date_input)


if __name__ == '__main__':
    main()

