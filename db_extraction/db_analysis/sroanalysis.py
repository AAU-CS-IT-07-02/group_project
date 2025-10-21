import pandas as pd
import matplotlib
# needed for running on a server without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse
from datetime import datetime
from datasets import load_dataset

# --- CONFIG ---
HF_REPO = "AAU-CS-IT-07-02/AAU-BUILD-sensor.actuator"
DATA_FILE = "6roomsOffice/dataset_with_occupancy_delimiter_comma.csv"
# ------------

def get_data(repo, file):
    """
    Downloads and preprocesses the main dataframe from Hugging Face.

    This function handles loading the specified dataset, cleaning it by
    dropping unnecessary columns, and setting a proper datetime index
    for time-series analysis.

    Args:
        repo (str): The repository ID on the Hugging Face Hub.
        file (str): The path to the specific data file within the repository.

    Returns:
        pd.DataFrame or None: A preprocessed pandas DataFrame if successful,
                              otherwise None.
    """
    print(f"--- Grabbing data from Hugging Face: '{repo}' ---")
    try:
        dataset = load_dataset(repo, data_files=file, split='train')
        df = dataset.to_pandas()
        print("...success.")

        # clean up the dataframe a bit
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # find the timestamp column, because the name might change
        if 'timestamp' in df.columns:
            ts_col = 'timestamp'
        else:
            print("ERROR: Could not find a 'timestamp' column.")
            return None

        # make sure the timestamp is the index for our time-series analysis
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df.set_index(ts_col, inplace=True)

        print("Data preprocessed.")
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def generate_files(df, user_choices, start_date, end_date):
    """
    Creates graphs and a filtered CSV based on the user's choices.

    This function filters the main dataframe by date, selects columns
    based on user input, saves the filtered data to a new CSV, and
    generates a separate hourly graph for each selected column.

    Args:
        df (pd.DataFrame): The main dataframe, preprocessed by get_data().
        user_choices (list): A list of strings representing the user's
                             analysis selections (e.g., ['A', 'HEATING']).
        start_date (str): The start date for filtering (DD-MM-YYYY).
        end_date (str): The end date for filtering (DD-MM-YYYY).
    """
    print(f"\n--- Preparing data for analysis ---")
    start_aware = pd.to_datetime(start_date, dayfirst=True, utc=True)
    end_aware = pd.to_datetime(end_date, dayfirst=True, utc=True) + pd.Timedelta(days=1)
    
    df_filtered = df.loc[start_aware:end_aware]
    duration_days = (end_aware - start_aware).days

    # figure out what to plot based on user input
    cols_to_plot = set()
    for choice in user_choices:
        if choice == 'VENTILATION':
            cols_to_plot.update({col for col in df.columns if 'ventilation' in col.lower()})
        elif choice == 'HEATING':
            cols_to_plot.update({col for col in df.columns if col.startswith('Heating:')})
        elif choice == 'OUTDOOR':
            cols_to_plot.update({col for col in df.columns if col.startswith('Outdoor:')})
        else: # It's a room letter
            cols_to_plot.update({col for col in df.columns if col.startswith(f'Room{choice}:')})

    if not cols_to_plot:
        print(f"ERROR: No columns found for your choices: {', '.join(user_choices)}")
        return
        
    cols_to_plot = sorted(list(cols_to_plot))
    
    # save the filtered data to a new csv
    choice_str = "+".join(user_choices)
    output_csv_name = f"filtered_data_{choice_str}_from_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}.csv"
    try:
        df_filtered[cols_to_plot].to_csv(output_csv_name)
        print(f"\nSuccessfully saved filtered data to '{output_csv_name}'")
    except Exception as e:
        print(f"\nCould not save filtered data file. Error: {e}")

    # now, make the graphs
    output_folder = 'graphs'
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nGraphs will be saved in the '{output_folder}' folder.")

    for col in cols_to_plot:
        df_hourly = df_filtered[[col]].resample('h').mean()

        if df_hourly[col].isnull().all():
            continue

        print(f"Creating graph for '{col}'...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(df_hourly.index, df_hourly[col], label=col, marker='o', ls='-', ms=4)

        ax.set_title(f'Hourly Average for {col}\n({start_date} to {end_date})', fontsize=16)
        ax.set_xlabel('Date and Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, which='both')

        # format the x-axis to be more readable depending on the time range
        if duration_days <= 3:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%d-%b-%Y'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()

        # make sure the filename is safe for any OS
        safe_filename = col.replace(':', '_').replace('__', '_').replace('/', '_')
        graph_path = os.path.join(output_folder, f"{safe_filename}.png")
        
        try:
            plt.savefig(graph_path, dpi=300)
            print(f"  -> Saved to '{graph_path}'")
        except Exception as e:
            print(f"  -> Could not save the graph. Error: {e}")
        plt.close(fig)


def main():
    """
    Handles command line arguments and runs the main analysis workflow.
    
    Parses start date, end date, and analysis choices from the command line,
    loads the data, validates the inputs, and then calls the function to
    generate the output files.
    """
    parser = argparse.ArgumentParser(
        description="Analyze building sensor data from Hugging Face.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("start_date", help="Start date (DD-MM-YYYY)")
    parser.add_argument("end_date", help="End date (DD-MM-YYYY)")
    parser.add_argument(
        "choices",
        help="One or more things to analyze (e.g., A, B, HEATING, etc.)",
        nargs='+',
        choices=['A', 'B', 'C', 'D', 'E', 'F', 'VENTILATION', 'HEATING', 'OUTDOOR', 'a', 'b', 'c', 'd', 'e', 'f', 'ventilation', 'heating', 'outdoor']
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    user_choices = [choice.upper() for choice in args.choices]

    df = get_data(HF_REPO, DATA_FILE)
    if df is None:
        return # stop if data loading failed

    min_date, max_date = df.index.min().date(), df.index.max().date()

    # Validate the dates to make sure they're within the dataset range
    try:
        dt_start = datetime.strptime(args.start_date, '%d-%m-%Y').date()
        dt_end = datetime.strptime(args.end_date, '%d-%m-%Y').date()
        if not (min_date <= dt_start <= max_date and dt_end >= dt_start and dt_end <= max_date):
             print(f"Error: Date range is invalid or outside available data ({min_date:%d-%m-%Y} to {max_date:%d-%m-%Y}).")
             sys.exit(1)
    except ValueError:
        print("Invalid date format. Please use DD-MM-YYYY.")
        sys.exit(1)

    print(f"\n--- Running analysis for '{'+'.join(user_choices)}' from {args.start_date} to {args.end_date} ---")
    generate_files(df, user_choices, args.start_date, args.end_date)


if __name__ == '__main__':
    main()

