import os
import pandas as pd

# =============================
# 1. Load original dataset
# =============================
CSV = "../../Database/AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv"
df = pd.read_csv(CSV, parse_dates=['timestamp'])


# Parse timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')

# =============================
# 2. Split by month
# =============================
test_month = 3  # Example: February for testing
df['month'] = df['timestamp'].dt.month

train_df = df[df['month'] != test_month].drop(columns=['month'])
test_df = df[df['month'] == test_month].drop(columns=['month'])

# =============================
# 3. Save to separate CSV files
# =============================
os.makedirs("./dataset_split", exist_ok=True)

train_file = "./dataset_split/train_data.csv"
test_file = "./dataset_split/test_data.csv"
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Train CSV saved as {train_file} with {len(train_df)} rows.")
print(f"Test CSV saved as {test_file} with {len(test_df)} rows.")

