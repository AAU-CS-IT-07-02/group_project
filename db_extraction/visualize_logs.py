import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual CSV filename
csv_file = "./TMV23_2025_02_28_MIN_2024_1__2024_1.csv"

# Read the CSV file
df = pd.read_csv(csv_file, parse_dates=['time'])

# Set 'time' as the DataFrame index for easier plotting
df.set_index('time', inplace=True)

# Plot all columns except 'time'
df.plot(subplots=True, figsize=(12, 10), title="Sensor Data Over Time")
plt.tight_layout()
plt.show()