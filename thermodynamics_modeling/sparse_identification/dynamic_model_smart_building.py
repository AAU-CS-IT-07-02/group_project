import argparse

import pandas as pd

import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import pysindy as ps
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split data CSV into Sensors/Actuators/Configuration CSVs.")
    # p.add_argument("--data", required=True, help="Path to the master data CSV.")
    p.add_argument("--sensors", default="../data_fragmentation/out/data_sensors.csv", help="Path to sensors list file.")
    p.add_argument("--actuators", default="../data_fragmentation/out/data_actuators.csv", help="Path to actuators list file.")
    p.add_argument("--configuration", default="../data_fragmentation/out/data_configuration.csv", help="Path to configuration list file.")
    p.add_argument("--outdir", default="out", help="Output directory for the split CSV files.")
    p.add_argument("--sep", default=None, help="CSV delimiter. If omitted, auto-detect.")
    return p.parse_args()

def get_csv_data(file_path: str, delimiter: str = ',', skip_header: bool = False, drop_timestamps: bool = True) -> np.ndarray:
    """
    Load a CSV file into a NumPy array.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file.
    delimiter : str, optional
        Column separator (default is ',').
    skip_header : bool, optional
        If True, skip the header row (default is False).

    Returns:
    ------
    np.ndarray
        Data from the CSV as a NumPy array.
    """
    # Read CSV using pandas for flexibility
    df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8-sig', engine='python')

    # Optionally drop header row
    if skip_header:
        data = df.to_numpy()
    else:
        # Keep header as first row if needed
        data = df.to_numpy()
        
    if drop_timestamps:
        # TODO: think about a better way to identify timestamp columns
        # Assume first column is timestamp, drop it
        data = data[:, 1:]

    return data

def process_data(sensors_data: np.ndarray, actuators_data: np.ndarray, configuration_data: np.ndarray) -> tuple:
    """
    Filter out rows with missing values (NaN) from all three datasets.
    If any row has a NaN in any column across any of the three datasets,
    that row is removed from all datasets to maintain alignment.
    
    Parameters:
    ----------
    sensors_data : np.ndarray
        Sensor data array.
    actuators_data : np.ndarray
        Actuator data array.
    configuration_data : np.ndarray
        Configuration data array.
        
    Returns:
    -------
    tuple
        (sensors_clean, actuators_clean, configuration_clean) with NaN rows removed.
    """
    # Check that all arrays have the same number of rows
    if not (len(sensors_data) == len(actuators_data) == len(configuration_data)):
        raise ValueError(
            f"Data arrays must have the same number of rows. "
            f"Got: sensors={len(sensors_data)}, actuators={len(actuators_data)}, "
            f"configuration={len(configuration_data)}"
        )
    
    # Find rows with any NaN values across all datasets
    sensors_nan_mask = np.isnan(sensors_data).any(axis=1)
    actuators_nan_mask = np.isnan(actuators_data).any(axis=1)
    configuration_nan_mask = np.isnan(configuration_data).any(axis=1)
    
    # Combined mask: True if ANY dataset has NaN in that row
    combined_nan_mask = sensors_nan_mask | actuators_nan_mask | configuration_nan_mask
    
    # Keep only rows without any NaN values
    valid_rows_mask = ~combined_nan_mask
    
    sensors_clean = sensors_data[valid_rows_mask]
    actuators_clean = actuators_data[valid_rows_mask]
    configuration_clean = configuration_data[valid_rows_mask]
    
    removed_rows = np.sum(combined_nan_mask)
    print(f"Removed {removed_rows} rows with missing values")
    print(f"Remaining data points: {len(sensors_clean)}")
    
    return sensors_clean, actuators_clean, configuration_clean

def process_data_simple_interpolation(sensors_data: np.ndarray, actuators_data: np.ndarray, configuration_data: np.ndarray) -> tuple:
    """
    Simple interpolation of missing values using pandas.
    
    Parameters:
    ----------
    sensors_data : np.ndarray
        Sensor data array.
    actuators_data : np.ndarray
        Actuator data array.
    configuration_data : np.ndarray
        Configuration data array.
        
    Returns:
    -------
    tuple
        (sensors_clean, actuators_clean, configuration_clean) with interpolated values.
    """
    # Convert to DataFrames for easy interpolation
    sensors_df = pd.DataFrame(sensors_data)
    actuators_df = pd.DataFrame(actuators_data)
    configuration_df = pd.DataFrame(configuration_data)
    
    # Count initial NaN values
    total_nans = sensors_df.isna().sum().sum() + actuators_df.isna().sum().sum() + configuration_df.isna().sum().sum()
    print(f"Total missing values: {total_nans}")
    
    # Simple linear interpolation
    sensors_clean = sensors_df.interpolate(method='linear').bfill().ffill().values
    actuators_clean = actuators_df.interpolate(method='linear').bfill().ffill().values
    configuration_clean = configuration_df.interpolate(method='linear').bfill().ffill().values
    
    # Check if any NaN values remain
    remaining_nans = np.isnan(sensors_clean).sum() + np.isnan(actuators_clean).sum() + np.isnan(configuration_clean).sum()
    print(f"Remaining NaN values after interpolation: {remaining_nans}")
    
    return sensors_clean, actuators_clean, configuration_clean

def main():
    args = parse_args()

    # Load data
    print("Loading data...")
    sensors_df = get_csv_data(args.sensors, args.sep)
    actuators_df = get_csv_data(args.actuators, args.sep)
    configuration_df = get_csv_data(args.configuration, args.sep)
    
    print(f"Loaded:")
    print(f"  Sensors: {sensors_df.shape}")
    print(f"  Actuators: {actuators_df.shape}")
    print(f"  Configuration: {configuration_df.shape}")
    
    # Convert to float (this may introduce NaN for invalid values)
    sensors_data = sensors_df.astype(float)
    actuators_data = actuators_df.astype(float)
    configuration_data = configuration_df.astype(float)
    
    # Simple interpolation
    print("\nInterpolating missing values...")
    sensors_data, actuators_data, configuration_data = process_data_simple_interpolation(
        sensors_data, actuators_data, configuration_data
    )
    
    # Combine sensors + configuration as states (X)
    X = np.hstack([sensors_data, configuration_data])
    
    # Use actuators as control inputs (U)
    U = actuators_data
    
    # Create time vector
    dt = 0.1  # Assume 0.1 time units between measurements
    t = np.arange(len(X)) * dt
    
    print(f"\nPrepared for SINDy:")
    print(f"  States X: {X.shape}")
    print(f"  Controls U: {U.shape}")
    print(f"  Time points: {len(t)}")
    
    # Build SINDy model
    print("\nTraining SINDy model...")
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2),
        optimizer=ps.STLSQ(threshold=0.1)
    )
    
    # Train the model
    # TODO: Consider normalizing data if needed
    # TODO: Find all of the "hyperparameters" of the script
    # TODO: Find a way to visualize the internal workings of the fit procedure, good for the report
    model.fit(X, u=U, t=dt)
    
    # Print discovered equations
    print("\nDiscovered equations:")
    model.print()
    
    # Simple validation: split data and test prediction
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    U_train, U_test = U[:split], U[split:]
    t_test = np.arange(len(X_test)) * dt
    
    # Retrain on training data only
    model.fit(X_train, u=U_train, t=dt)
    
    # Predict on test data
    try:
        X_pred = model.simulate(X_test[0], t_test, u=U_test)
        
        # Calculate error
        min_len = min(len(X_test), len(X_pred))
        rmse = np.sqrt(np.mean((X_test[:min_len] - X_pred[:min_len])**2))
        
        print(f"\nValidation RMSE: {rmse:.4f}")
        
        # Simple plot
        plt.figure(figsize=(12, 6))
        for i in range(min(3, X.shape[1])):  # Plot first 3 states
            plt.subplot(3, 1, i+1)
            plt.plot(X_test[:min_len, i], 'k-', label='True')
            plt.plot(X_pred[:min_len, i], 'r--', label='Predicted')
            plt.ylabel(f'State {i+1}')
            plt.legend()
        plt.xlabel('Time steps')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Simulation failed: {e}")

if __name__ == "__main__":
    main()