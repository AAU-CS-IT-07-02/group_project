"""
Data Processing for Dynamic Building Modeling

This module handles all data-related operations including:
- CSV data loading and preprocessing
- Missing value interpolation
- Data normalization and scaling
- Data downsampling for performance optimization
- State space preparation for SINDy modeling

Real building management systems often have missing data due to sensor failures,
communication issues, or maintenance periods. This module provides robust
preprocessing capabilities to handle such real-world data challenges.

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any
import argparse


def get_csv_data(file_path: str, delimiter: str = ',', skip_header: bool = False, drop_timestamps: bool = True, return_names: bool = False):
    """
    Load a CSV file into a NumPy array, optionally returning column names.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file.
    delimiter : str, optional
        Column separator (default is ',').
    skip_header : bool, optional
        If True, skip the header row (default is False).
    drop_timestamps : bool, optional
        If True, drop the first column assuming it contains timestamps.
    return_names : bool, optional
        If True, return tuple (data, column_names), else return just data.

    Returns:
    ------
    np.ndarray or tuple
        If return_names=False: Data from the CSV as a NumPy array.
        If return_names=True: Tuple (data, column_names) where column_names is a list.
    """
    # Read CSV using pandas for flexibility
    df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8-sig', engine='python')

    # Get column names before any processing
    column_names = list(df.columns)

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
        if return_names:
            column_names = column_names[1:]

    
    return data, column_names


def process_data_simple_interpolation(sensors_data: np.ndarray, actuators_data: np.ndarray, 
                                    configuration_data: np.ndarray, interpolation_method: str = "linear") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate missing values in building sensor and actuator data.
    
    Real building management systems often have missing data due to sensor failures,
    communication issues, or maintenance periods. This function handles missing values
    using pandas interpolation methods with forward/backward filling for edge cases.
    
    Parameters:
        sensors_data: Sensor measurements (temperature, CO2, humidity, occupancy)
        actuators_data: Actuator states (HVAC setpoints, ventilation rates, blind positions)
        configuration_data: System configuration parameters
        interpolation_method: Interpolation strategy ('linear', 'cubic', 'spline', etc.)
        
    Returns:
        tuple: (sensors_clean, actuators_clean, configuration_clean) with interpolated values
        
    Note:
        Reports the number of missing values before and after interpolation for data quality assessment.
    """
    # Convert to DataFrames for easy interpolation
    sensors_df = pd.DataFrame(sensors_data)
    actuators_df = pd.DataFrame(actuators_data)
    configuration_df = pd.DataFrame(configuration_data)
    
    # Count initial NaN values
    total_nans = sensors_df.isna().sum().sum() + actuators_df.isna().sum().sum() + configuration_df.isna().sum().sum()
    print(f"Total missing values: {total_nans}")
    
    # Simple interpolation with specified method
    sensors_clean = sensors_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    actuators_clean = actuators_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    configuration_clean = configuration_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    
    # Check if any NaN values remain
    remaining_nans = np.isnan(sensors_clean).sum() + np.isnan(actuators_clean).sum() + np.isnan(configuration_clean).sum()
    print(f"Remaining NaN values after interpolation: {remaining_nans}")
    
    return sensors_clean, actuators_clean, configuration_clean


def normalize_data(X: np.ndarray, U: np.ndarray, method: str = "minmax") -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Normalize building sensor and actuator data for numerical stability.
    
    Building data involves variables with very different scales:
    - Temperature: ~15-30°C
    - CO2: ~400-2000 ppm  
    - Occupancy: 0-50 people
    - HVAC setpoints: 0-100%
    
    Normalization ensures all variables contribute equally to the sparse regression.
    
    Parameters:
        X: State variables (sensor measurements)
        U: Control inputs (actuator commands)
        method: Normalization method ('minmax', 'standard', 'robust')
        
    Returns:
        tuple: (X_normalized, U_normalized, X_scaler, U_scaler)
            Normalized data arrays and fitted scaler objects for inverse transformation
            
    Raises:
        ValueError: If unknown normalization method is specified
    """
    if method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        X_scaler = MinMaxScaler()
        U_scaler = MinMaxScaler()
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler
        X_scaler = StandardScaler()
        U_scaler = StandardScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        X_scaler = RobustScaler()
        U_scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_normalized = X_scaler.fit_transform(X)
    U_normalized = U_scaler.fit_transform(U)
    
    return X_normalized, U_normalized, X_scaler, U_scaler


def downsample_data(sensors_data: np.ndarray, actuators_data: np.ndarray, configuration_data: np.ndarray, sampling_rate: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample building data for faster training by taking every N-th sample.
    
    Parameters:
        sensors_data: Sensor measurements array
        actuators_data: Actuator commands array
        configuration_data: Configuration data array
        sampling_rate: Take every N-th sample (1=no downsampling, 10=10x speedup)
        
    Returns:
        tuple: (sensors_downsampled, actuators_downsampled, configuration_downsampled)
    """
    if sampling_rate <= 1:
        return sensors_data, actuators_data, configuration_data
    
    # Take every N-th sample
    indices = np.arange(0, len(sensors_data), sampling_rate)
    return sensors_data[indices], actuators_data[indices], configuration_data[indices]


def load_and_process_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[Any, Any]]]:
    """
    Complete data loading and preprocessing pipeline.
    
    This function orchestrates the entire data preprocessing workflow:
    1. Load CSV files for sensors, actuators, and configuration
    2. Convert to float and handle type conversion errors
    3. Interpolate missing values using specified method
    4. Downsample data for performance optimization (if requested)
    5. Combine data based on configuration (sensors only vs sensors+config)
    6. Normalize data for numerical stability (if requested)
    7. Create time vector for SINDy modeling
    8. Extract column names if requested for named variables
    
    Parameters:
        args: Configuration namespace containing all processing parameters
        
    Returns:
        tuple: (X, U, t, scalers, feature_names)
            - X: State variables (sensor measurements, optionally with configuration)
            - U: Control inputs (actuator commands)  
            - t: Time vector
            - scalers: Optional tuple of (X_scaler, U_scaler) if normalization was applied
            - feature_names: Optional list of variable names for SINDy equations (if named_variables=True)
            
    The returned data is ready for SINDy model training and follows the format:
        X: (n_timesteps, n_state_variables)
        U: (n_timesteps, n_control_inputs)
        t: (n_timesteps,)
    """
    print("Loading data...")
    
    # Load data from CSV files, optionally extract column names
    feature_names = None
    if args.named_variables:
        sensors_df, sensors_names = get_csv_data(args.sensors, args.sep, return_names=True)
        actuators_df, actuators_names = get_csv_data(args.actuators, args.sep, return_names=True)
        configuration_df, configuration_names = get_csv_data(args.configuration, args.sep, return_names=True)
    else:
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
    
    # Interpolate missing values
    print(f"\nInterpolating missing values using {args.interpolation_method} method...")
    sensors_data, actuators_data, configuration_data = process_data_simple_interpolation(
        sensors_data, actuators_data, configuration_data, args.interpolation_method
    )
    
    # Downsample data for faster training
    original_size = len(sensors_data)
    if args.sampling_rate > 1:
        print(f"\nDownsampling data (every {args.sampling_rate} samples)...")
        sensors_data, actuators_data, configuration_data = downsample_data(
            sensors_data, actuators_data, configuration_data, args.sampling_rate
        )
        print(f"  Original size: {original_size:,} timesteps")
        print(f"  Downsampled size: {len(sensors_data):,} timesteps")
        print(f"  Speedup: {args.sampling_rate}x")
    else:
        print(f"No downsampling applied (using full dataset)")
    
    # Combine data based on configuration
    if args.include_configuration:
        X = np.hstack([sensors_data, configuration_data])
        print(f"\nUsing sensors + configuration as state variables")
        # Combine feature names if named_variables is enabled
        if args.named_variables:
            # State variables (X) names: sensors + configuration
            X_names = sensors_names + configuration_names
    else:
        X = sensors_data
        print(f"\nUsing sensors only as state variables")
        # Use only sensor names if named_variables is enabled
        if args.named_variables:
            # State variables (X) names: sensors only
            X_names = sensors_names
    
    # Use actuators as control inputs (U)
    U = actuators_data
    
    # Combine all feature names: state variables (X) + control inputs (U)
    if args.named_variables:
        feature_names = X_names + actuators_names
        print(f"  State variable names (X): {X_names}")
        print(f"  Control variable names (U): {actuators_names}")
    else:
        feature_names = None
    
    # Optionally normalize data
    scalers = None
    if args.normalize_data:
        print(f"Normalizing data using {args.normalization_method} method...")
        X, U, X_scaler, U_scaler = normalize_data(X, U, args.normalization_method)
        scalers = (X_scaler, U_scaler)
    
    # Create time vector
    t = np.arange(len(X)) * args.dt
    
    print(f"\nFinal data prepared for SINDy:")
    print(f"  States X: {X.shape}")
    print(f"  Controls U: {U.shape}")
    print(f"  Time points: {len(t)}")
    print(f"  dt: {args.dt}")
    if args.named_variables and feature_names:
        print(f"  Total feature names ({len(feature_names)}): {feature_names}")
        print(f"    - State variables (X): {len(X_names)}")
        print(f"    - Control variables (U): {len(actuators_names)}")
    
    return X, U, t, scalers, feature_names


def prepare_training_data(X: np.ndarray, U: np.ndarray, t: np.ndarray, train_split: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets for temporal validation.
    
    Parameters:
        X: State variables
        U: Control inputs
        t: Time vector
        train_split: Fraction of data to use for training
        
    Returns:
        tuple: (X_train, X_test, U_train, U_test, t_train, t_test)
    """
    split_idx = int(train_split * len(X))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    U_train, U_test = U[:split_idx], U[split_idx:]
    t_train, t_test = t[:split_idx], t[split_idx:]
    
    return X_train, X_test, U_train, U_test, t_train, t_test