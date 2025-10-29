import argparse

import pandas as pd

from contextlib import contextmanager
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

import pysindy as ps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic modeling of smart building systems using sparse identification techniques.")
    
    # Data file paths
    p.add_argument("--sensors", default="../data_fragmentation/out/data_sensors.csv", help="Path to sensors data file.")
    p.add_argument("--actuators", default="../data_fragmentation/out/data_actuators.csv", help="Path to actuators data file.")
    p.add_argument("--configuration", default="../data_fragmentation/out/data_configuration.csv", help="Path to configuration data file.")
    p.add_argument("--outdir", default="out", help="Output directory for results.")
    p.add_argument("--sep", default=None, help="CSV delimiter. If omitted, auto-detect.")
    
    # Data processing hyperparameters
    p.add_argument("--interpolation-method", default="linear", 
                   choices=["linear", "time", "index", "nearest", "zero", "slinear", "quadratic", "cubic"],
                   help="Method for interpolating missing values")
    p.add_argument("--include-configuration", action="store_true", 
                   help="Include configuration variables in state vector (default: sensors only)")
    p.add_argument("--dt", type=float, default=0.1, 
                   help="Time step between measurements")
    
    # SINDy model hyperparameters
    p.add_argument("--polynomial-degree", type=int, default=2, 
                   help="Degree of polynomial features for SINDy")
    p.add_argument("--threshold", type=float, default=0.1, 
                   help="Sparsity threshold for STLSQ optimizer")
    p.add_argument("--alpha", type=float, default=0.0, 
                   help="Regularization parameter for STLSQ optimizer")
    p.add_argument("--max-iter", type=int, default=20, 
                   help="Maximum iterations for STLSQ optimizer")
    p.add_argument("--normalize-columns", action="store_true", 
                   help="Normalize feature matrix columns")
    
    # Training/validation hyperparameters
    p.add_argument("--train-split", type=float, default=0.7, 
                   help="Fraction of data to use for training (rest for validation)")
    
    # Data normalization options
    p.add_argument("--normalize-data", action="store_true", 
                   help="Normalize data before training")
    p.add_argument("--normalization-method", default="minmax", choices=["minmax", "standard", "robust"], 
                   help="Data normalization method")
    
    # Feature library options
    p.add_argument("--feature-library", default="polynomial", choices=["polynomial", "fourier", "identity"], 
                   help="Type of feature library to use")
    p.add_argument("--fourier-n-frequencies", type=int, default=2, 
                   help="Number of frequencies for Fourier library")
    
    # Optimizer options
    p.add_argument("--optimizer", default="stlsq", choices=["stlsq", "lasso", "ridge"], 
                   help="Optimizer type for SINDy")
    p.add_argument("--lasso-alpha", type=float, default=0.01, 
                   help="Alpha parameter for Lasso optimizer")
    
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

def process_data_simple_interpolation(sensors_data: np.ndarray, actuators_data: np.ndarray, 
                                    configuration_data: np.ndarray, interpolation_method: str = "linear") -> tuple:
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
    interpolation_method : str
        Method for interpolation ('linear', 'cubic', 'spline', 'polynomial')
        
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
    
    # Simple interpolation with specified method
    sensors_clean = sensors_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    actuators_clean = actuators_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    configuration_clean = configuration_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    
    # Check if any NaN values remain
    remaining_nans = np.isnan(sensors_clean).sum() + np.isnan(actuators_clean).sum() + np.isnan(configuration_clean).sum()
    print(f"Remaining NaN values after interpolation: {remaining_nans}")
    
    return sensors_clean, actuators_clean, configuration_clean

def create_feature_library(library_type: str, polynomial_degree: int = 2, fourier_n_frequencies: int = 2):
    """Create feature library based on specified type."""
    if library_type == "polynomial":
        return ps.PolynomialLibrary(degree=polynomial_degree)
    elif library_type == "fourier":
        return ps.FourierLibrary(n_frequencies=fourier_n_frequencies)
    elif library_type == "identity":
        return ps.IdentityLibrary()
    else:
        raise ValueError(f"Unknown feature library type: {library_type}")

def create_optimizer(optimizer_type: str, threshold: float = 0.1, alpha: float = 0.0, 
                    max_iter: int = 20, normalize_columns: bool = False, lasso_alpha: float = 0.01):
    """Create optimizer based on specified type."""
    if optimizer_type == "stlsq":
        return ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=max_iter, normalize_columns=normalize_columns)
    # TODO: Finish adding all the optimizers and their parameters to the constructor
    # elif optimizer_type == "ssr":
    #     return ps.SSR(threshold=threshold, alpha=alpha, max_iter=max_iter, normalize_columns=normalize_columns)
    # elif optimizer_type == "frols":
    #     return ps.FROLS(threshold=threshold, max_iter=max_iter, normalize_columns=normalize_columns)
    # elif optimizer_type == "sr3":
    #     return ps.SR3(threshold=threshold, nu=alpha, max_iter=max_iter, normalize_columns=normalize_columns)
    # elif optimizer_type == "constrainedsr3":
    #     return ps.ConstrainedSR3(threshold=threshold, nu=alpha, max_iter=max_iter, normalize_columns=normalize_columns)
    # elif optimizer_type == "miosr":
    #     return ps.MIOSR(target_sparsity=int(1/threshold) if threshold > 0 else 10, normalize_columns=normalize_columns)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def normalize_data(X: np.ndarray, U: np.ndarray, method: str = "minmax"):
    """Normalize data using specified method."""
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
    
    # Interpolate missing values
    print(f"\nInterpolating missing values using {args.interpolation_method} method...")
    sensors_data, actuators_data, configuration_data = process_data_simple_interpolation(
        sensors_data, actuators_data, configuration_data, args.interpolation_method
    )
    
    # Combine data based on configuration
    if args.include_configuration:
        X = np.hstack([sensors_data, configuration_data])
        print("Using sensors + configuration as state variables")
    else:
        X = sensors_data
        print("Using sensors only as state variables")
    
    # Use actuators as control inputs (U)
    U = actuators_data
    
    # Create time vector
    t = np.arange(len(X)) * args.dt
    
    print(f"\nPrepared for SINDy:")
    print(f"  States X: {X.shape}")
    print(f"  Controls U: {U.shape}")
    print(f"  Time points: {len(t)}")
    print(f"  dt: {args.dt}")
    
    # Optionally normalize data
    if args.normalize_data:
        print(f"\nNormalizing data using {args.normalization_method} method...")
        X, U, X_scaler, U_scaler = normalize_data(X, U, args.normalization_method)
    
    # Create feature library and optimizer
    feature_library = create_feature_library(args.feature_library, args.polynomial_degree, args.fourier_n_frequencies)
    optimizer = create_optimizer(args.optimizer, args.threshold, args.alpha, args.max_iter, 
                                args.normalize_columns, args.lasso_alpha)
    
    # Build SINDy model
    print(f"\nTraining SINDy model...")
    print(f"  Feature library: {args.feature_library} (degree={args.polynomial_degree})")
    print(f"  Optimizer: {args.optimizer} (threshold={args.threshold})")
    
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    # Train the model
    model.fit(X, u=U, t=args.dt)
    
    # Print discovered equations
    print("\nDiscovered equations:")
    model.print()
    
    # Simple validation: split data and test prediction
    split = int(args.train_split * len(X))
    X_train, X_test = X[:split], X[split:]
    U_train, U_test = U[:split], U[split:]
    t_test = np.arange(len(X_test)) * args.dt
    
    print(f"\nValidation with {args.train_split:.1%} train / {1-args.train_split:.1%} test split...")
    
    # Retrain on training data only
    model.fit(X_train, u=U_train, t=args.dt)
    
    # Predict on test data
    try:
        X_pred = model.simulate(X_test[0], t_test, u=U_test)
        
        # Calculate error
        min_len = min(len(X_test), len(X_pred))
        # TODO: are there more interesting error metrics?
        rmse = np.sqrt(np.mean((X_test[:min_len] - X_pred[:min_len])**2))
        
        print(f"\nValidation RMSE: {rmse:.6f}")
        
        # Simple plot (fixed visualization parameters)
        # TODO: plot all states on a multiplot figure
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