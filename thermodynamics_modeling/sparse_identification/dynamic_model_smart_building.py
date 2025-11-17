
"""
# Notes
Observation arrays follow the following axis conventions: (spatial_1, ..., spatial_n, time, coordinate). For ODEs (no spatial dependence), that means the first axis is time, the second axis is coordinate. pysindy also requires the timepoints of the observations. 
While there are several ways to pass this information, the most straightfowrads is a 1-D array of timepoints.

t, x, y = gen_data1()
X = np.stack((x, y), axis=-1)  # First column is x, second is y
print(f"Data is shape: {X.shape}")
print(f"time is shape: {t.shape}")
Data is shape: (50, 2)
time is shape: (50,)
"""
"""
# TODO:
- Rethink the state space + observations
"""

"""
Command reference:
python dynamic_model_smart_building.py --sensors ../data_fragmentation/out/R_A_All_May_NS/rooma_5_1_2023_09_05_5_30_2023_22_20/data_sensors.csv --actuators ../data_fragmentation/out/R_A_All_May_NS/rooma_5_1_2023_09_05_5_30_2023_22_20/data_actuators.csv --configuration ../data_fragmentation/out/R_A_All_May_NS/rooma_5_1_2023_09_05_5_30_2023_22_20/data_configuration.csv
"""

"""
Dynamic Modeling of Smart Building Systems using Sparse Identification of Nonlinear Dynamics.

This module implements data-driven thermodynamic modeling for the AAU BUILD facility
using PySINDy (Sparse Identification of Nonlinear Dynamics). The goal is to discover
interpretable mathematical equations governing building behavior from real sensor and
actuator data.

Key Features:
    - Robust handling of real-world building management system data
    - Multiple interpolation methods for missing sensor values  
    - Configurable feature libraries (polynomial, Fourier, identity)
    - Flexible normalization and optimization strategies
    - Temporal validation with predictive simulation (via custom_numba_simulation)
    - Comprehensive hyperparameter tuning capabilities

The discovered models serve as the foundation for Model Predictive Control (MPC)
implementation, providing both accuracy for control and interpretability for
safety verification in smart building applications.

Example Usage:
    ```bash
    # Standard usage
    python dynamic_model_smart_building.py \
        --sensors data_sensors.csv \
        --actuators data_actuators.csv \
        --polynomial-degree 2 \
        --threshold 0.1 \
        --normalize-data
    ```

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import argparse
import threading
import time

import pandas as pd

from contextlib import contextmanager
from copy import copy

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import psutil

# --- NEW IMPORTS (from scipy_numba.py) ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import custom_numba_simulation
# ----------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the dynamic building modeling script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - Data file paths (sensors, actuators, configuration)
            - Data processing parameters (interpolation, normalization)
            - SINDy model hyperparameters (polynomial degree, threshold)
            - Training/validation settings (train split, time step)
            - Feature library and optimizer options
    
    Example:
        ```bash
        python dynamic_model_smart_building.py --sensors data_sensors.csv --polynomial-degree 3 --threshold 0.05
        ```
    """
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
    p.add_argument("--dt", type=float, default=5, 
                   help="Time step between measurements")
    
    # SINDy model hyperparameters
    p.add_argument("--polynomial-degree", type=int, default=1,
                   help="Degree of polynomial features for SINDy. (Note: Numba simulation is hard-coded for degree 1)")
    p.add_argument("--threshold", type=float, default=0.1, 
                   help="Sparsity threshold for STLSQ optimizer")
    p.add_argument("--alpha", type=float, default=0.0, 
                   help="Regularization parameter for STLSQ optimizer")
    p.add_argument("--max-iter", type=int, default=20, 
                   help="Maximum iterations for STLSQ optimizer")
    p.add_argument("--normalize-columns", action="store_true",
                   help="Normalize feature matrix columns")
    p.add_argument("--coefficient-threshold", type=float, default=1000.0, 
                   help="Maximum allowed coefficient magnitude (for stability)")
    
    # Training/validation hyperparameters
    p.add_argument("--train-split", type=float, default=0.7, 
                   help="Fraction of data to use for training (rest for validation)")
    p.add_argument("--skip-validation", action="store_true", 
                   help="Skip validation step after training")
    p.add_argument("--skip-visualization", action="store_true", 
                   help="Skip plotting and visualization")
    p.add_argument("--validation-subsample", type=int, default=1, 
                   help="Subsample validation data for faster simulation (1=full, 10=every 10th sample)")
    p.add_argument("--simulation-timeout", type=int, default=60, 
                   help="Maximum time (seconds) to wait for simulation before timeout")
    
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
    p.add_argument("--no-interactions", action="store_true", 
                   help="Disable interaction terms in feature library (default: include interactions)")
    
    # Optimizer options
    p.add_argument("--optimizer", default="stlsq", choices=["stlsq", "lasso", "ridge"], 
                   help="Optimizer type for SINDy")
    p.add_argument("--lasso-alpha", type=float, default=0.01, 
                   help="Alpha parameter for Lasso optimizer")
    
    # System monitoring options
    p.add_argument("--monitor-interval", type=int, default=20, 
                   help="Interval in seconds for logging system usage (CPU, RAM, etc). Set to 0 to disable monitoring.")
    
    # Data sampling options for performance optimization
    p.add_argument("--sampling-rate", type=int, default=1, 
                   help="Downsample data by taking every N-th sample (1=no sampling, 10=10x speedup)")
    
    return p.parse_args()

def get_csv_data(file_path: str, delimiter: str = ',', drop_timestamps: bool = True) -> np.ndarray:
    """
    Load a CSV file into a NumPy array, forcing float64 dtype.

    This function uses pandas to read the CSV, which allows it to
    robustly handle non-numeric values (e.g., 'OFF', 'AUTO'). These
    values are coerced to `np.nan` and can be fixed later by interpolation.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    delimiter : str, optional
        Column separator (default is ',').
    drop_timestamps : bool, optional
        If True, assumes the first column is a timestamp and drops it.
        Defaults to True.

    Returns:
    ------
    np.ndarray
        Data from the CSV as a float64 NumPy array.
    """
    # Read CSV using pandas for flexibility
    df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8-sig', engine='python')

    if drop_timestamps:
        # TODO: think about a better way to identify timestamp columns
        # Assume first column is timestamp, drop it
        df = df.iloc[:, 1:]

    # Force all columns to numeric, coercing errors to NaN
    # This ensures the output is always float and never object-dtype
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Now convert to numpy, which will be float64
    data = df.to_numpy()

    return data

def process_data_simple_interpolation(sensors_data: np.ndarray, actuators_data: np.ndarray, 
                                    configuration_data: np.ndarray, interpolation_method: str = "linear") -> tuple:
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
    print(f"Total missing values (including coerced): {total_nans}")
    
    # Simple interpolation with specified method
    sensors_clean = sensors_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    actuators_clean = actuators_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    configuration_clean = configuration_df.interpolate(method=interpolation_method).bfill().ffill().values # type: ignore
    
    # Check if any NaN values remain
    remaining_nans = np.isnan(sensors_clean).sum() + np.isnan(actuators_clean).sum() + np.isnan(configuration_clean).sum()
    print(f"Remaining NaN values after interpolation: {remaining_nans}")
    
    return sensors_clean, actuators_clean, configuration_clean

def create_feature_library(library_type: str, polynomial_degree: int = 2, fourier_n_frequencies: int = 2, include_interactions: bool = True):
    """
    Create a PySINDy feature library for building dynamics modeling.
    
    Different feature libraries capture different aspects of building thermodynamics:
    - Polynomial: Nonlinear thermal relationships, heat transfer dependencies
    - Fourier: Daily/seasonal cycles, periodic occupancy patterns  
    - Identity: Linear relationships between variables
    
    Parameters:
        library_type: Type of feature library ('polynomial', 'fourier', 'identity')
        polynomial_degree: Maximum polynomial degree for nonlinear features
        fourier_n_frequencies: Number of frequency components for periodic patterns
        include_interactions: Whether to include interaction terms between variables
        
    Returns:
        PySINDy feature library object
        
    Raises:
        ValueError: If unknown library_type is specified
    """
    """Create feature library based on specified type."""
    if library_type == "polynomial":
        return ps.PolynomialLibrary(degree=polynomial_degree, include_interaction=include_interactions)
    elif library_type == "fourier":
        return ps.FourierLibrary(n_frequencies=fourier_n_frequencies)
    elif library_type == "identity":
        return ps.IdentityLibrary()
    else:
        raise ValueError(f"Unknown feature library type: {library_type}")

def create_optimizer(optimizer_type: str, threshold: float = 0.1, alpha: float = 0.0, 
                    max_iter: int = 20, normalize_columns: bool = False, lasso_alpha: float = 0.01):
    """
    Create a sparse regression optimizer for SINDy model training.
    
    The optimizer determines which terms are included in the discovered equations
    by enforcing sparsity (keeping only the most important relationships).
    
    Parameters:
        optimizer_type: Type of optimizer ('stlsq' currently supported)
        threshold: Sparsity threshold - smaller values remove more terms
        alpha: Regularization parameter for numerical stability
        max_iter: Maximum iterations for iterative algorithms
        normalize_columns: Whether to normalize feature matrix columns
        lasso_alpha: Alpha parameter for Lasso-based optimizers
        
    Returns:
        PySINDy optimizer object
        
    Raises:
        ValueError: If unknown optimizer_type is specified
        
    Note:
        Additional optimizers (SR3, FROLS, etc.) are planned for future implementation.
    """
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
    """Normalize data using specified method."""
    if method == "minmax":
        X_scaler = MinMaxScaler()
        U_scaler = MinMaxScaler()
    elif method == "standard":
        X_scaler = StandardScaler()
        U_scaler = StandardScaler()
    elif method == "robust":
        X_scaler = RobustScaler()
        U_scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_normalized = X_scaler.fit_transform(X)
    U_normalized = U_scaler.fit_transform(U)
    
    return X_normalized, U_normalized, X_scaler, U_scaler

def downsample_data(sensors_data: np.ndarray, actuators_data: np.ndarray, configuration_data: np.ndarray, sampling_rate: int = 1) -> tuple:
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

def main():
    """
    Main function for dynamic building modeling using Sparse Identification of Nonlinear Dynamics.
    
    This function implements the complete pipeline for discovering mathematical equations
    that govern AAU BUILD's thermodynamic behavior from real sensor and actuator data.
    
    Workflow:
        1. Load sensor, actuator, and configuration data from CSV files
        2. Interpolate missing values using specified method
        3. Prepare state variables (X) and control inputs (U) 
        4. Enforce stability-driven hyperparameters (normalization, optimizer)
        5. Optionally normalize data for numerical stability
        6. Create SINDy model with specified feature library and optimizer
        7. Train model on full dataset and display discovered equations
        8. Validate model using train/test split (retrain on train data)
        9. Call high-speed Numba/SciPy simulation for validation and plotting
        
    The discovered equations represent the building as a controlled dynamical system:
        dX/dt = f(X, U)
    where X are sensor measurements and U are actuator commands.
    
    Command-line arguments control all aspects of the modeling process,
    allowing experimentation with different hyperparameters and configurations.
    
    Raises:
        Does not raise simulation exceptions directly, handled in custom module
        
    Example:
        ```bash
        python dynamic_model_smart_building.py \
            --polynomial-degree 1 \
            --threshold 0.0 \
            --normalize-data \
            --sampling-rate 10
        ```
    """
    args = parse_args()

    # Record start time for job duration
    import time
    start_time = time.time()
    
    # Print comprehensive parameter summary for cluster data collection
    print("="*80)
    print("PYSINDY BUILDING DYNAMICS MODELING - JOB CONFIGURATION")
    print("="*80)
    print(f"Data Files:")
    print(f"  Sensors:       {args.sensors}")
    print(f"  Actuators:     {args.actuators}")
    print(f"  Configuration: {args.configuration}")
    print(f"  CSV Separator: {args.sep}")
    print(f"")
    print(f"Data Processing:")
    print(f"  Interpolation Method:  {args.interpolation_method}")
    print(f"  Include Configuration: {args.include_configuration}")
    print(f"  Sampling Rate:         {args.sampling_rate} ({'no downsampling' if args.sampling_rate <= 1 else f'{args.sampling_rate}x speedup'})")
    print(f"  Normalize Data:        {args.normalize_data}")
    print(f"  Normalization Method:  {args.normalization_method}")
    print(f"  Time Step (dt):        {args.dt}")
    print(f"")
    print(f"SINDy Model Configuration:")
    print(f"  Feature Library:       {args.feature_library}")
    print(f"  Polynomial Degree:     {args.polynomial_degree}")
    print(f"  Fourier Frequencies:   {args.fourier_n_frequencies}")
    print(f"  Include Interactions:  {not args.no_interactions}")
    print(f"  Optimizer:             {args.optimizer}")
    print(f"  Sparsity Threshold:    {args.threshold}")
    print(f"  Regularization Alpha:  {args.alpha}")
    print(f"  Max Iterations:        {args.max_iter}")
    print(f"  Normalize Columns:     {args.normalize_columns}")
    print(f"  Lasso Alpha:           {args.lasso_alpha}")
    print(f"")
    print(f"Training/Validation:")
    print(f"  Train Split:           {args.train_split}")
    print(f"  Skip Validation:       {args.skip_validation}")
    print(f"  Skip Visualization:    {args.skip_visualization}")
    print(f"")
    print(f"System:")
    print(f"  Monitor Interval:      {args.monitor_interval}s")
    print(f"  Output Directory:      {args.outdir}")
    print("="*80)
    print("")

    # Start system monitoring
    start_monitoring(args.monitor_interval)
    
    # Load data
    print("Loading data...")
    sensors_data = get_csv_data(args.sensors, args.sep)
    actuators_data = get_csv_data(args.actuators, args.sep)
    configuration_data = get_csv_data(args.configuration, args.sep)
    
    print(f"Loaded:")
    print(f"  Sensors: {sensors_data.shape}")
    print(f"  Actuators: {actuators_data.shape}")
    print(f"  Configuration: {configuration_data.shape}")
    
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
    else:
        X = sensors_data
        print(f"\nUsing sensors only as state variables")
    
    # Use actuators as control inputs (U)
    U = actuators_data
    
    # --- STABILITY FIXES (from scipy_numba.py) ---
    if args.feature_library == "polynomial" and args.polynomial_degree > 1:
        print(f"\n--- WARNING: Using Polynomial Degree {args.polynomial_degree}. ---")
        print("--- The Numba simulation function is hard-coded for degree 1 ---")

    if not args.normalize_data:
        print("\n--- WARNING: Forcing data normalization (minmax) for numerical stability. ---")
        args.normalize_data = True
        args.normalization_method = "minmax"

    if args.optimizer != "stlsq":
        print(f"\n--- WARNING: Forcing optimizer to 'stlsq'. ---")
        args.optimizer = "stlsq"

    print(f"\n--- NOTE: Forcing STLSQ(threshold=0.0, alpha=10.0) for stability. ---")
    args.threshold = 0.1
    args.alpha = 75.0
    # ----------------------------------------------------------------------

    # Create time vector
    t = np.arange(len(X)) * args.dt
    
    print(f"\nFinal data prepared for SINDy:")
    print(f"  States X: {X.shape}")
    print(f"  Controls U: {U.shape}")
    print(f"  Time points: {len(t)}")
    print(f"  dt: {args.dt}")
    
    # Optionally normalize data
    if args.normalize_data:
        print(f"\nNormalizing data using {args.normalization_method} method...")
        X, U, X_scaler, U_scaler = normalize_data(X, U, args.normalization_method)
    
    # Checks for residual numerical issues
    if np.isnan(X).any() or np.isinf(X).any() or np.isnan(U).any() or np.isinf(U).any():
        print("\n!!! CRITICAL WARNING: NaN or Inf values detected after normalization. Check input data. !!!")

    # Create feature library and optimizer
    feature_library = create_feature_library(args.feature_library, args.polynomial_degree, args.fourier_n_frequencies, 
                                            include_interactions=not args.no_interactions)
    optimizer = create_optimizer(args.optimizer, args.threshold, args.alpha, args.max_iter, 
                                args.normalize_columns, args.lasso_alpha)
    
    # Build SINDy model
    print(f"\nTraining SINDy model...")
    print(f"  Feature library: {args.feature_library} (degree={args.polynomial_degree})")
    print(f"  Optimizer: {args.optimizer} (threshold={args.threshold}, alpha={args.alpha})")
    
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    # Train the model
    model.fit(X, u=U, t=args.dt)
    
    # Check model stability
    coeffs = model.coefficients()
    max_coeff = np.abs(coeffs).max()
    print(f"\nModel stability check:")
    print(f"  Max coefficient magnitude: {max_coeff:.3f}")
    print(f"  Coefficient threshold: {args.coefficient_threshold}")
    
    if max_coeff > args.coefficient_threshold:
        print(f"  WARNING: Large coefficients detected! Model may be unstable.")
        print(f"  Suggestions:")
        print(f"    1. Increase sparsity threshold: --threshold {args.threshold * 2}")
        print(f"    2. Add regularization: --alpha {max(0.1, args.alpha * 2)}")
        print(f"    3. Force normalization: --force-normalization")
        print(f"    4. Reduce polynomial degree: --polynomial-degree {max(1, args.polynomial_degree - 1)}")
    
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
    print("Retraining model on training data...")
    model.fit(X_train, u=U_train, t=args.dt)
    
    # --- CALL CUSTOM SIMULATION MODULE ---
    # The original simulation block is replaced with this call.
    print("\nPassing to Numba-accelerated simulation module...")
    custom_numba_simulation.run_numba_validation(
        model=model,
        X_test=X_test,
        U_test=U_test,
        t_test=t_test,
        dt=args.dt,
        output_filename="validation_plot.png"
    )
    # -------------------------------------

if __name__ == "__main__":
    main()