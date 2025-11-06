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
    - Simple data downsampling for faster training
    - Temporal validation with predictive simulation
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
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import psutil


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

def log_system_usage():
    """Log current CPU and RAM usage."""
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    print(f"[MONITOR] CPU: {cpu:.1f}% | RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

def start_monitoring(interval_seconds: int):
    """Start background monitoring thread."""
    if interval_seconds <= 0:
        return None
    
    def monitor():
        while True:
            time.sleep(interval_seconds)
            log_system_usage()
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread

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
        3. Downsample data for performance optimization (if requested)
        4. Combine sensor/configuration data into state variables (X)
        5. Normalize data for numerical stability (if requested)
        6. Create SINDy model with specified feature library and optimizer
        7. Train model on preprocessed dataset and display discovered equations
        8. Validate model using train/test split and temporal simulation
        9. Calculate prediction errors and generate comparison plots
        
    The discovered equations represent the building as a controlled dynamical system:
        dX/dt = f(X, U)
    where X are sensor measurements and U are actuator commands.
    
    Command-line arguments control all aspects of the modeling process,
    allowing experimentation with different hyperparameters and configurations.
    
    Raises:
        Exception: If simulation fails during validation phase
        
    Example:
        ```bash
        python dynamic_model_smart_building.py \
            --polynomial-degree 3 \
            --threshold 0.05 \
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
    else:
        X = sensors_data
        print(f"\nUsing sensors only as state variables")
    
    # Use actuators as control inputs (U)
    U = actuators_data
    
    # Optionally normalize data
    if args.normalize_data:
        print(f"Normalizing data using {args.normalization_method} method...")
        X, U, X_scaler, U_scaler = normalize_data(X, U, args.normalization_method)
    
    # Create time vector
    t = np.arange(len(X)) * args.dt
    
    print(f"\nFinal data prepared for SINDy:")
    print(f"  States X: {X.shape}")
    print(f"  Controls U: {U.shape}")
    print(f"  Time points: {len(t)}")
    print(f"  dt: {args.dt}")
    
    # Create feature library and optimizer
    feature_library = create_feature_library(args.feature_library, args.polynomial_degree, args.fourier_n_frequencies, 
                                            include_interactions=not args.no_interactions)
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
    
    # Optional validation step
    if not args.skip_validation:
        validation_start = time.time()
        
        # Simple validation: split data and test prediction
        split = int(args.train_split * len(X))
        X_train, X_test = X[:split], X[split:]
        U_train, U_test = U[:split], U[split:]
        t_test = np.arange(len(X_test)) * args.dt
        
        print(f"\nValidation with {args.train_split:.1%} train / {1-args.train_split:.1%} test split...")
        print(f"  Training set: {len(X_train):,} timesteps")
        print(f"  Test set: {len(X_test):,} timesteps")
        
        # Optionally subsample validation data for faster simulation
        if args.validation_subsample > 1:
            val_indices = np.arange(0, len(X_test), args.validation_subsample)
            X_test_sub = X_test[val_indices]
            U_test_sub = U_test[val_indices]
            t_test_sub = np.arange(len(X_test_sub)) * args.dt
            print(f"  Validation subsampling: every {args.validation_subsample} samples ({len(X_test_sub):,} timesteps)")
            X_test, U_test, t_test = X_test_sub, U_test_sub, t_test_sub
        
        # Retrain on training data only
        retrain_start = time.time()
        model.fit(X_train, u=U_train, t=args.dt)
        retrain_time = time.time() - retrain_start
        print(f"  Retraining time: {retrain_time:.2f}s")
        
        # Predict on test data
        try:
            simulation_start = time.time()
            
            # Debug: Check model stability before simulation
            print(f"  Debugging model before simulation...")
            print(f"    Model coefficients shape: {model.coefficients().shape}")
            print(f"    Model coefficient sparsity: {(model.coefficients() == 0).sum()}/{model.coefficients().size}")
            print(f"    Max coefficient magnitude: {np.abs(model.coefficients()).max():.6f}")
            
            # Debug: Check initial conditions
            print(f"    Initial state X_test[0]: {X_test[0][:5]}")  # Show first 5 values
            print(f"    Initial state range: [{X_test[0].min():.3f}, {X_test[0].max():.3f}]")
            
            # Try shorter simulation first to check stability
            short_t = t_test[:min(100, len(t_test))]  # Only first 100 timesteps
            short_U = U_test[:len(short_t)]
            
            print(f"    Testing short simulation ({len(short_t)} steps)...")
            try:
                X_pred_short = model.simulate(X_test[0], short_t, u=short_U)
                print(f"    Short simulation successful!")
                print(f"    Short prediction range: [{X_pred_short.min():.3f}, {X_pred_short.max():.3f}]")
                
                # Check for numerical issues
                if np.any(np.isnan(X_pred_short)) or np.any(np.isinf(X_pred_short)):
                    print(f"    WARNING: NaN/Inf detected in short simulation!")
                    raise ValueError("Numerical instability in simulation")
                
                if np.abs(X_pred_short).max() > 1e6:
                    print(f"    WARNING: Very large values in simulation (max: {np.abs(X_pred_short).max():.2e})")
                    raise ValueError("Simulation appears unstable (explosive growth)")
                
            except Exception as e:
                print(f"    Short simulation failed: {e}")
                raise e
            
            # If short simulation works, try full simulation with timeout
            print(f"    Running full simulation ({len(t_test)} steps)...")
            
            # Set maximum simulation time (safety timeout)
            max_sim_time = args.simulation_timeout
            
            def simulate_with_timeout():
                return model.simulate(X_test[0], t_test, u=U_test)
            
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Simulation timed out after {max_sim_time}s")
            
            # Set timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_sim_time)
            
            try:
                X_pred = simulate_with_timeout()
                signal.alarm(0)  # Cancel alarm
            except TimeoutError as e:
                signal.alarm(0)  # Cancel alarm
                print(f"    {e}")
                print(f"    Try using --validation-subsample for faster validation")
                raise e
            
            simulation_time = time.time() - simulation_start
            print(f"  Simulation time: {simulation_time:.2f}s")
            
            # Calculate error
            min_len = min(len(X_test), len(X_pred))
            # TODO: are there more interesting error metrics?
            rmse = np.sqrt(np.mean((X_test[:min_len] - X_pred[:min_len])**2))
            
            print(f"\nValidation RMSE: {rmse:.6f}")
            
            validation_total = time.time() - validation_start
            print(f"Total validation time: {validation_total:.2f}s")
            
            # Optional visualization
            if not args.skip_visualization:
                # Configure matplotlib for headless operation (cluster-friendly)
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                # Simple plot (fixed visualization parameters)
                plt.figure(figsize=(12, 8))
                n_states = min(5, X.shape[1])  # Plot up to 5 states
                
                for i in range(n_states):
                    plt.subplot(n_states, 1, i+1)
                    plt.plot(X_test[:min_len, i], 'k-', label='True', linewidth=1.5)
                    plt.plot(X_pred[:min_len, i], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
                    plt.ylabel(f'State {i+1}')
                    plt.legend()
                    
                    # Calculate per-state error
                    state_rmse = np.sqrt(np.mean((X_test[:min_len, i] - X_pred[:min_len, i])**2))
                    plt.title(f'State {i+1} - RMSE: {state_rmse:.4f}')
                    
                plt.xlabel('Time steps')
                plt.suptitle(f'Model Validation - Overall RMSE: {rmse:.4f}')
                plt.tight_layout()
                
                # Save plot instead of showing (cluster-friendly)
                plot_filename = f"{args.outdir}/validation_plot_sampling{args.sampling_rate}_degree{args.polynomial_degree}.png"
                import os
                os.makedirs(args.outdir, exist_ok=True)
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()  # Important: close to free memory
                print(f"Validation plot saved to: {plot_filename}")
                
                # Additional error analysis plot
                plt.figure(figsize=(10, 6))
                
                # Plot prediction errors over time
                error = X_test[:min_len] - X_pred[:min_len]
                plt.subplot(2, 1, 1)
                for i in range(min(3, X.shape[1])):
                    plt.plot(error[:, i], label=f'State {i+1} Error', alpha=0.7)
                plt.ylabel('Prediction Error')
                plt.legend()
                plt.title('Prediction Errors Over Time')
                
                # Plot error distribution
                plt.subplot(2, 1, 2)
                plt.hist(error.flatten(), bins=50, alpha=0.7, density=True)
                plt.xlabel('Prediction Error')
                plt.ylabel('Density')
                plt.title('Error Distribution')
                
                plt.tight_layout()
                error_filename = f"{args.outdir}/error_analysis_sampling{args.sampling_rate}_degree{args.polynomial_degree}.png"
                plt.savefig(error_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Error analysis plot saved to: {error_filename}")
                
        except Exception as e:
            print(f"Simulation failed: {e}")
    
    # Calculate and print job duration
    end_time = time.time()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60
    
    # Print parseable summary for cluster data collection
    print("")
    print("CLUSTER_DATA_SUMMARY_START")
    print(f"SAMPLING_RATE={args.sampling_rate}")
    print(f"POLYNOMIAL_DEGREE={args.polynomial_degree}")
    print(f"THRESHOLD={args.threshold}")
    print(f"NORMALIZE_DATA={args.normalize_data}")
    print(f"FEATURE_LIBRARY={args.feature_library}")
    print(f"OPTIMIZER={args.optimizer}")
    print(f"INTERPOLATION_METHOD={args.interpolation_method}")
    print(f"TOTAL_DURATION_SECONDS={total_duration:.2f}")
    print(f"SPEEDUP_FACTOR={args.sampling_rate}")
    print(f"SKIP_VALIDATION={args.skip_validation}")
    print("CLUSTER_DATA_SUMMARY_END")
    
    print("")
    print("="*80)
    print("JOB COMPLETED - TIMING SUMMARY")
    print("="*80)
    print(f"Total Duration: {hours:02d}h {minutes:02d}m {seconds:05.2f}s ({total_duration:.2f}s total)")
    print(f"Start Time:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End Time:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print("="*80)

if __name__ == "__main__":
    main()