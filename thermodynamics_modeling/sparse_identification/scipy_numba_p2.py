"""
Dynamic Modeling of Smart Building Systems using Sparse Identification of Nonlinear Dynamics.

This script discovers ordinary differential equations (ODEs) from building sensor
data using the PySINDy library. It includes a high-performance, Numba-accelerated
validation loop that uses SciPy's ODE solvers for robust simulation.

"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import time 

# --- IMPORTS FOR OPTIMIZATION ---
from numba import njit, types
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pysindy.optimizers import STLSQ  # <-- Import STLSQ explicitly
# --------------------------------

# --- GLOBAL NUMBA FUNCTIONS ---
# Define explicit Numba types for JIT-compiled functions.

# 'A' layout (Any) - for the 'x' state array from solve_ivp
float_array_1d_a = types.float64[:] 

# 'C' layout (Contiguous) - for all 'safe' arrays
float_array_1d_c = types.float64[::1]
float_array_2d_c = types.float64[:, ::1]

# ---
# Signature for interpolation function (All 'C' arrays)
u_interp_sig = float_array_1d_c(
    types.float64, 
    float_array_1d_c, 
    float_array_2d_c
)

@njit(u_interp_sig, cache=True)
def u_interp_numba(t: float, t_interp: np.ndarray, U_interp: np.ndarray) -> np.ndarray:
    """
    JIT-compiled 1D linear interpolation for control inputs U(t).
    """
    # Find the index for interpolation
    idx = np.searchsorted(t_interp, t, side='left')

    # Handle boundary conditions
    if idx == 0:
        return U_interp[0]
    if idx >= len(U_interp):
        return U_interp[-1]

    # Get the two points to interpolate between
    t1, t2 = t_interp[idx - 1], t_interp[idx]
    u1, u2 = U_interp[idx - 1], U_interp[idx]

    # Calculate interpolation factor
    factor = (t - t1) / (t2 - t1) if (t2 - t1) != 0 else 0.0

    # Interpolate each control variable
    n_ctrl = U_interp.shape[1]
    u_out = np.empty(n_ctrl, dtype=np.float64)

    for i in range(n_ctrl):
        u_out[i] = u1[i] + (u2[i] - u1[i]) * factor

    return u_out

# ---
#  ODE RHS signature to return a C-layout array
#  accept a flexible A-layout array for 'x'.
ode_rhs_sig = float_array_1d_c(  # Returns 1d_C
    types.float64, 
    float_array_1d_a,   # Accepts 1d_A for x
    float_array_2d_c,   # Accepts 1d_C for coeffs
    float_array_1d_c,   # Accepts 1d_C for t_interp
    float_array_2d_c    # Accepts 2d_C for U_interp
)


@njit(ode_rhs_sig, cache=True)
def _ode_rhs_numba_jit(t: float, x: np.ndarray, coeffs: np.ndarray,
                       t_interp: np.ndarray, U_interp: np.ndarray) -> np.ndarray:
    """
    JIT-compiled core logic for the SINDy model's Right-Hand Side (RHS).
    
    [MODIFIED]: This version correctly builds a Polynomial(degree=2)
    feature vector to match the training.
    """
    
    # 1. Get the control input U(t) by interpolating
    u = u_interp_numba(t, t_interp, U_interp) # JIT -> JIT call
    
    n_states = len(x)
    n_ctrl = len(u)
    n_vars = n_states + n_ctrl
    
    # 2. Create the combined [x, u] array
    z = np.empty(n_vars, dtype=np.float64)
    for i in range(n_states):
        z[i] = x[i]
    for i in range(n_ctrl):
        z[n_states + i] = u[i]

    # 3. Calculate the number of features for Polynomial(degree=2)
    n_features = 1 + n_vars + (n_vars * (n_vars + 1) // 2)
    
    if coeffs.shape[1] != n_features:
        return np.full(n_states, np.nan) 
            
    feature_vector = np.empty(n_features, dtype=np.float64)
    
    # 4. Manually build the Polynomial(degree=2) feature vector
    idx = 0
    
    # Add bias term (1)
    feature_vector[idx] = 1.0
    idx += 1
    
    # Add degree 1 terms (z[0], z[1], ...)
    for i in range(n_vars):
        feature_vector[idx] = z[i]
        idx += 1
        
    # Add degree 2 terms (z[0]*z[0], z[0]*z[1], z[1]*z[1], ...)
    for i in range(n_vars):
        for j in range(i, n_vars):
            feature_vector[idx] = z[i] * z[j]
            idx += 1
            
    # 5. Calculate the derivative: dX/dt = Xi * Theta
    dXdt = np.dot(coeffs, feature_vector)
    
    return dXdt


def ode_rhs_numba_wrapper(t: float, x: np.ndarray, coeffs: np.ndarray,
                          t_interp: np.ndarray, U_interp: np.ndarray) -> np.ndarray:
    """
    Python wrapper function for the JIT-compiled ODE RHS.
    """
    return _ode_rhs_numba_jit(t, x, coeffs, t_interp, U_interp)



def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the SINDy modeling script.
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
    p.add_argument("--dt", type=float, default=0.1,
                   help="Time step between measurements")

    # SINDy model hyperparameters
    p.add_argument("--polynomial-degree", type=int, default=2,
                   help="Degree of polynomial features for SINDy")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Sparsity threshold for STLSQ optimizer")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Regularization parameter for STLSQ or Ridge optimizer")
    p.add_argument("--max-iter", type=int, default=20,
                   help="Maximum iterations for STLSQ optimizer")
    p.add_argument("--normalize-columns", action="store_true",
                   help="Normalize feature matrix columns")

    # Training/validation hyperparameters
    p.add_argument("--train-split", type=float, default=0.7,
                   help="Fraction of data to use for training (rest for validation)")
    
    p.add_argument("--fast-test", action="store_true",
                   help="Run a very short train AND simulation (1000/100 steps) to quickly test for stability.")

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

def get_csv_data(file_path: str, delimiter: str = ',', drop_timestamps: bool = True) -> np.ndarray:
    """
    Robustly load a CSV file into a NumPy array, forcing float64 dtype.
    """
    df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8-sig', engine='python')

    if drop_timestamps:
        df = df.iloc[:, 1:]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    data = df.to_numpy()
    return data


def process_data_simple_interpolation(sensors_data: np.ndarray, actuators_data: np.ndarray,
                                    configuration_data: np.ndarray, interpolation_method: str = "linear") -> tuple:
    """
    Interpolate missing values (NaNs) in the data arrays.
    """
    sensors_df = pd.DataFrame(sensors_data)
    actuators_df = pd.DataFrame(actuators_data)
    configuration_df = pd.DataFrame(configuration_data)

    total_nans = sensors_df.isna().sum().sum() + actuators_df.isna().sum().sum() + configuration_df.isna().sum().sum()
    print(f"Total missing values (including coerced): {total_nans}")

    sensors_clean = sensors_df.interpolate(method=interpolation_method).bfill().ffill().values  # type: ignore
    actuators_clean = actuators_df.interpolate(method=interpolation_method).bfill().ffill().values  # type: ignore
    configuration_clean = configuration_df.interpolate(method=interpolation_method).bfill().ffill().values  # type: ignore

    remaining_nans = np.isnan(sensors_clean).sum() + np.isnan(actuators_clean).sum() + np.isnan(configuration_clean).sum()
    print(f"Remaining NaN values after interpolation: {remaining_nans}")

    return sensors_clean, actuators_clean, configuration_clean


def create_feature_library(library_type: str, polynomial_degree: int = 2, fourier_n_frequencies: int = 2):
    """
    Created a PySINDy feature library instance.
    """
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
    """
    Created a PySINDy optimizer instance.
    """
    
    if optimizer_type == "stlsq":
        print(f"--- Using STLSQ Optimizer (normalize_columns={normalize_columns}) ---")
        return ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=max_iter, normalize_columns=normalize_columns)
    
    else:
        raise ValueError(f"Unknown or unsupported optimizer type: {optimizer_type}")


def normalize_data(X: np.ndarray, U: np.ndarray, method: str = "minmax"):
    """
    Normalize state (X) and control (U) data using scikit-learn.
    """
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


def main():
    """Main function for dynamic building modeling using Sparse Identification of Nonlinear Dynamics."""
    
    start_time = time.time()
    args = parse_args()

    # Load and process data 
    print("Loading data...")
    sensors_data = get_csv_data(args.sensors, args.sep)
    actuators_data = get_csv_data(args.actuators, args.sep)
    configuration_data = get_csv_data(args.configuration, args.sep)
    
    print(f"\nInterpolating missing values using {args.interpolation_method} method...")
    sensors_data, actuators_data, configuration_data = process_data_simple_interpolation(
        sensors_data, actuators_data, configuration_data, args.interpolation_method
    )
    
    if args.include_configuration:
        X = np.hstack([sensors_data, configuration_data])
    else:
        X = sensors_data
    
    U = actuators_data
    
    # --- CRITICAL STABILITY/MEMORY FIXES (Enforced) ---
    
    # 1. Set Polynomial Degree (User wants 2)
    print(f"\n--- Using Polynomial Degree {args.polynomial_degree} ---")
    
    # 2. Enforce Normalization
    if not args.normalize_data:
        print("\n--- WARNING: Forcing data normalization to solve numerical overflow. ---")
        args.normalize_data = True
        
    # 3. Revert to 'minmax' scaler
    if args.normalization_method != "minmax":
        print("\n--- FINAL STRATEGY: Reverting to 'minmax' normalization. ---")
        args.normalization_method = "minmax"

    # 4. Revert to STLSQ optimizer
    if args.optimizer != "stlsq":
        print("\n--- REVERTING: Forcing optimizer back to 'stlsq'. ---")
        args.optimizer = "stlsq"

    # 5. [SVD FIX]: Disable optimizer normalization
    if args.normalize_columns:
        print("\n--- APPLYING FIX: Forcing 'normalize_columns=False' in STLSQ to prevent SVD error. ---")
    args.normalize_columns = False 
        
    # 6. [STABILITY TUNING]: Set Alpha
    args.alpha = 50000.0
    
    print(f"\n--- Using STLSQ(threshold={args.threshold}, alpha={args.alpha}) ---")

    # Optionally normalize data
    if args.normalize_data:
        print(f"\nNormalizing data using {args.normalization_method} method...")
        X, U, X_scaler, U_scaler = normalize_data(X, U, args.normalization_method)
    
    if np.isnan(X).any() or np.isinf(X).any() or np.isnan(U).any() or np.isinf(U).any():
        print("\n!!! WARNING: NaN or Inf values detected after normalization. Check input data. !!!")
        
    # Create time vector
    t = np.arange(len(X)) * args.dt
    
    # Create feature library and optimizer
    feature_library = create_feature_library(args.feature_library, args.polynomial_degree, args.fourier_n_frequencies)
    optimizer = create_optimizer(args.optimizer, args.threshold, args.alpha, args.max_iter, 
                                 args.normalize_columns, args.lasso_alpha)
    
    # Build SINDy model object
    print(f"\nBuilding SINDy model object...")
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    
    # --- REMOVED: Redundant 4-hour training step ---
    # print(f"\nTraining SINDy model...")
    # model.fit(X, u=U, t=args.dt)
    # print("\nDiscovered equations:")
    # model.print()
    
    # Validation: split data
    split = int(args.train_split * len(X))
    X_train, X_test = X[:split], X[split:]
    U_train, U_test = U[:split], U[split:]
    t_test = np.arange(len(X_test)) * args.dt
    
    
    # -----------------------------------------------------------------
    # START: NEW FAST-TRAIN LOGIC
    # This block now uses --fast-test to shorten training time
    # -----------------------------------------------------------------
    if args.fast_test:
        train_len = min(1000, len(X_train)) # Use 1000 steps for fast training
        print(f"\n--- FAST-TEST: Training on a small subset of {train_len} training steps... ---")
        X_train_subset = X_train[:train_len]
        U_train_subset = U_train[:train_len]
        model.fit(X_train_subset, u=U_train_subset, t=args.dt)
    else:
        print(f"\n--- FULL-TRAIN: Retraining on {args.train_split*100}% training data ({len(X_train)} steps)... ---")
        model.fit(X_train, u=U_train, t=args.dt) # <-- This is the original (slow) call
    # -----------------------------------------------------------------
    # END: NEW FAST-TRAIN LOGIC
    # -----------------------------------------------------------------
    
    # Print equations *after* the model is actually trained
    print("\nDiscovered equations:")
    model.print()

    # --- PRE-SIMULATION SANITY CHECK ---
    print("\n--- PRE-SIMULATION SANITY CHECK ---")
    try:
        coeffs_shape = model.optimizer.coef_.shape
        print(f"Model coefficients (Xi) shape: {coeffs_shape}")

        n_states = X.shape[1]
        n_ctrl = U.shape[1]
        n_vars = n_states + n_ctrl
        expected_features = 1 + n_vars + (n_vars * (n_vars + 1) // 2)
        
        print(f"Number of states: {n_states}, Number of controls: {n_ctrl}")
        print(f"Total variables (z = x + u): {n_vars}")
        print(f"Expected degree-2 features: {expected_features}")

        if coeffs_shape[1] == expected_features:
            print("\n SUCCESS: Coefficient matrix columns match expected features.")
        else:
            print("\n ERROR: Mismatch detected. DO NOT RUN SIMULATION.")
            raise ValueError("Pre-simulation check failed. Mismatch in feature dimensions.")

    except Exception as e:
        print(f"An error occurred during the pre-flight check: {e}")
        print("Stopping execution.")
        exit()
        
    print("--- END OF SANITY CHECK ---")

    
    # -----------------------------------------------------------------
    # FAST-TEST LOGIC (Simulation part)
    # -----------------------------------------------------------------
    X_truth = None # Will store the ground truth for RMSE
    
    if args.fast_test:
        print("\n--- FAST-TEST: Preparing for short simulation ---")
        # Run a simulation on just the first 100 steps
        test_len = min(100, len(X_test)) # Use 100 steps or less
        
        print(f"   Simulating for {test_len} time steps...")
        
        # Create a *short* test set
        X_truth = X_test[:test_len] # Ground truth for RMSE
        U_test_short = U_test[:test_len]
        t_test_short = t_test[:test_len]
        
        # Set the time vectors and initial conditions for Numba
        t_interp_safe = np.ascontiguousarray(t_test_short, dtype=np.float64)
        U_interp_safe = np.ascontiguousarray(U_test_short, dtype=np.float64)
        y0_safe = np.ascontiguousarray(X_truth[0], dtype=np.float64)
        
    else:
        print("\n--- RUNNING FULL VALIDATION SIMULATION ---")
        # Use the full test set (original behavior)
        X_truth = X_test # Ground truth for RMSE
        t_interp_safe = np.ascontiguousarray(t_test, dtype=np.float64)
        U_interp_safe = np.ascontiguousarray(U_test, dtype=np.float64)
        y0_safe = np.ascontiguousarray(X_truth[0], dtype=np.float64)
    # -----------------------------------------------------------------
    # FAST-TEST LOGIC
    # -----------------------------------------------------------------


    # --- PREPARE DATA FOR NUMBA ---
    print("\nCreating Numba-safe contiguous float64 arrays...")

    # 1. Clean the coefficients array (FIX: use .optimizer.coef_)
    coeffs_safe = np.ascontiguousarray(model.optimizer.coef_, dtype=np.float64)
    
    # 2, 3, 4 (t, U, y0) are now defined in the logic block above

    # 5. Create the args tuple with the safe arrays
    args_tuple = (coeffs_safe, t_interp_safe, U_interp_safe)
    
    
    # --- [DEBUGGING BLOCK] ---
    print("\n--- DEBUGGING: Checking arrays before simulation ---")
    
    if np.isnan(coeffs_safe).any() or np.isinf(coeffs_safe).any():
        print("!!! FATAL ERROR: 'coeffs_safe' (the model) contains NaN or Inf. !!!")
        print("This means the SINDy optimizer (model.fit) is failing.")
        return # Exit the main function
    else:
        print("  coeffs_safe (model): OK")

    if np.isnan(y0_safe).any() or np.isinf(y0_safe).any():
        print("!!! FATAL ERROR: 'y0_safe' (the initial state) contains NaN or Inf. !!!")
        return # Exit the main function
    else:
        print("  y0_safe (initial state): OK")
        
    if np.isnan(t_interp_safe).any() or np.isinf(t_interp_safe).any():
        print("!!! FATAL ERROR: 't_interp_safe' (time) contains NaN or Inf. !!!")
        return # Exit the main function
    else:
        print("  t_interp_safe (time): OK")
        
    if np.isnan(U_interp_safe).any() or np.isinf(U_interp_safe).any():
        print("!!! FATAL ERROR: 'U_interp_safe' (control inputs) contains NaN or Inf. !!!")
        return # Exit the main function
    else:
        print("  U_interp_safe (control): OK")

    print("--- DEBUGGING: All inputs are valid. Proceeding to simulation. ---")

    
    # Predict on test data using the Numba-compiled RHS function
    # Predict on test data using the Numba-compiled RHS function
    try:
        print("\nStarting Numba-accelerated simulation via solve_ivp...")
        
        sol = solve_ivp(
            fun=ode_rhs_numba_wrapper,
            t_span=(t_interp_safe[0], t_interp_safe[-1]),
            y0=y0_safe,
            t_eval=t_interp_safe,
            args=args_tuple,
            method='Radau', 
            max_step=args.dt,
            rtol=1e-6,
            atol=1e-9
        )
        
        X_pred = sol.y.T
        
        # Calculate error and plot
        # X_truth was defined in the --fast-test logic block
        
        
        min_len = min(len(X_truth), len(X_pred)) 
        # -----------------------------------------------------------------
        
        rmse = np.sqrt(np.mean((X_truth[:min_len] - X_pred[:min_len])**2))
        
        if np.isnan(rmse):
            print(f"\nValidation failed: RMSE is nan. Model is unstable. {'(FAST-TEST)' if args.fast_test else ''}")
        else:
            print(f"\nValidation RMSE (Numba): {rmse:.6f} {'(FAST-TEST)' if args.fast_test else ''}")
        
    except Exception as e:
        print(f"\nNumba Simulation failed: {e}")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n--- Total execution time: {elapsed_time:.2f} seconds ---")

if __name__ == "__main__":
    main()