"""
Custom Numba-Accelerated Simulation Module for PySINDy Models.

This module provides a high-performance validation function for SINDy models
discovered by the main 'dynamic_model_smart_building.py' script.

It uses Numba to JIT-compile the right-hand side (RHS) of the ODE
and SciPy's 'solve_ivp' for robust simulation.

Key Functions:
    - run_numba_validation: The main entry point for running a validation.
    - u_interp_numba: JIT-compiled interpolation for control inputs.
    - _ode_rhs_numba_jit: JIT-compiled ODE RHS function for SciPy.
    - ode_rhs_numba_wrapper: Python wrapper for the JIT-compiled RHS.

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import numpy as np
import matplotlib.pyplot as plt

# --- IMPORTS FOR OPTIMIZATION ---
from numba import njit, types
from scipy.integrate import solve_ivp
# --------------------------------

# ====================================================================
# --- GLOBAL NUMBA FUNCTIONS ---
# Define explicit Numba types for JIT-compiled functions.
# ====================================================================

# 1D C-contiguous float64 array
float_array_1d_c = types.float64[::1]
# 2D C-contiguous float64 array
float_array_2d_c = types.float64[:, ::1]

# Define the Numba signature for the interpolation function:
# (float, 1D_array, 2D_array) -> 1D_array
u_interp_sig = float_array_1d_c(
    types.float64,
    float_array_1d_c,
    float_array_2d_c
)


@njit(u_interp_sig, cache=True)
def u_interp_numba(t: float, t_interp: np.ndarray, U_interp: np.ndarray) -> np.ndarray:
    """
    JIT-compiled 1D linear interpolation for control inputs U(t).

    This function is called repeatedly by the ODE solver to find the
    value of the control input `U` at an arbitrary time `t`.
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


# Define the Numba signature for the ODE Right-Hand Side (RHS) function:
ode_rhs_sig = float_array_1d_c(
    types.float64,
    float_array_1d_c,
    float_array_2d_c,  # coeffs
    float_array_1d_c,  # t_interp
    float_array_2d_c   # U_interp
)


@njit(ode_rhs_sig, cache=True)
def _ode_rhs_numba_jit(t: float, x: np.ndarray, coeffs: np.ndarray,
                       t_interp: np.ndarray, U_interp: np.ndarray) -> np.ndarray:
    """
    JIT-compiled core logic for the SINDy model's Right-Hand Side (RHS).

    This function calculates the derivative dX/dt at time `t` and state `x`,
    using the discovered SINDy coefficients.
    
    NOTE: This implementation is hard-coded for a Polynomial(degree=1)
    or IdentityLibrary, as it manually builds the feature vector:
    [1, x0, x1, ..., u0, u1, ...]
    """
    # 1. Get the control input U(t) by interpolating
    u = u_interp_numba(t, t_interp, U_interp)

    # 2. Manually build the feature library Theta(X, U)
    n_states = len(x)
    n_ctrl = len(u)

    # Pre-allocate the feature vector for speed
    feature_vector = np.empty(1 + n_states + n_ctrl, dtype=np.float64)

    # Fill the vector
    feature_vector[0] = 1.0              # Bias term
    feature_vector[1: 1 + n_states] = x  # State variables
    feature_vector[1 + n_states:] = u    # Control variables

    # 3. Calculate the derivative: dX/dt = Xi * Theta
    dXdt = np.dot(coeffs, feature_vector)

    return dXdt


def ode_rhs_numba_wrapper(t: float, x: np.ndarray, coeffs: np.ndarray,
                          t_interp: np.ndarray, U_interp: np.ndarray) -> np.ndarray:
    """
    Python wrapper function for the JIT-compiled ODE RHS.

    This function acts as a "safe" bridge between the pure-Python
    `scipy.integrate.solve_ivp` function and the Numba-compiled
    `_ode_rhs_numba_jit` function.
    """
    return _ode_rhs_numba_jit(t, x, coeffs, t_interp, U_interp)


def run_numba_validation(model, X_test: np.ndarray, U_test: np.ndarray,
                         t_test: np.ndarray, dt: float,
                         output_filename: str = "validation_plot.png"):
    """
    Runs the high-speed Numba/SciPy simulation and validation.

    This function prepares Numba-compatible arrays, calls `solve_ivp`
    with the JIT-compiled RHS, calculates the RMSE, and saves a
    comparison plot.
    
    Parameters
    ----------
    model : ps.SINDy
        The *trained* PySINDy model.
    X_test : np.ndarray
        The validation state data (ground truth).
    U_test : np.ndarray
        The validation control input data.
    t_test : np.ndarray
        The time vector for the validation data.
    dt : float
        The time step, used as 'max_step' for the solver.
    output_filename : str
        The file path to save the resulting plot.
    """
    # --- PREPARES DATA FOR NUMBA ---
    print("\nCreating Numba-safe contiguous float64 arrays...")
    try:
        # 1. Gets coefficients from the trained model's optimizer
        coeffs_safe = np.ascontiguousarray(model.optimizer.coef_, dtype=np.float64)
        # 2. Cleans the time vector for interpolation
        t_interp_safe = np.ascontiguousarray(t_test, dtype=np.float64)
        # 3. Cleans the control input vector for interpolation
        U_interp_safe = np.ascontiguousarray(U_test, dtype=np.float64)
        # 4. Cleans the initial condition (y0), which is a slice
        y0_safe = np.ascontiguousarray(X_test[0], dtype=np.float64)
        # 5. Creates the args tuple with the safe arrays
        args_tuple = (coeffs_safe, t_interp_safe, U_interp_safe)
    except Exception as e:
        print(f"Error preparing Numba arrays: {e}")
        return

    # --- Predicts on test data using the Numba-compiled RHS function ---
    try:
        print("\nStarting Numba-accelerated simulation via solve_ivp...")

        # Uses solve_ivp directly with the Python wrapper and "safe" arrays
        sol = solve_ivp(
            fun=ode_rhs_numba_wrapper,       # Uses the Python wrapper
            t_span=(t_interp_safe[0], t_interp_safe[-1]),
            y0=y0_safe,                      # Uses the safe initial condition
            t_eval=t_interp_safe,            # Uses the safe time vector
            args=args_tuple,                 # Pass (coeffs, t, U) as args
            method='RK45',                   # Uses standard, stable solver
            max_step=dt,
        )

        # The solution is in sol.y (needs transpose)
        X_pred = sol.y.T

        # Calculate error and plot
        min_len = min(len(X_test), len(X_pred))
        rmse = np.sqrt(np.mean((X_test[:min_len] - X_pred[:min_len])**2))

        if np.isnan(rmse):
            print("\nValidation failed: RMSE is nan. Model is unstable.")
        else:
            print(f"\nValidation RMSE (Numba/SciPy): {rmse:.6f}")

            plt.figure(figsize=(12, 6))
            n_states_to_plot = min(3, X_test.shape[1])
            for i in range(n_states_to_plot):
                plt.subplot(n_states_to_plot, 1, i+1)
                plt.plot(X_test[:min_len, i], 'k-', label='True')
                plt.plot(X_pred[:min_len, i], 'r--', label='Predicted (Numba)')
                plt.ylabel(f'State {i+1}')
                plt.legend()
            plt.xlabel('Time steps')
            plt.tight_layout()

            # --- FIX: Save plot to file instead of showing ---
            plt.savefig(output_filename)
            print(f"\nSuccessfully saved plot to {output_filename}")

    except Exception as e:
        print(f"\nNumba Simulation failed: {e}")