import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re

DT = 0.5
ODE_FILE = "../../thermodynamics_modeling/sparse_identification/v0.2/room_ode.txt"



def load_data(sensor_path: str, actuator_path: str):
    """
    Load and preprocess sensor and actuator CSV data.

    Reads sensor and actuator CSV files, interpolates missing values, and returns:
    - time vector `t`
    - sensor measurements `x_measured`
    - actuator commands `U_df`
    - actuator column names `actuator_cols`

    Parameters
    sensor_path : Path to the sensor CSV file.
    actuator_path : Path to the actuator CSV file.

    Returns
    t : np.ndarray
        Time steps array.
    x_measured : np.ndarray
        Interpolated sensor measurements.
    U_df : pd.DataFrame
        Interpolated actuator data.
    actuator_cols : list of str
        List of actuator column names.
    """
    df_sensors = pd.read_csv(sensor_path)
    df_actuators = pd.read_csv(actuator_path)

    # Sensor column
    room_temp_col = [col for col in df_sensors.columns if col != "timestamp"][0]
    # Automate the actuator_cols extraction
    actuator_cols = [col for col in df_actuators.columns if col != "timestamp"]

    x_measured = df_sensors[room_temp_col].interpolate().values
    U_df = df_actuators[actuator_cols].interpolate().fillna(method="bfill").fillna(method="ffill")

    t = np.arange(len(df_sensors)) * DT
    return t, x_measured, U_df, actuator_cols


def parse_ode(ode_file: str, actuator_cols):
    """
    Parse an ODE file and return a symbolic and numerical RHS function.

    Reads a text file containing an ODE of the form `x' = ...`, validates that
    the actuator variables `u0, u1, ...` match the provided actuator columns,
    and returns a fast NumPy-compatible function for evaluation.

    Parameters
    ode_file : Path to the text file containing the ODE.
    actuator_cols : List of actuator column names from the CSV data.

    Returns
    -------
    rhs_func : function
        A NumPy-compatible function `rhs_func(x, u_vals)` representing the ODE right-hand side.
        Expects `u_vals` as an array of length equal to `len(actuator_cols)`.

    """
    with open(ode_file, "r") as f:
        ode_str = f.read().strip()

    # right-hand side of the ODE
    rhs = ode_str.split("=")[1].strip()

    # Validate if the data mathes ODE 
    matches = re.findall(r"u(\d+)\b", rhs)
    max_index = -1
    if matches:
        max_index = max([int(m) for m in matches])
    
    # Number of controls (u's)
    num_symbols_needed = max_index + 1
    # Number of actual actuators
    num_actuators_provided = len(actuator_cols)

    # Check for missmatch
    if num_symbols_needed > num_actuators_provided:
        raise ValueError(
            f"\nYour ODE file ('{ode_file}') requires actuators "
            f"up to u{max_index} (which requires {num_symbols_needed} actuator columns)."
            f"\nYour actuator CSV file only provided {num_actuators_provided} columns."
        )
    

    # Define control inputs
    x = sp.symbols("x0")
    u_symbols = sp.symbols(f"u0:{num_actuators_provided}")

    # Replace u-variables with indexed symbols 
    rhs = re.sub(r"u(\d+)\b", r"u_symbols[\1]", rhs)

    # Parse into SymPy expression
    local_dict = {"x0": x, "u_symbols": u_symbols}
    expr = sp.sympify(rhs, locals=local_dict)

    # Create fast NumPy function
    # This function will correctly expect a u_vals array of length num_actuators_provided
    rhs_func = sp.lambdify((x, u_symbols), expr, "numpy")

    print("✅ ODE parsed successfully!")
    return rhs_func

def fast_euler_simulation(rhs_func, x0, U_df, DT, t):
    """Run a fast Euler integration given RHS function and actuator data."""
    x_sim = np.zeros(len(t))
    x_sim[0] = x0

    for i in range(len(t) - 1):
        u_vals = U_df.iloc[i].values
        val = rhs_func(x_sim[i], u_vals)
        if isinstance(val, (np.ndarray, list)):
            val = float(val)
        x_sim[i + 1] = x_sim[i] + DT * val

    return x_sim

def compute_rmse(x_sim, x_measured):
    """Compute RMSE between simulated and measured data."""
    valid = np.isfinite(x_sim)
    if np.any(valid):
        return np.sqrt(np.nanmean((x_sim[valid] - x_measured[valid]) ** 2))
    return np.nan


def plot_results(t, x_measured, x_sim):
    """Plot measured vs simulated room temperature."""
    plt.figure(figsize=(10, 5))
    plt.plot(t / 3600, x_measured, label="Measured", lw=2)
    plt.plot(t / 3600, x_sim, "--", label="Simulated (Euler fast)")
    plt.xlabel("Timeframe")
    plt.ylabel("Temperature [°C]")
    plt.title("Room Temperature – Fast ODE Validation")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
     # Load data and get actuator columns dynamically
    t, x_measured, U_df, actuator_cols = load_data(
        "../../data_fragmentation/out/R_A_All_May/rooma_5_1_2023_09_05_5_30_2023_22_20/data_sensors.csv",
        "../../data_fragmentation/out/R_A_All_May/rooma_5_1_2023_09_05_5_30_2023_22_20/data_actuators.csv"
    )
    print(actuator_cols)
    # Use the dynamically generated actuator columns
    rhs_func = parse_ode(ODE_FILE, actuator_cols)
    print(rhs_func)

    # Run simulation
    x_sim = fast_euler_simulation(rhs_func, x_measured[0], U_df, DT, t)

    # Compute RMSE
    rmse = compute_rmse(x_sim, x_measured)
    print(f"✅ Validation RMSE: {rmse:.4f}")

    # Plot results
    plot_results(t, x_measured, x_sim)


if __name__ == "__main__":
    main()
