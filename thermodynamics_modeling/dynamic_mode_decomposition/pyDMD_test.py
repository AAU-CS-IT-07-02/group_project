#!/usr/bin/env python3
"""
PyDMD Test Script for Building Data Analysis

This script performs Dynamic Mode Decomposition (DMD), Higher-Order DMD (HODMD),
and Multi-Resolution DMD (MrDMD) on building sensor, actuator, and configuration data.

Steps:
1. Load and clean data (remove timestamps, handle NaNs).

2. Interpolate missing values using pandas.

3. Normalize and reduce dimensionality using PCA.

4. Apply DMD variants and save results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pydmd import DMD, HODMD, MrDMD
import os

def process_data_simple_interpolation(sensors_data: np.ndarray, actuators_data: np.ndarray, 
                                      configuration_data: np.ndarray, interpolation_method: str = "linear") -> tuple:
    """
    Interpolate missing values in building sensor and actuator data.

    Parameters:
        sensors_data (np.ndarray): Sensor measurements.
        actuators_data (np.ndarray): Actuator states.
        configuration_data (np.ndarray): System configuration parameters.
        interpolation_method (str): Interpolation strategy ('linear', 'cubic', etc.)

    Returns:
        tuple: Interpolated arrays (sensors_clean, actuators_clean, configuration_clean)
    """
    sensors_df = pd.DataFrame(sensors_data)
    actuators_df = pd.DataFrame(actuators_data)
    configuration_df = pd.DataFrame(configuration_data)

    total_nans = sensors_df.isna().sum().sum() + actuators_df.isna().sum().sum() + configuration_df.isna().sum().sum()
    print(f"🔍 Total missing values before interpolation: {total_nans}")

    sensors_clean = sensors_df.interpolate(method=interpolation_method).bfill().ffill().values
    actuators_clean = actuators_df.interpolate(method=interpolation_method).bfill().ffill().values
    configuration_clean = configuration_df.interpolate(method=interpolation_method).bfill().ffill().values

    remaining_nans = np.isnan(sensors_clean).sum() + np.isnan(actuators_clean).sum() + np.isnan(configuration_clean).sum()
    print(f"[SUCCESS] Remaining NaNs after interpolation: {remaining_nans}")

    return sensors_clean, actuators_clean, configuration_clean

def load_csv(file: str, data_path: str) -> np.ndarray:
    """
    Load CSV file and remove timestamp column.

    Parameters:
        file (str): Filename.
        data_path (str): Path to data directory.

    Returns:
        np.ndarray: Cleaned data array.
    """
    df = pd.read_csv(os.path.join(data_path, file))
    df = df.drop(columns=['timestamp'], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.values

def run_dmd_variant(dmd_model, name: str, X_pca: np.ndarray, data_path: str):
    """
    Fit and visualize DMD variant.

    Parameters:
        dmd_model: DMD, HODMD, or MrDMD instance.
        name (str): Name of the method.
        X_pca (np.ndarray): PCA-reduced data.
        data_path (str): Path to save results.
    """
    try:
        dmd_model.fit(X_pca)
        plt.figure(figsize=(10, 4))
        for i, mode in enumerate(dmd_model.modes.T):
            plt.plot(np.real(mode), label=f'Mode {i+1}')
        plt.title(f'{name} Modes')
        plt.xlabel('Feature Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, f'{name.lower()}_modes.png'))
        plt.close()

        reconstructed = dmd_model.reconstructed_data.real
        np.savetxt(os.path.join(data_path, f'{name.lower()}_reconstructed.csv'), reconstructed.T, delimiter=',')
        print(f"[SUCCESS] {name} complete. Results saved.")
    except Exception as e:
        print(f"[ERROR] {name} failed:", e)

def main():
    """
    Main function for dynamic building modeling using Dynamic Mode Decomposition.
    """

    data_path = '../data_fragmentation/out/<data_path>'

    sensors_data = load_csv('data_sensors.csv', data_path)
    actuators_data = load_csv('data_actuators.csv', data_path)
    configuration_data = load_csv('data_configuration.csv', data_path)

    sensors_clean, actuators_clean, configuration_clean = process_data_simple_interpolation(
        sensors_data, actuators_data, configuration_data, interpolation_method="linear"
    )

    combined = np.concatenate([sensors_clean, actuators_clean, configuration_clean], axis=1)
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    std_safe = np.where(std == 0, 1, std)
    combined = (combined - mean) / std_safe
    X = combined.T

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X.T).T

    run_dmd_variant(DMD(svd_rank=-1), 'DMD', X_pca, data_path)
    run_dmd_variant(HODMD(svd_rank=-1, d=2), 'HODMD', X_pca, data_path)

    base_dmd = DMD(svd_rank=-1)
    mrdmd = MrDMD(base_dmd, max_level=2, max_cycles=1)
    run_dmd_variant(mrdmd, 'MrDMD', X_pca, data_path)


if __name__ == "__main__":
    main()
