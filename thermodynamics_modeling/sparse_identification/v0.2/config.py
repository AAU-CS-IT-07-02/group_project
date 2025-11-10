"""
Configuration Management for Dynamic Building Modeling

This module handles all configuration-related functionality including:
- Command-line argument parsing
- Future: YAML configuration file loading
- Configuration validation and merging

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import argparse
from typing import Optional


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
        python main.py --sensors data_sensors.csv --polynomial-degree 3 --threshold 0.05
        ```
    """
    p = argparse.ArgumentParser(description="Dynamic modeling of smart building systems using sparse identification techniques.")
    
    # Data file paths
    p.add_argument("--sensors", default="../data_fragmentation/out/data_sensors.csv", 
                   help="Path to sensors data file.")
    p.add_argument("--actuators", default="../data_fragmentation/out/data_actuators.csv", 
                   help="Path to actuators data file.")
    p.add_argument("--configuration", default="../data_fragmentation/out/data_configuration.csv", 
                   help="Path to configuration data file.")
    p.add_argument("--outdir", default="out", 
                   help="Output directory for results.")
    p.add_argument("--sep", default=None, 
                   help="CSV delimiter. If omitted, auto-detect.")
    
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


def load_config_yaml(yaml_path: str) -> dict:
    """
    Load configuration from YAML file (Future implementation).
    
    Parameters:
        yaml_path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Note:
        This function is planned for future implementation to support
        configuration files alongside command-line arguments.
    """
    # TODO: Implement YAML configuration loading
    # This will allow users to specify complex configurations in files
    # and override specific parameters via command line
    raise NotImplementedError("YAML configuration loading not yet implemented")


def merge_config(args: argparse.Namespace, yaml_config: Optional[dict] = None) -> argparse.Namespace:
    """
    Merge command-line arguments with YAML configuration (Future implementation).
    
    Parameters:
        args: Parsed command-line arguments
        yaml_config: Optional YAML configuration dictionary
        
    Returns:
        argparse.Namespace: Merged configuration with CLI args taking precedence
        
    Note:
        This function is planned for future implementation to support
        hierarchical configuration management.
    """
    # TODO: Implement configuration merging logic
    # Priority: CLI args > YAML config > defaults
    if yaml_config is not None:
        raise NotImplementedError("Configuration merging not yet implemented")
    
    return args


def validate_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Validate configuration parameters and apply business logic constraints.
    
    Parameters:
        args: Configuration to validate
        
    Returns:
        argparse.Namespace: Validated and potentially modified configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate file paths exist (basic check)
    import os
    
    # Validate numerical constraints
    if args.polynomial_degree < 1:
        raise ValueError("Polynomial degree must be >= 1")
    
    if args.threshold < 0:
        raise ValueError("Threshold must be >= 0")
    
    if not (0 < args.train_split < 1):
        raise ValueError("Train split must be between 0 and 1")
    
    if args.dt <= 0:
        raise ValueError("Time step (dt) must be > 0")
    
    if args.sampling_rate < 1:
        raise ValueError("Sampling rate must be >= 1")
    
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    return args