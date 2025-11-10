"""
Configuration Management for Dynamic Building Modeling

This module handles all configuration-related functionality including:
- Command-line argument parsing
- Future: YAML configuration file loading    # Feature library options
    p.add_argument("--feature-library", default="polynomial", 
                   choices=["polynomial", "fourier", "identity", "custom", "pde", "weakpde", "parameterized"], 
                   help="Type of feature library to use (single library mode)")
    p.add_argument("--feature-libraries", nargs='+', default=None,
                   help="List of feature libraries to combine (e.g., polynomial fourier identity custom pde)")
    p.add_argument("--library-combination-strategy", default="concat", 
                   choices=["concat", "tensored", "generalized"],
                   help="Strategy to combine multiple feature libraries")
    p.add_argument("--fourier-n-frequencies", type=int, default=2, 
                   help="Number of frequencies for Fourier library")
    p.add_argument("--no-interactions", action="store_true", 
                   help="Disable interaction terms in feature library (default: include interactions)")
    p.add_argument("--library-parameters", default=None,
                   help="JSON string with per-library parameters (e.g., '{\"polynomial\": {\"degree\": 3}, \"fourier\": {\"n_frequencies\": 5}}')")iguration validation and merging

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import os
import argparse
from typing import Optional, Dict, Any
import yaml


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
            
    If --config is provided, YAML configuration will be loaded and merged
    with CLI arguments, with CLI arguments taking precedence.
    
    Example:
        ```bash
        python main.py --sensors data_sensors.csv --polynomial-degree 3 --threshold 0.05
        python main.py --config config.yaml --threshold 0.01  # Override YAML threshold
        ```
    """
    p = argparse.ArgumentParser(description="Dynamic modeling of smart building systems using sparse identification techniques.")
    
    # Add all argument definitions
    _add_all_arguments(p)
    
    # Parse command-line arguments
    args = p.parse_args()
    
    # If config file is specified, load and merge YAML configuration
    if args.config:
        try:
            yaml_config = load_config_yaml(args.config)
            args = merge_config(args, yaml_config)
            print(f"✓ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"✗ Error loading configuration file: {e}")
            raise
    
    return args


def load_config_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters:
        yaml_path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary loaded from YAML file
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        
    Example:
        ```python
        config = load_config_yaml("config.yaml")
        ```
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            if config is None:
                config = {}
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file {yaml_path}: {e}")
    except Exception as e:
        raise Exception(f"Error reading configuration file {yaml_path}: {e}")


def merge_config(args: argparse.Namespace, yaml_config: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    """
    Merge command-line arguments with YAML configuration.
    
    Parameters:
        args: Parsed command-line arguments
        yaml_config: Optional YAML configuration dictionary
        
    Returns:
        argparse.Namespace: Merged configuration with CLI args taking precedence
        
    Priority order:
        1. Command-line arguments (highest priority)
        2. YAML configuration values
        3. Default values (lowest priority)
        
    Example:
        ```python
        yaml_config = {"polynomial_degree": 3, "threshold": 0.05}
        merged_args = merge_config(args, yaml_config)
        ```
    """
    if yaml_config is None:
        return args
    
    # Convert argparse.Namespace to dict for easier manipulation
    args_dict = vars(args)
    
    # Get the default argument parser to identify which values are defaults
    temp_parser = argparse.ArgumentParser()
    _add_all_arguments(temp_parser)
    defaults = vars(temp_parser.parse_args([]))
    
    # Merge YAML config into args, but only if the arg value is still the default
    for yaml_key, yaml_value in yaml_config.items():
        # Convert YAML key format (snake_case) to argument format (hyphen-case)
        arg_key = yaml_key.replace('-', '_')
        
        # Only override if the current value is the default (CLI args take precedence)
        if arg_key in args_dict and args_dict[arg_key] == defaults.get(arg_key):
            args_dict[arg_key] = yaml_value
    
    # Convert back to Namespace
    return argparse.Namespace(**args_dict)


def _add_all_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Helper function to add all arguments to a parser (used for getting defaults).
    
    Parameters:
        parser: ArgumentParser instance to add arguments to
    """
    # Configuration file
    parser.add_argument("--config", default=None, 
                   help="Path to YAML configuration file. CLI arguments override YAML settings.")
    
    # Data file paths
    parser.add_argument("--sensors", default="../data_fragmentation/out/data_sensors.csv", 
                   help="Path to sensors data file.")
    parser.add_argument("--actuators", default="../data_fragmentation/out/data_actuators.csv", 
                   help="Path to actuators data file.")
    parser.add_argument("--configuration", default="../data_fragmentation/out/data_configuration.csv", 
                   help="Path to configuration data file.")
    parser.add_argument("--outdir", default="out", 
                   help="Output directory for results.")
    parser.add_argument("--sep", default=None, 
                   help="CSV delimiter. If omitted, auto-detect.")
    
    # Data processing hyperparameters
    parser.add_argument("--interpolation-method", default="linear", 
                   choices=["linear", "time", "index", "nearest", "zero", "slinear", "quadratic", "cubic"],
                   help="Method for interpolating missing values")
    parser.add_argument("--include-configuration", action="store_true", 
                   help="Include configuration variables in state vector (default: sensors only)")
    parser.add_argument("--dt", type=float, default=5, 
                   help="Time step between measurements")
    
    # SINDy model hyperparameters
    parser.add_argument("--polynomial-degree", type=int, default=2, 
                   help="Degree of polynomial features for SINDy")
    parser.add_argument("--threshold", type=float, default=0.1, 
                   help="Sparsity threshold for STLSQ optimizer")
    parser.add_argument("--alpha", type=float, default=0.0, 
                   help="Regularization parameter for STLSQ optimizer")
    parser.add_argument("--max-iter", type=int, default=20, 
                   help="Maximum iterations for STLSQ optimizer")
    parser.add_argument("--normalize-columns", action="store_true",
                   help="Normalize feature matrix columns")
    parser.add_argument("--coefficient-threshold", type=float, default=1000.0, 
                   help="Maximum allowed coefficient magnitude (for stability)")
    
    # Training/validation hyperparameters
    parser.add_argument("--train-split", type=float, default=0.7, 
                   help="Fraction of data to use for training (rest for validation)")
    parser.add_argument("--skip-validation", action="store_true", 
                   help="Skip validation step after training")
    parser.add_argument("--skip-visualization", action="store_true", 
                   help="Skip plotting and visualization")
    parser.add_argument("--validation-subsample", type=int, default=1, 
                   help="Subsample validation data for faster simulation (1=full, 10=every 10th sample)")
    parser.add_argument("--simulation-timeout", type=int, default=60, 
                   help="Maximum time (seconds) to wait for simulation before timeout")
    
    # Data normalization options
    parser.add_argument("--normalize-data", action="store_true", 
                   help="Normalize data before training")
    parser.add_argument("--normalization-method", default="minmax", choices=["minmax", "standard", "robust"], 
                   help="Data normalization method")
    
    # Feature library options
    parser.add_argument("--feature-library", default="polynomial", 
                   choices=["polynomial", "fourier", "identity", "custom", "pde", "weakpde", "parameterized"], 
                   help="Type of feature library to use (single library mode)")
    parser.add_argument("--feature-libraries", nargs='+', default=None,
                   help="List of feature libraries to combine (e.g., polynomial fourier identity custom pde)")
    parser.add_argument("--library-combination-strategy", default="concat", 
                   choices=["concat", "tensored", "generalized"],
                   help="Strategy to combine multiple feature libraries")
    parser.add_argument("--fourier-n-frequencies", type=int, default=2, 
                   help="Number of frequencies for Fourier library")
    parser.add_argument("--no-interactions", action="store_true", 
                   help="Disable interaction terms in feature library (default: include interactions)")
    parser.add_argument("--library-parameters", default=None,
                   help="JSON string with per-library parameters (e.g., '{\"polynomial\": {\"degree\": 3}, \"fourier\": {\"n_frequencies\": 5}}')")
    
    # Optimizer options
    parser.add_argument("--optimizer", default="stlsq", choices=["stlsq", "lasso", "ridge"], 
                   help="Optimizer type for SINDy")
    parser.add_argument("--lasso-alpha", type=float, default=0.01, 
                   help="Alpha parameter for Lasso optimizer")
    
    # System monitoring options
    parser.add_argument("--monitor-interval", type=int, default=20, 
                   help="Interval in seconds for logging system usage (CPU, RAM, etc). Set to 0 to disable monitoring.")
    
    # Data sampling options for performance optimization
    parser.add_argument("--sampling-rate", type=int, default=1, 
                   help="Downsample data by taking every N-th sample (1=no sampling, 10=10x speedup)")