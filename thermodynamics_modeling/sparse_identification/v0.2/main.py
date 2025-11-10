#!/usr/bin/env python3
"""
Dynamic Modeling of Smart Building Systems using Sparse Identification of Nonlinear Dynamics.

This is the main orchestration script that coordinates the entire pipeline for discovering
interpretable mathematical equations governing AAU BUILD facility behavior from real 
sensor and actuator data using PySINDy.

The modular structure provides:
- Clean separation of concerns across 4 focused modules
- Easy expansion and maintenance 
- Professional codebase suitable for production use
- LLM-friendly file sizes and interfaces

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
    python main.py \
        --sensors data_sensors.csv \
        --actuators data_actuators.csv \
        --polynomial-degree 2 \
        --threshold 0.1 \
        --normalize-data
    ```

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import time

# Import our modular components
from config import parse_args, validate_config
from data_processing import load_and_process_data
from sindy_modeling import build_and_validate_model
from utils import start_monitoring, print_job_configuration, print_job_summary, create_results_dict


def main():
    """
    Main orchestration function for dynamic building modeling using SINDy.
    
    This function coordinates the complete pipeline:
    1. Parse and validate configuration arguments
    2. Load and preprocess building sensor/actuator data
    3. Build and train SINDy model to discover governing equations
    4. Validate model performance using temporal simulation
    5. Generate analysis plots and performance reports
    
    The workflow is designed for both interactive use and cluster-based 
    experimentation with comprehensive logging and error handling.
    
    Raises:
        Exception: If critical errors occur during processing
        
    Example:
        The script maintains the same CLI interface as the original monolith:
        ```bash
        python main.py \
            --sensors data_sensors.csv \
            --actuators data_actuators.csv \
            --configuration data_configuration.csv \
            --polynomial-degree 3 \
            --threshold 0.05 \
            --normalize-data \
            --sampling-rate 10
        ```
    """
    # Record start time for job duration tracking
    start_time = time.time()
    
    try:
        # Parse and validate configuration
        args = parse_args()
        args = validate_config(args)
        
        # Print comprehensive parameter summary
        print_job_configuration(args)
        
        # Start system resource monitoring
        start_monitoring(args.monitor_interval)
        
        # Load and preprocess data
        X, U, t, scalers = load_and_process_data(args)
        
        # Build and validate SINDy model
        model, results = build_and_validate_model(X, U, t, args)
        
        # Calculate and print job completion summary
        end_time = time.time()
        total_duration = end_time - start_time
        print_job_summary(args, total_duration, results)
        
    except KeyboardInterrupt:
        print("\n\nJob interrupted by user (Ctrl+C)")
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Partial execution time: {total_duration:.2f}s")
        
    except Exception as e:
        print(f"\n\nJob failed with error: {e}")
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Execution time before failure: {total_duration:.2f}s")
        raise


if __name__ == "__main__":
    main()