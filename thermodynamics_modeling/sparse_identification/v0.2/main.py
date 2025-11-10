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
import threading
import psutil

# Import our modular components
from config import parse_args, validate_config
from data_processing import load_and_process_data
from sindy_modeling import build_and_validate_model


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


def print_job_configuration(args):
    """Print comprehensive parameter summary for cluster data collection."""
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


def print_job_summary(args, total_duration: float, results: dict):
    """Print parseable summary for cluster data collection."""
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
    print(f"MODEL_STABLE={results.get('stable', False)}")
    print(f"VALIDATION_RMSE={results.get('rmse', 'N/A')}")
    print("CLUSTER_DATA_SUMMARY_END")
    
    print("")
    print("="*80)
    print("JOB COMPLETED - TIMING SUMMARY")
    print("="*80)
    print(f"Total Duration: {hours:02d}h {minutes:02d}m {seconds:05.2f}s ({total_duration:.2f}s total)")
    print(f"Start Time:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - total_duration))}")
    print(f"End Time:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    print("="*80)


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