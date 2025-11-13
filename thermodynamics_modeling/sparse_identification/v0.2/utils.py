"""
Utility functions for the dynamic building modeling pipeline.

This module contains system monitoring, job reporting, and other utility functions
that support the main workflow but are not core domain logic.

Functions:
    - System resource monitoring and logging
    - Job configuration and summary reporting  
    - Background task management
    - Timing and progress utilities

Authors: AAU CS Master's Team (Group Project 2025)
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import time
import threading
import psutil


def log_system_usage():
    """Log current CPU and RAM usage."""
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    print(f"[MONITOR] CPU: {cpu:.1f}% | RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")


def start_monitoring(interval_seconds: int):
    """
    Start background monitoring thread for system resource tracking.
    
    Parameters:
        interval_seconds: Time interval between monitoring reports (0 to disable)
        
    Returns:
        Thread object if monitoring started, None if disabled
        
    Note:
        Uses daemon thread that automatically terminates when main process exits
    """
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
    """
    Print comprehensive parameter summary for cluster data collection.
    
    Parameters:
        args: Parsed command-line arguments namespace
        
    Note:
        Formatted for easy parsing by cluster job management systems
    """
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
    """
    Print parseable summary for cluster data collection and job analysis.
    
    Parameters:
        args: Parsed command-line arguments namespace
        total_duration: Total job execution time in seconds
        results: Dictionary containing model results and metrics
        
    Note:
        Output format designed for automated parsing by cluster analysis scripts
    """
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


def format_time_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Parameters:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45.67s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes:02d}m {remaining_seconds:05.2f}s"
    elif minutes > 0:
        return f"{minutes}m {remaining_seconds:05.2f}s"
    else:
        return f"{remaining_seconds:.2f}s"


def create_results_dict(**kwargs) -> dict:
    """
    Create standardized results dictionary for job summary.
    
    Parameters:
        **kwargs: Key-value pairs for results
        
    Returns:
        Dictionary with results for job summary reporting
    """
    return {
        'stable': kwargs.get('stable', False),
        'rmse': kwargs.get('rmse', 'N/A'),
        'model_complexity': kwargs.get('model_complexity', 'N/A'),
        'training_time': kwargs.get('training_time', 'N/A'),
        'validation_time': kwargs.get('validation_time', 'N/A'),
        **kwargs  # Include any additional results
    }