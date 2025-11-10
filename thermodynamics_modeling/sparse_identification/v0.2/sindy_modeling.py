"""
SINDy Modeling for Dynamic Building Systems

This module handles all SINDy-related operations including:
- Feature library construction (polynomial, Fourier, identity)
- Optimizer configuration and model building
- Model validation and temporal simulation
- Visualization and error analysis
- Model stability checks and diagnostics

The module implements Sparse Identification of Nonlinear Dynamics (SINDy) to discover
interpretable mathematical equations governing building thermodynamics from sensor and
actuator data. Future expansions will include differentiators and integrators.

Authors: AAU CS Master's Team (Group Project 2025)  
Project: Intelligent Building Management System through Data-Driven Thermodynamics Modeling
"""

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import matplotlib
import argparse
import time
import signal
import os
from typing import Tuple, Optional, Any

from data_processing import prepare_training_data

def create_single_library(library_type: str, parameters: dict = None):
    """
    Create a single PySINDy feature library with specified parameters.
    
    Parameters:
        library_type: Type of library ('polynomial', 'fourier', 'identity', 'custom', 'pde')
        parameters: Dictionary of library-specific parameters
        
    Returns:
        PySINDy feature library object
    """
    if parameters is None:
        parameters = {}
        
    if library_type == "polynomial":
        degree = parameters.get('degree', 2)
        include_interaction = parameters.get('include_interaction', True)
        return ps.PolynomialLibrary(degree=degree, include_interaction=include_interaction)
    elif library_type == "fourier":
        n_frequencies = parameters.get('n_frequencies', 2)
        return ps.FourierLibrary(n_frequencies=n_frequencies)
    elif library_type == "identity":
        return ps.IdentityLibrary()
    elif library_type == "custom":
        # For future extension - custom function libraries
        functions = parameters.get('functions', [lambda x: x])
        function_names = parameters.get('function_names', ['x'])
        return ps.CustomLibrary(library_functions=functions, function_names=function_names)
    elif library_type == "pde":
        # For future extension - PDE libraries
        return ps.PDELibrary()
    else:
        raise ValueError(f"Unknown feature library type: {library_type}")


def create_feature_library(args):
    """
    Create a PySINDy feature library for building dynamics modeling.
    
    Supports both single and composite libraries:
    - Single: One library type (polynomial, fourier, identity)
    - Composite: Multiple libraries combined via concat/tensored/generalized strategies
    
    Different libraries capture different building thermodynamics aspects:
    - Polynomial: Nonlinear thermal relationships, heat transfer dependencies
    - Fourier: Daily/seasonal cycles, periodic occupancy patterns  
    - Identity: Linear relationships between variables
    
    Combination strategies:
    - Concat: Side-by-side feature concatenation
    - Tensored: Multiplicative combinations (tensor products)
    - Generalized: Flexible library organization
    
    Parameters:
        args: Namespace containing library configuration
        
    Returns:
        PySINDy feature library object
    """
    import json
    
    # Parse library-specific parameters
    library_params = {}
    if hasattr(args, 'library_parameters') and args.library_parameters:
        if isinstance(args.library_parameters, dict):
            # Already parsed (from YAML)
            library_params = args.library_parameters
        elif isinstance(args.library_parameters, str):
            # JSON string (from CLI)
            try:
                library_params = json.loads(args.library_parameters)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid library_parameters JSON: {e}")
                library_params = {}
        else:
            print(f"Warning: Invalid library_parameters type: {type(args.library_parameters)}")
            library_params = {}
    
    # Check if using composite libraries
    if hasattr(args, 'feature_libraries') and args.feature_libraries:
        # Composite library mode
        libraries = []
        
        for lib_type in args.feature_libraries:
            # Get parameters for this specific library
            lib_specific_params = library_params.get(lib_type, {})
            
            # Add global parameters as defaults
            if lib_type == "polynomial":
                lib_specific_params.setdefault('degree', getattr(args, 'polynomial_degree', 2))
                lib_specific_params.setdefault('include_interaction', not getattr(args, 'no_interactions', False))
            elif lib_type == "fourier":
                lib_specific_params.setdefault('n_frequencies', getattr(args, 'fourier_n_frequencies', 2))
            
            libraries.append(create_single_library(lib_type, lib_specific_params))
        
        # Combine libraries based on strategy
        combination_strategy = getattr(args, 'library_combination_strategy', 'concat')
        
        if combination_strategy == "concat":
            return ps.ConcatLibrary(libraries)
        elif combination_strategy == "tensored":
            return ps.TensoredLibrary(libraries)
        elif combination_strategy == "generalized":
            return ps.GeneralizedLibrary(libraries)
        else:
            raise ValueError(f"Unknown combination strategy: {combination_strategy}")
    
    else:
        # Single library mode (backward compatibility)
        library_type = getattr(args, 'feature_library', 'polynomial')
        single_params = library_params.get(library_type, {})
        
        # Add backward compatibility parameters
        if library_type == "polynomial":
            single_params.setdefault('degree', getattr(args, 'polynomial_degree', 2))
            single_params.setdefault('include_interaction', not getattr(args, 'no_interactions', False))
        elif library_type == "fourier":
            single_params.setdefault('n_frequencies', getattr(args, 'fourier_n_frequencies', 2))
        
        return create_single_library(library_type, single_params)


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


def build_model(args: argparse.Namespace) -> ps.SINDy:
    """
    Build a SINDy model architecture without training.
    
    Parameters:
        args: Configuration namespace with model parameters
        
    Returns:
        ps.SINDy: Untrained SINDy model ready for fitting
    """
    feature_library = create_feature_library(args)
    
    optimizer = create_optimizer(
        args.optimizer, 
        args.threshold, 
        args.alpha, 
        args.max_iter,
        args.normalize_columns, 
        args.lasso_alpha
    )
    
    print(f"\nBuilding SINDy model architecture...")
    
    # Display feature library configuration
    if hasattr(args, 'feature_libraries') and args.feature_libraries:
        print(f"  Feature libraries: {args.feature_libraries} (strategy: {getattr(args, 'library_combination_strategy', 'concat')})")
        if hasattr(args, 'library_parameters') and args.library_parameters:
            print(f"  Library parameters: {args.library_parameters}")
    else:
        print(f"  Feature library: {getattr(args, 'feature_library', 'polynomial')} (degree={getattr(args, 'polynomial_degree', 2)})")
    
    print(f"  Optimizer: {args.optimizer} (threshold={args.threshold})")
    
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    return model


def fit_model(model: ps.SINDy, X: np.ndarray, U: np.ndarray, args: argparse.Namespace) -> ps.SINDy:
    """
    Fit/train the SINDy model on the provided data.
    
    Parameters:
        model: Untrained SINDy model
        X: State variables (sensor measurements)
        U: Control inputs (actuator commands)
        args: Configuration namespace with model parameters
        
    Returns:
        ps.SINDy: Trained SINDy model
    """
    print(f"\nFitting SINDy model to data...")
    print(f"  Training data shape: X={X.shape}, U={U.shape}")
    print(f"  Time step: {args.dt}")
    
    fit_start = time.time()
    model.fit(X, u=U, t=args.dt)
    fit_time = time.time() - fit_start
    
    print(f"  Model fitting completed in {fit_time:.2f}s")
    
    return model


def check_model_stability(model: ps.SINDy, coefficient_threshold: float = 1000.0) -> bool:
    """
    Check SINDy model stability by analyzing coefficient magnitudes.
    
    Parameters:
        model: Trained SINDy model
        coefficient_threshold: Maximum allowed coefficient magnitude
        
    Returns:
        bool: True if model appears stable, False otherwise
    """
    coeffs = model.coefficients()
    max_coeff = np.abs(coeffs).max()
    
    print(f"\nModel stability check:")
    print(f"  Max coefficient magnitude: {max_coeff:.3f}")
    print(f"  Coefficient threshold: {coefficient_threshold}")
    
    if max_coeff > coefficient_threshold:
        print(f"  WARNING: Large coefficients detected! Model may be unstable.")
        print(f"  Suggestions:")
        print(f"    1. Increase sparsity threshold (current: {max_coeff * 0.1:.3f})")
        print(f"    2. Add regularization (try alpha: 0.1)")
        print(f"    3. Force normalization: --normalize-data")
        print(f"    4. Reduce polynomial degree")
        return False
    
    return True


def validate_model(model: ps.SINDy, X_train: np.ndarray, X_test: np.ndarray, 
                  U_train: np.ndarray, U_test: np.ndarray, t_test: np.ndarray, 
                  args: argparse.Namespace) -> Tuple[float, Optional[np.ndarray]]:
    """
    Validate SINDy model using temporal train/test split with simulation.
    
    Parameters:
        model: Trained SINDy model  
        X_train: Training state data
        X_test: Test state data
        U_train: Training control data
        U_test: Test control data
        t_test: Test time vector
        args: Configuration namespace
        
    Returns:
        tuple: (rmse, X_pred) where X_pred is None if simulation fails
    """
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
    
    # Predict on test data with safety checks
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
        short_t = t_test[:min(10, len(t_test))]  # Only first 10 timesteps
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
        rmse = np.sqrt(np.mean((X_test[:min_len] - X_pred[:min_len])**2))
        
        print(f"\nValidation RMSE: {rmse:.6f}")
        
        return rmse, X_pred
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return float('inf'), None


def create_validation_plots(X_test: np.ndarray, X_pred: np.ndarray, rmse: float, args: argparse.Namespace) -> None:
    """
    Create validation and error analysis plots for model assessment.
    
    Parameters:
        X_test: True test data
        X_pred: Model predictions
        rmse: Overall RMSE
        args: Configuration namespace
    """
    if args.skip_visualization or X_pred is None:
        return
    
    # Configure matplotlib for headless operation (cluster-friendly)
    matplotlib.use('Agg')  # Use non-interactive backend
    
    min_len = min(len(X_test), len(X_pred))
    
    # Main validation plot
    plt.figure(figsize=(12, 8))
    n_states = min(5, X_test.shape[1])  # Plot up to 5 states
    
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
    os.makedirs(args.outdir, exist_ok=True)
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Important: close to free memory
    print(f"Validation plot saved to: {plot_filename}")
    
    # Additional error analysis plot
    plt.figure(figsize=(10, 6))
    
    # Plot prediction errors over time
    error = X_test[:min_len] - X_pred[:min_len]
    plt.subplot(2, 1, 1)
    for i in range(min(3, X_test.shape[1])):
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


def validate_model_pipeline(model: ps.SINDy, X: np.ndarray, U: np.ndarray, t: np.ndarray, args: argparse.Namespace) -> dict:
    """
    Complete model validation pipeline including stability checks and temporal validation.
    
    Parameters:
        model: Trained SINDy model
        X: Full state variables dataset
        U: Full control inputs dataset  
        t: Full time vector
        args: Configuration namespace
        
    Returns:
        dict: Validation results containing stability, RMSE, and predictions
    """
    # Check model stability
    is_stable = check_model_stability(model, args.coefficient_threshold)
    
    # Print discovered equations
    print("\nDiscovered equations:")
    model.print()
    
    results = {
        'stable': is_stable,
        'coefficients': model.coefficients(),
        'rmse': None,
        'predictions': None
    }
    
    # Optional validation step
    if not args.skip_validation:
        
        validation_start = time.time()
        
        # Split data for validation
        X_train, X_test, U_train, U_test, t_train, t_test = prepare_training_data(
            X, U, t, args.train_split
        )
        
        # Validate model with simulation
        rmse, X_pred = validate_model(model, X_train, X_test, U_train, U_test, t_test, args)
        
        # Create validation plots
        if X_pred is not None:
            create_validation_plots(X_test, X_pred, rmse, args)
        
        results['rmse'] = rmse
        results['predictions'] = X_pred
        
        validation_total = time.time() - validation_start
        print(f"Total validation time: {validation_total:.2f}s")
    
    return results