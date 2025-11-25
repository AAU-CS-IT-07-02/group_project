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
import os
import inspect
import concurrent.futures
import signal
from typing import Tuple, Optional, Any

from data_processing import prepare_training_data

def create_single_library(library_type: str, parameters: dict):
    """
    Create a single PySINDy feature library with specified parameters.
    
    Parameters:
        library_type: Type of library ('polynomial', 'fourier', 'identity', 'custom', 'pde')
        parameters: Dictionary of library-specific parameters
        
    Returns:
        PySINDy feature library object
    """
     
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
        # TODO: revisit this and make sure it works with the PDE optimizer
        # TODO: im sure PDELibrary needs way more parameters, make sure to trace them back to argparse and YAML
        # PDE library for partial differential equations
        library_functions = parameters.get('library_functions', None)
        derivative_order = parameters.get('derivative_order', 0)
        spatial_grid = parameters.get('spatial_grid', None)
        if library_functions is not None:
            return ps.PDELibrary(library_functions=library_functions, 
                               derivative_order=derivative_order,
                               spatial_grid=spatial_grid)
        else:
            return ps.PDELibrary()
    elif library_type == "weakpde":
        # Weak PDE library for weak formulation problems
        library_functions = parameters.get('library_functions', None)
        derivative_order = parameters.get('derivative_order', 0)
        spatiotemporal_grid = parameters.get('spatiotemporal_grid', None)
        if library_functions is not None:
            return ps.WeakPDELibrary(library_functions=library_functions,
                                   derivative_order=derivative_order,
                                   spatiotemporal_grid=spatiotemporal_grid)
        else:
            return ps.WeakPDELibrary()
    elif library_type == "parameterized":
        # Parameterized library - tensor product with different inputs
        libraries = parameters.get('libraries', [ps.PolynomialLibrary(), ps.PolynomialLibrary()])
        library_ensemble = parameters.get('library_ensemble', False)
        return ps.ParameterizedLibrary(libraries[0], libraries[1], 
                                     library_ensemble=library_ensemble)
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


def prepare_optimizer_kwargs(args: argparse.Namespace) -> dict:
    """
    Prepare optimizer kwargs by filtering args to only include parameters relevant to the specific optimizer.
    
    This function converts argparse.Namespace to dict and filters out parameters that are not
    supported by the specified optimizer to prevent TypeError when passing kwargs.
    
    Parameters:
        args: argparse.Namespace containing all CLI/config parameters
        optimizer_type: Type of optimizer ('stlsq', 'sr3', 'frols', etc.)
        
    Returns:
        dict: Filtered kwargs dictionary containing only parameters relevant to the optimizer
        
    Raises:
        ValueError: If unknown optimizer_type is specified
    """
    """
    Prepare optimizer kwargs by filtering args to include only parameters accepted by the
    chosen PySINDy optimizer. This implementation introspects the optimizer class
    signature (via inspect.signature) to avoid brittle hard-coded parameter lists.
    
    Parameters:
        args: argparse.Namespace containing CLI/config parameters
        optimizer_type: string key for optimizer (e.g. 'stlsq', 'sr3', ...)
    
    Returns:
        dict of filtered kwargs suitable to pass to the optimizer constructor
    """

    all_kwargs = vars(args).copy()

    opt_cls = getattr(ps, args.optimizer.upper(), None)

    # Inspect the constructor signature and accept only supported parameters
    sig = inspect.signature(opt_cls.__init__)
    allowed = set(sig.parameters.keys())
    # remove 'self' if present
    allowed.discard('self')

    filtered_kwargs = {}
    for k, v in all_kwargs.items():
        if k in allowed and v is not None:
            filtered_kwargs[k] = v

    return filtered_kwargs


def create_optimizer(args: argparse.Namespace) -> Any:
    """
    Create a sparse regression optimizer for SINDy model training.
    
    The optimizer determines which terms are included in the discovered equations
    by enforcing sparsity (keeping only the most important relationships).
    
    Currently supported optimizers:
    - STLSQ: Sequential Thresholded Least Squares (default)
    
    Future optimizers planned:
    - SR3: Sparse Relaxed Regularized Regression
    - FROLS: Forward Regression with Orthogonal Least Squares
    - SSR: Stepwise Sparse Regression
    - ConstrainedSR3: Constrained SR3
    - MIOSR: Mixed Integer Optimization for Sparse Regression
    
    Parameters:
        args: Configuration namespace containing optimizer parameters from CLI/YAML
        
    Returns:
        PySINDy optimizer object configured with specified parameters
        
    Raises:
        ValueError: If unknown optimizer_type is specified
    """
    
    optimizer_kwargs = prepare_optimizer_kwargs(args)

    # Simple mapping to known pysindy optimizer class names. Use getattr to
    # avoid AttributeError if a class isn't present in the installed pysindy.
    name = args.optimizer.lower()
    mapping = {
        'stlsq': 'STLSQ',
        'sr3': 'SR3',
        'constrainedsr3': 'ConstrainedSR3',
        'stablelinearsr3': 'StableLinearSR3',
        'trappingsr3': 'TrappingSR3',
        'ssr': 'SSR',
        'frols': 'FROLS',
        'sindypi': 'SINDyPI',
        'miosr': 'MIOSR',
    }

    cls_name = mapping.get(name)
    if cls_name is None:
        raise ValueError(f"Unknown optimizer type: {args.optimizer}")

    opt_cls = getattr(ps, cls_name, None)
    if opt_cls is None:
        raise ValueError(f"Optimizer '{cls_name}' is not available in the installed pysindy. "
                         "Please install/upgrade pysindy or choose a different optimizer.")

    return opt_cls(**optimizer_kwargs)

def build_model(args: argparse.Namespace) -> ps.SINDy:
    """
    Build a SINDy model architecture without training.
    
    Parameters:
        args: Configuration namespace with model parameters
        
    Returns:
        ps.SINDy: Untrained SINDy model ready for fitting
    """
    feature_library = create_feature_library(args)
    
    optimizer = create_optimizer(args)
    
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


def fit_model(model: ps.SINDy, X: np.ndarray, U: np.ndarray, args: argparse.Namespace, feature_names: Optional[list] = None) -> ps.SINDy:
    """
    Fit/train the SINDy model on the provided data.
    
    Parameters:
        model: Untrained SINDy model
        X: State variables (sensor measurements)
        U: Control inputs (actuator commands)
        args: Configuration namespace with model parameters
        feature_names: Optional list of feature names for named variables in equations
        
    Returns:
        ps.SINDy: Trained SINDy model
    """
    print(f"\nFitting SINDy model to data...")
    print(f"  Training data shape: X={X.shape}, U={U.shape}")
    print(f"  Time step: {args.dt}")
    if feature_names:
        print(f"  Using named variables: {feature_names}")
    
    fit_start = time.time()
    model.fit(X, u=U, t=args.dt, feature_names=feature_names)
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
                  args: argparse.Namespace, feature_names: Optional[list] = None) -> Tuple[float, Optional[np.ndarray]]:
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
    model.fit(X_train, u=U_train, t=args.dt, feature_names=feature_names)
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
        
        # Run simulation with a thread-based timeout (cross-platform)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(simulate_with_timeout)
            try:
                X_pred = future.result(timeout=max_sim_time)
            except concurrent.futures.TimeoutError:
                future.cancel()
                print(f"    Simulation timed out after {max_sim_time}s")
                print(f"    Try using --validation-subsample for faster validation")
                raise TimeoutError(f"Simulation timed out after {max_sim_time}s")
        
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


def validate_model_pipeline(model: ps.SINDy, X: np.ndarray, U: np.ndarray, t: np.ndarray, args: argparse.Namespace, feature_names: Optional[list] = None) -> dict:
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
        rmse, X_pred = validate_model(model, X_train, X_test, U_train, U_test, t_test, args, feature_names)
        
        # Create validation plots
        if X_pred is not None:
            create_validation_plots(X_test, X_pred, rmse, args)
        
        results['rmse'] = rmse
        results['predictions'] = X_pred
        
        validation_total = time.time() - validation_start
        print(f"Total validation time: {validation_total:.2f}s")
    
    return results