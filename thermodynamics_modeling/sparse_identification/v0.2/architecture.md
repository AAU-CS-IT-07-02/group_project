# SINDy Building Dynamics Modeling - Architecture Documentation

## Overview

This document describes the modular architecture of the SINDy (Sparse Identification of Nonlinear Dynamics) building modeling system. The codebase has been split into focused modules to improve maintainability, readability, and enable efficient collaboration with Large Language Models (LLMs).

## File Structure

```
v0.2/
├── __init__.py              # Package initialization
├── main.py                  # Main orchestration script (~120 lines)
├── config.py                # Configuration management (~120 lines)
├── data_processing.py       # Data loading and preprocessing (~250 lines)
├── sindy_modeling.py        # SINDy model operations (~320 lines)
├── utils.py                 # Utility functions (~80 lines)
└── architecture.md          # This documentation file
```

## Module Responsibilities

### 1. `main.py` - Orchestration Layer
**Purpose**: Coordinates the entire modeling pipeline and job execution.

**Key Functions**:
- `main()`: Primary workflow orchestration
- Job timing and progress tracking
- Error handling and cleanup
- Result summary generation

**Dependencies**: All other modules
**Size**: ~120 lines - optimal for LLM context windows

### 2. `config.py` - Configuration Management
**Purpose**: Handles all configuration parsing and validation.

**Key Functions**:
- `parse_args()`: Command-line argument parsing with 30+ parameters
- Future: YAML configuration file support
- Parameter validation and defaults

**Dependencies**: `argparse`, `typing`
**Size**: ~120 lines
**Future Expansion**: YAML config loading, configuration validation

### 3. `data_processing.py` - Data Pipeline
**Purpose**: All data-related operations from loading to preprocessing.

**Key Functions**:
- `load_and_process_data()`: Main data pipeline entry point
- `get_csv_data()`: CSV file loading with encoding handling
- `process_data_interpolation()`: Missing value interpolation
- `normalize_data()`: Data scaling and normalization
- `downsample_data()`: Performance optimization sampling
- `prepare_state_space()`: Combine sensors/actuators into state variables

**Dependencies**: `pandas`, `numpy`, `sklearn`
**Size**: ~250 lines
**Data Flow**: CSV files → Raw arrays → Processed arrays → Normalized data

### 4. `sindy_modeling.py` - Core Modeling Engine
**Purpose**: SINDy model construction, training, validation, and analysis.

**Key Functions**:
- `build_and_validate_model()`: Main modeling pipeline entry point
- `create_feature_library()`: Feature library construction (polynomial, Fourier, identity)
- `create_optimizer()`: Sparse regression optimizer setup (STLSQ, future: SR3, FROLS)
- `train_sindy_model()`: Model training and equation discovery
- `validate_model()`: Train/test validation with simulation
- `create_validation_plots()`: Visualization and error analysis
- `check_model_stability()`: Coefficient magnitude analysis

**Dependencies**: `pysindy`, `matplotlib`, `numpy`
**Size**: ~320 lines
**Future Expansion**: Differentiators, integrators, additional optimizers

### 5. `utils.py` - Supporting Utilities
**Purpose**: System monitoring, logging, and helper functions.

**Key Functions**:
- `start_monitoring()`: Background system resource monitoring
- `log_system_usage()`: CPU/RAM usage logging
- `print_job_configuration()`: Comprehensive parameter summary
- `print_job_summary()`: Final results and timing summary

**Dependencies**: `psutil`, `threading`, `time`
**Size**: ~80 lines

## Data Flow Architecture

```
CLI Args → config.py → main.py
                         ↓
CSV Files → data_processing.py → Processed Data (X, U, t)
                                       ↓
                              sindy_modeling.py → Trained Model + Validation
                                       ↓
                                utils.py → Monitoring & Reporting
```

## LLM Working Guidelines

### For Code Modifications

1. **Single Module Focus**: When making changes, focus on one module at a time
2. **Interface Preservation**: Maintain function signatures and return types
3. **Import Dependencies**: Check imports when adding new functionality
4. **Error Handling**: Follow existing error handling patterns in each module

### Module-Specific Guidelines

#### Working with `config.py`:
- Add new CLI arguments in logical groups
- Maintain type hints and help strings
- Consider future YAML integration when adding parameters

#### Working with `data_processing.py`:
- Handle pandas/numpy type conversions carefully
- Preserve data shape information in docstrings
- Add data quality checks for new preprocessing steps

#### Working with `sindy_modeling.py`:
- Test model stability with coefficient checks
- Add visualization for new validation metrics
- Document mathematical assumptions in docstrings

#### Working with `utils.py`:
- Keep functions pure and stateless where possible
- Add appropriate logging levels
- Consider thread safety for monitoring functions

### Common Patterns

#### Function Interface Pattern:
```python
def process_function(input_data: Type, args: argparse.Namespace) -> OutputType:
    """
    Brief description.
    
    Parameters:
        input_data: Description with expected shape/format
        args: Command-line arguments namespace
        
    Returns:
        Processed data with description
        
    Raises:
        SpecificError: When this specific condition occurs
    """
```

#### Error Handling Pattern:
```python
try:
    result = complex_operation()
    print(f"✓ Operation successful: {summary}")
except SpecificError as e:
    print(f"✗ Operation failed: {e}")
    # Provide helpful suggestions
    raise
```

## Extension Points

### Adding New Features

1. **New Data Sources**: Extend `data_processing.py` with new loader functions
2. **New SINDy Optimizers**: Add to `sindy_modeling.py` optimizer factory
3. **New Validation Metrics**: Extend validation functions in `sindy_modeling.py`
4. **YAML Configuration**: Add to `config.py` while maintaining CLI compatibility

### Performance Optimization

1. **Data Pipeline**: Optimize pandas operations in `data_processing.py`
2. **Model Training**: Add parallel processing options in `sindy_modeling.py`
3. **Memory Management**: Monitor and optimize large array operations

## Testing Strategy

### Unit Testing Structure:
```
tests/
├── test_config.py           # Configuration parsing tests
├── test_data_processing.py  # Data pipeline tests
├── test_sindy_modeling.py   # Model building tests
└── test_utils.py           # Utility function tests
```

### Integration Testing:
- End-to-end pipeline tests using sample data
- Regression tests for model accuracy
- Performance benchmarks for different configurations

## Debugging Guidelines

### Common Issues and Solutions:

1. **Import Errors**: Check `__init__.py` and relative imports
2. **Data Shape Mismatches**: Verify array dimensions in data_processing.py
3. **Model Instability**: Check coefficient magnitudes in sindy_modeling.py
4. **Memory Issues**: Monitor data sizes in utils.py logging

### Debugging Workflow:
1. Check configuration parsing in `config.py`
2. Verify data loading and shapes in `data_processing.py`
3. Test model training step-by-step in `sindy_modeling.py`
4. Monitor system resources with `utils.py`

## Performance Considerations

### File Size Optimization:
- Each module is sized for optimal LLM context windows (80-320 lines)
- Related functionality is grouped to minimize cross-file dependencies
- Clear interfaces reduce the need to understand multiple modules simultaneously

### Runtime Optimization:
- Data downsampling options in `data_processing.py`
- Model complexity controls in `sindy_modeling.py`
- System monitoring in `utils.py` for resource awareness

## Future Roadmap

### Planned Enhancements:
1. **YAML Configuration**: Extend `config.py` with file-based configuration
2. **Advanced SINDy Features**: Add differentiators and integrators to `sindy_modeling.py`
3. **Enhanced Validation**: More sophisticated metrics and cross-validation
4. **Parallel Processing**: Multi-core support for large datasets
5. **Model Persistence**: Save/load trained models and preprocessing parameters

### Maintenance Guidelines:
- Keep modules focused on single responsibilities
- Maintain comprehensive docstrings and type hints
- Add new features to appropriate modules based on functionality
- Update this architecture document when making structural changes

---

**Note for LLMs**: When working with this codebase, focus on one module at a time. Each module is self-contained with clear interfaces, making it easier to understand and modify specific functionality without needing full system context.