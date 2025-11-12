## Smart Building Dynamic Modeling

The `dynamic_model_smart_building.py` script implements a comprehensive **data-driven approach** for modeling the thermodynamic behavior of the AAU BUILD facility using **Sparse Identification of Nonlinear Dynamics (SINDy)**. This script represents the core implementation of **Milestone 1** in the project.

### Purpose and Context

Unlike the previous examples that work with synthetic mathematical systems (oscillators, Lorenz system), this script tackles **real-world building data** from AAU BUILD's sensor network. The goal is to discover the mathematical equations governing how the building responds to environmental conditions and control inputs.

### Key Features

**Data Integration**: The script processes three types of data:

- **Sensor data**: Temperature, CO₂, humidity, occupancy sensors throughout the building
- **Actuator data**: HVAC setpoints, ventilation controls, blind positions
- **Configuration data**: Room settings, schedules, operational parameters

**Robust Data Processing**: Real building data comes with challenges:

- Missing values from sensor failures or communication issues
- Different measurement scales (°C, ppm, people count)
- Irregular sampling rates and timestamp synchronization

**Flexible Model Architecture**: The script supports multiple approaches:

- **Feature Libraries**: Polynomial (for nonlinear thermal relationships), Fourier (for daily/seasonal cycles), Identity (for linear dependencies)
- **Optimizers**: STLSQ with configurable sparsity thresholds and regularization
- **Normalization Methods**: MinMax, Standard, and Robust scaling for numerical stability

### Mathematical Framework

The building is modeled as a **controlled dynamical system**:

```
dX/dt = f(X, U)
```

Where:

- **X** represents the building state (temperatures, CO₂ levels, etc.)
- **U** represents control inputs (heating/cooling setpoints, ventilation rates)
- **f** is the function that SINDy discovers from data

This formulation allows the model to predict how the building will respond to different control strategies, which is essential for **Model Predictive Control (MPC)** implementation in Milestone 2.

### Usage Example

```bash
python dynamic_model_smart_building.py \
    --sensors ../data_fragmentation/out/data_sensors.csv \
    --actuators ../data_fragmentation/out/data_actuators.csv \
    --polynomial-degree 2 \
    --threshold 0.1 \
    --train-split 0.7 \
    --normalize-data
```

### Validation and Results

The script implements **temporal validation** by:

1. Training on historical data (70% by default)
2. Simulating forward in time using the discovered equations
3. Comparing predictions with actual sensor measurements
4. Computing RMSE and visualizing prediction accuracy

## Smart Building Dynamic Modeling API
::: thermodynamics_modeling.sparse_identification.dynamic_model_smart_building

## Data fragmentation for the SBDM API
::: thermodynamics_modeling.data_fragmentation.split_by_rooms_category_timeframe

