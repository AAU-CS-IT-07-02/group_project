# Neural ODE for Building Thermodynamics Modeling - Project Summary

## Overview

This project implements a **Neural ODE (Ordinary Differential Equation) model** for predicting room temperatures in a multi-room building based on control inputs and environmental disturbances. The system uses PyTorch with `torchdiffeq` for solving continuous-time differential equations and is designed to model the thermodynamic dynamics of the AAU BUILD sensor network (6-room office environment).

### Key Concept
Instead of traditional RNNs or sequence models, this system:
- Encodes initial conditions and controls into a **latent state** (z)
- Uses a **neural ODE** to model continuous latent dynamics: `dz/dt = f(z, u(t))`
- Decodes the evolved latent state back to room temperature predictions at each timestep

---

## Architecture Components

### 1. **Core Neural ODE Model** (`torchdiffeq_model.py`)

The main training pipeline that orchestrates the entire workflow.

#### Key Classes:

**`NeuralODEModel`** (main model)
- **Encoder**: Maps [initial_temperature, control_input] → latent_state
  - Input: `[B, d_y + d_u]` (concatenated: room temps + control features)
  - Output: `[B, latent_dim]` (latent representation)
  
- **ODEFunc**: Defines latent dynamics `dz/dt = f(z, u(t))`
  - Takes interpolated control sequence and current latent state
  - Neural network that combines latent state and interpolated controls
  - Output: time derivative of latent state
  
- **Decoder**: Maps latent_state → predicted_temperatures
  - Input: `[N, latent_dim]`
  - Output: `[N, d_y]` (predicted room temperatures)

**`ControlInterpolator`**
- Implements **piecewise linear interpolation** for continuous control signals
- Takes discrete control sequence `[H, d_u]` and reference time points `[H]`
- Enables the ODE solver to query control values at any time `t`
- Supports batched and non-batched control inputs

**`WindowedDataset`**
- Creates **sliding windows** of length H from time-series data
- Each window contains: (controls_seq, y_seq, y0)
- Used for batched training with overlap

#### Data Flow:
1. Load CSV → Parse into dicts → Extract feature columns as tensors
2. Normalize controls and targets using mean/std statistics
3. Create sliding windows
4. For each window in training:
   - Encoder: `[y0, u0] → z0`
   - ODE Integration: `odeint(ODEFunc, z0, t_span, method) → z(t)`
   - Decoder: `z(t) → ŷ(t)`
   - Loss: MSE between predicted and actual temperatures

#### Configuration (`config.yml`):
- Feature lists: observations, disturbances, outdoor conditions, room temperatures
- Dataset params: horizon H, batch size, train/val/test split ranges
- Model params: latent dimension, number of epochs, learning rate warmup, early stopping patience
- Threading config for CPU optimization

#### Training Features:
- **Warmup Schedule**: Linear LR ramp for first N epochs
- **Early Stopping**: Stops if validation loss doesn't improve after `patience` epochs
- **Solver Options**: RK4, dopri5, etc. (via `torchdiffeq`)
- **Threading**: Configurable CPU thread allocation for numerical libraries

---

### 2. **Evaluation & Visualization** (`evaluate_and_plot.py`)

Comprehensive evaluation script that loads a trained model and generates analysis plots.

#### Features:

**Dynamic Module Loading**
- Uses `runpy.run_path()` to import utilities from `torchdiffeq_model.py` at runtime
- Avoids code duplication for CSV parsing, normalization, and model definition

**Window Selection Modes**:
- **`last`**: Plot the last N windows (useful for recent behavior)
- **`random`**: Randomly sample N windows from test set
- **`range`**: Plot all windows in index range [start_idx, end_idx]

**Output Metrics** (saved as JSON):
- **MAE (Mean Absolute Error)**: per-room prediction error in physical units
- **RMSE (Root Mean Square Error)**: per-room prediction variance
- Computed across all windows

**Visualization Outputs**:
1. **Time-Series Plots** (`ts_window_{start_idx}.png`)
   - 6 subplots (one per room)
   - Overlays: Actual (blue) vs Predicted (red dashed)
   - X-axis: timesteps [0...H-1], Y-axis: Temperature

2. **Parity Scatter Plot** (`parity_scatter.png`)
   - Aggregates all predictions across all windows
   - X-axis: Actual temperature, Y-axis: Predicted temperature
   - Includes y=x reference line for perfect predictions
   - Points below line: underprediction, above: overprediction

#### Command-Line Arguments:
```bash
python evaluate_and_plot.py \
  --test ./dataset_split/test_data.csv \
  --out ./out \
  --H 16 \
  --mode last \
  --start_idx -500 \
  --solver rk4 \
  --windows 1 \
  --dpi 140
```

#### Workflow:
1. Load trained model checkpoint and scalers
2. Load test CSV and normalize
3. Select windows based on mode
4. For each window:
   - Run inference: `model(y0, controls_seq, t_span, method=solver)`
   - Denormalize predictions
   - Save per-window time-series plot
5. Aggregate all windows and create parity scatter plot
6. Save metrics JSON

---

### 3. **Data Splitting** (`split_for_test.py`)

Utility script for train/test separation.

#### Functionality:
- **Input**: Full dataset CSV with timestamp column
- **Source**: `../../Database/AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv`
- **Logic**: Split by calendar month (configurable, e.g., month 3 for testing)
- **Output**: Two CSV files
  - `./dataset_split/train_data.csv`: All months except test month
  - `./dataset_split/test_data.csv`: Only test month
  
#### Note:
Timestamp parsing expects format: `%m/%d/%Y %H:%M`

---

### 4. **Model Export Scripts**

#### `to_onnx.py` - Export to ONNX Format
**Purpose**: Convert trained PyTorch model to ONNX (Open Neural Network Exchange) for deployment in other frameworks (C++, Java, JavaScript, etc.)

**Process**:
1. Load model weights and config
2. Create dummy inputs matching model signature
3. Export using `torch.onnx.export()` with:
   - Input/output names
   - Dynamic batch axis annotations
   - opset_version=17 for broad compatibility

**Inputs**: 
- `y0`: [batch_size, d_y] initial temperatures
- `controls_seq`: [batch_size, H, d_u] control sequence
- `t_span`: [H] time vector

**Output**: 
- `neural_ode_model.onnx` in `./out/` directory

**Limitation**: ONNX export works but may lose some custom ODE solver logic; users may need to implement ODE integration in the target framework.

---

#### `to_jit.py` - Export to TorchScript Format
**Purpose**: Convert to TorchScript (PyTorch's serialized format) for faster inference or C++ integration.

**Strategy** (graceful degradation):
1. **Attempt 1**: `torch.jit.trace()` - fastest, works for most models
2. **Fallback 1**: `torch.jit.script()` - slower but handles Python constructs
3. **Fallback 2**: Trace encoder/decoder separately
   - Rationale: `torchdiffeq.odeint()` is inherently Python-based and not TorchScriptable
   - Solution: C++ can use traced encoder/decoder for pre/post-processing while Python handles ODE integration
4. **Fallback 3**: Suggest ONNX export as final option

**Outputs**:
- `neural_ode_model_jit.pt` (traced) or `neural_ode_model_jit_scripted.pt` (scripted)
- Optionally: `encoder_jit.pt`, `decoder_jit.pt` (if full model tracing fails)

**Key Note**: The ODE integration step using `torchdiffeq.odeint()` cannot be directly compiled to TorchScript due to dynamic control flow. The fallback approach separates static components (encoder/decoder) from the dynamic integration.

---

## Data Pipeline

### Stage 1: Data Preparation
```
Original Dataset (raw CSV)
    ↓ [split_for_test.py]
├─ train_data.csv
└─ test_data.csv
```

### Stage 2: Training
```
train_data.csv
    ↓ [torchdiffeq_model.py]
    ├─ CSV → Dicts → Tensors
    ├─ Normalize (mean/std)
    ├─ Create windowed dataset
    ├─ Train NeuralODE model with odeint
    └─ Save: best_model.pt, scalers.pt, config.yml
```

### Stage 3: Evaluation
```
test_data.csv + best_model.pt
    ↓ [evaluate_and_plot.py]
    ├─ Dynamically load model/utils from trainer
    ├─ Load & normalize test data
    ├─ Run inference with various solvers (RK4, dopri5, ...)
    ├─ Select windows (last/random/range modes)
    ├─ Generate plots & metrics
    └─ Save: ts_window_*.png, parity_scatter.png, plot_metrics.json
```

### Stage 4: Export (Optional)
```
best_model.pt
    ├─ [to_onnx.py] → neural_ode_model.onnx
    └─ [to_jit.py] → neural_ode_model_jit.pt (or fallback traces)
```

---

## Configuration File (`config.yml`)

Expected structure:
```yaml
# Feature definitions
observations: [list of observation column names]
disturbances: [list of disturbance column names]  
outdoor: [list of outdoor condition column names]
rooms_temp: [list of room temperature column names]

# Dataset configuration
dataset:
  H: 48              # horizon (window length in timesteps)
  batch_size: 32
  train_range: [0, -500]  # [start_idx, end_idx] for training split
  val_range: [-500, -100]  # validation split
  test_range: [-100, -1]   # test split (can override with --test file)

# Model architecture
model:
  latent_space_dimensions: 16
  epochs: 100
  patience: 10
  warmup: 5

# Threading
num_threads: 8  # or use NUM_THREADS env var

# Output
outdir: "./out"
```

---

## Key Design Decisions

### 1. **Continuous ODE Formulation**
- **Why**: Captures natural thermodynamic dynamics; allows variable-length predictions
- **Trade-off**: More complex than RNNs but better for physical systems

### 2. **Piecewise-Linear Control Interpolation**
- **Why**: ODE solvers need continuous control signals; linear interpolation is efficient
- **Supports**: Batched operations for speed

### 3. **Dynamic Module Loading in evaluate_and_plot.py**
- **Why**: Avoids code duplication; ensures model definition stays centralized
- **Tool**: `runpy.run_path()` executes trainer module and imports its symbols

### 4. **Graceful Degradation in to_jit.py**
- **Why**: TorchScript cannot handle `torchdiffeq.odeint()` easily
- **Solution**: Trace encode/decoder separately when full model fails

### 5. **Flexible Window Selection in Evaluation**
- **Why**: Different analyses need different data subsets (recent behavior, random samples, full range)
- **Options**: `last`, `random`, `range` modes

---

## Common Workflows

### Training from Scratch
```bash
cd thermodynamics_modeling/pythorch_node

# 1. Prepare data
python split_for_test.py

# 2. Train model
python torchdiffeq_model.py [--device cuda] [--seed 42]

# This generates:
# - ./out/best_model.pt (best checkpoint)
# - ./out/scalers.pt (normalization stats)
# - ./out/training_log.txt (optional logging)
```

### Evaluating Trained Model
```bash
# Plot last 500 timesteps
python evaluate_and_plot.py \
  --test ./dataset_split/test_data.csv \
  --out ./out \
  --H 48 \
  --mode last \
  --start_idx -500 \
  --solver rk4

# Or random sampling
python evaluate_and_plot.py \
  --test ./dataset_split/test_data.csv \
  --out ./out \
  --mode random \
  --windows 5 \
  --seed 123
```

### Exporting Model
```bash
# To ONNX (recommended for broad deployment)
python to_onnx.py

# To TorchScript
python to_jit.py  # attempts trace, then script, then fallback traces
```

---

## Dependencies

Core:
- `torch` - Deep learning framework
- `torchdiffeq` - Neural ODE solver integration
- `yaml` - Config file parsing
- `matplotlib` - Visualization
- `pandas` - Data manipulation (split_for_test.py)

Optional:
- `onnx`, `onnxruntime` - For ONNX export/inference
- `tensorboard` - For training monitoring (not in current code but compatible)

---

## Tensor Dimension Conventions

Throughout the codebase:
- **T**: Total time steps in dataset
- **H**: Horizon/window length (constant, configurable)
- **B**: Batch size
- **d_u**: Control/input dimension (observations + disturbances + outdoor)
- **d_y**: Output dimension (number of rooms)
- **latent_dim**: Latent space dimension (configurable)

Common shapes:
- Controls: `[T, d_u]` full time series, `[B, H, d_u]` batch windows, `[H, d_u]` single window
- Targets: `[T, d_y]` full series, `[B, H, d_y]` batch
- Latent: `[B, latent_dim]` initial, `[H, B, latent_dim]` integrated
- Predictions: `[H, B, d_y]` from odeint, reshaped to match targets

---

## Troubleshooting Notes

1. **Model not found**: Ensure `best_model.pt` exists in output directory
2. **CSV parsing errors**: Check delimiter (comma) and timestamp format (%m/%d/%Y %H:%M)
3. **TorchScript export fails**: Use `to_onnx.py` instead or check `to_jit.py` fallback traces
4. **Out of memory**: Reduce batch_size, H, or latent_dim in config.yml
5. **Poor predictions**: Try longer horizon H, more training epochs, or adjust warmup

---

## File Organization

```
pythorch_node/
├── torchdiffeq_model.py          # Main trainer (configurable via config.yml)
├── evaluate_and_plot.py          # Evaluation & visualization
├── split_for_test.py             # Data splitting utility
├── to_onnx.py                    # Export to ONNX
├── to_jit.py                     # Export to TorchScript (with fallbacks)
├── config.yml                    # Configuration
├── dataset_split/                # Auto-created by split_for_test.py
│   ├── train_data.csv
│   └── test_data.csv
├── out/                          # Auto-created by trainer/exporter
│   ├── best_model.pt
│   ├── scalers.pt
│   ├── neural_ode_model.onnx     # If exported
│   ├── neural_ode_model_jit.pt   # If exported
│   ├── ts_window_*.png
│   └── parity_scatter.png
└── README.md (if exists)
```

---

## Summary for LLMs

**What**: Neural ODE model for predicting multi-room building temperatures
**How**: Encodes initial conditions → integrates latent ODE with control inputs → decodes to temperature predictions
**Tools**: PyTorch, torchdiffeq (ODE solver), matplotlib (visualization)
**Data**: Time-series CSV of sensors/actuators from 6-room office
**Stages**: Data split → Train with odeint → Evaluate with plots/metrics → Optional export (ONNX/TorchScript)
**Key Files**: 
- Trainer: `torchdiffeq_model.py`
- Evaluator: `evaluate_and_plot.py`
- Exporters: `to_onnx.py`, `to_jit.py`
- Utils: `split_for_test.py`

When working on this project:
1. Always reference `config.yml` for feature names and hyperparameters
2. Respect tensor dimension conventions for data shapes
3. Use `evaluate_and_plot.py` for post-training analysis
4. For deployment, prefer ONNX unless C++ TorchScript integration is essential
5. The encoder/decoder can be traced separately if full model JIT fails
