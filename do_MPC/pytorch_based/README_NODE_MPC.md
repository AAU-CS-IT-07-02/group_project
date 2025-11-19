# NODE-based MPC with do-mpc

This directory contains an implementation of Model Predictive Control (MPC) using a trained Neural ODE (NODE) model as the surrogate dynamics.

## Files

- `main_node_mpc.py`: The main script to run the closed-loop simulation.
- `node_adapter.py`: Contains the model architecture definitions, weight loading logic, and conversion to CasADi.
- `inspect_weights.py`: Helper script to inspect the keys in the trained model checkpoint.

## Prerequisites

- `do-mpc`
- `casadi`
- `torch`
- `numpy`
- `matplotlib`

## Usage

1. Ensure the trained model weights are available at the path specified in `main_node_mpc.py` (variable `weights_path`).
   - Default path: `../../thermodynamics_modeling/neuromancer/out_300/best_model_state_dict.pth`
2. Run the main script:
   ```bash
   python main_node_mpc.py
   ```

## Implementation Details

- **Model Architecture**: The surrogate model consists of a Dynamics Network (MLP with Tanh activations) and a Decoder Network (MLP with ReLU activations).
- **Dynamics**: The dynamics are modeled as a latent ODE. The MPC uses an explicit RK4 integration step (symbolically defined in CasADi) to predict the next latent state.
- **Objective**: The MPC minimizes the error between the decoded output (room temperatures) and a setpoint (currently 0.0, assuming normalized data).
- **Disturbances**: The model expects disturbances (weather, occupancy). In this example, they are set to zero (mean values) for the prediction horizon.

## Adaptation

To use your own trained model:
1. Update `DynamicsNet` and `DecoderNet` in `node_adapter.py` to match your architecture.
2. Update the key mapping in `load_models` in `node_adapter.py` if your state dict keys differ.
3. Adjust `nx`, `nu`, `nd`, `ny` dimensions in `node_adapter.py`.
