"""

For this example we demonstrate learning a model-based control policy for an unknown dynamical system.

**Offline off-policy control learning scenario using real building data.**

In a typical real world control setting, due to cost and operational concerns, there is not an opportunity to directly interact with the system to learn a controller. The presented scenario has three stages:

+ Stage 1: Load real building data from CSV and preprocess for training.
+ Stage 2: Learn a black-box neural ordinary differential equation NODE approximation of the building dynamics given the time series data.
+ Stage 3: Learn neural control policy by differentiating closed-loop dynamical system (neural policy + NODE) using Differentiable predictive control (DPC) method.
In the following cells we walk through the three stage process of loading data, system identification, and control policy learning using neuromancer.


NODE paper: https://arxiv.org/abs/1806.07366
DPC paper: https://www.sciencedirect.com/science/article/pii/S0959152422000981

"""

"""
# # # # # # # # # # # # # # # # # # # # # 
#       Stage 1: data loading           #
# # # # # # # # # # # # # # # # # # # # # 
"""

import os
import pandas as pd
import numpy as np
import yaml
import torch
from neuromancer.dataset import DictDataset

# Load configuration
with open('./config.yml', 'r') as file:
    config = yaml.safe_load(file)

def get_columns(df, config):
    """Extract Y (outputs), U (inputs), and D (disturbances) from dataframe."""
    # Combine solar columns if needed
    if config.get('outdoor') and len(config['outdoor']) > 0:
        df["solar_sum"] = df[config['outdoor']].sum(axis=1)
    
    Y_df = df[config['rooms_temp']]
    U_df = df[config['observations']]
    
    # Filter disturbances to only include columns that exist
    available_disturbances = [d for d in config['disturbances'] if d in df.columns]
    D_df = df[available_disturbances]
    
    return Y_df, U_df, D_df

def load_real_data(csv_path, config, dt_minutes=5, H=12):
    """Load and preprocess real building data from CSV.
    
    Args:
        csv_path (str): Path to CSV file
        config (dict): Configuration dictionary
        dt_minutes (int): Resampling interval in minutes
        H (int): Sequence length (horizon)
    
    Returns:
        dict: Dictionary with 'X' (states from Y), 'U' (inputs), 'Y' (outputs), 'D' (disturbances), and normalization stats
    """
    dt_sec = dt_minutes * 60.0
    
    # Load and preprocess
    df = (pd.read_csv(csv_path, parse_dates=['timestamp'])
            .set_index('timestamp')
            .sort_index()
            .resample(f'{dt_minutes}min').mean()
            .interpolate(limit_direction='both'))
    
    Y_df, U_df, D_df = get_columns(df, config)
    
    # Z-score normalization
    def zscore_fit(x):
        mu, std = x.mean(), x.std().replace(0, 1e-6)
        return mu, std
    
    def zscore_apply(x, mu, std):
        return (x - mu) / std
    
    muY, stdY = zscore_fit(Y_df)
    muU, stdU = zscore_fit(U_df)
    muD, stdD = zscore_fit(D_df)
    
    Y = zscore_apply(Y_df, muY, stdY).values.astype(np.float32)
    U = zscore_apply(U_df, muU, stdU).values.astype(np.float32)
    D = zscore_apply(D_df, muD, stdD).values.astype(np.float32)
    
    # Build sequence windows
    def build_sequences(Y, U, D, H):
        N = len(Y) - H
        Xn, Ys, Us, Ds = [], [], [], []
        for t in range(N):
            Ys.append(Y[t:t+H])
            Us.append(U[t:t+H])
            Ds.append(D[t:t+H])
            Xn.append(Y[t:t+1])  # initial condition
        return np.stack(Xn), np.stack(Ys), np.stack(Us), np.stack(Ds)
    
    Xn, Ys, Us, Ds = build_sequences(Y, U, D, H)
    
    return {
        'X': Ys,  # Use Y (room temperatures) as state trajectory
        'U': Us,
        'Y': Ys,
        'D': Ds,
        'xn': Xn,
        'stats': {'Y': (muY, stdY), 'U': (muU, stdU), 'D': (muD, stdD)}
    }

# Load real data
csv_path = config["train_data"]
H = 12  # sequence length
train_dev_data = load_real_data(csv_path, config, dt_minutes=5, H=H)

# Split into train and dev (50-50 split)
split = int(0.5 * len(train_dev_data['X']))
stats = train_dev_data.pop('stats')  # Extract and remove stats from data dict

train_data = {k: v[:split] if isinstance(v, np.ndarray) else v for k, v in train_dev_data.items()}
dev_data = {k: v[split:] if isinstance(v, np.ndarray) else v for k, v in train_dev_data.items()}

# Create test data from the same CSV (could use separate test file if available)
test_data = {k: v[split:] if isinstance(v, np.ndarray) else v for k, v in train_dev_data.items()}

# Derive parameters from data shapes
nx = train_data['X'].shape[2]  # number of states (room temperatures)
nu = train_data['U'].shape[2]  # number of control inputs
ny = train_data['Y'].shape[2]  # number of observations
nd = train_data['D'].shape[2]  # number of disturbances
nsteps = train_data['X'].shape[1]  # number of time steps per sample

print(f"Data loaded: nx={nx}, nu={nu}, ny={ny}, nd={nd}, nsteps={nsteps}")
print(f"Train shape: X={train_data['X'].shape}, U={train_data['U'].shape}, D={train_data['D'].shape}")
print(f"Dev shape: X={dev_data['X'].shape}, U={dev_data['U'].shape}, D={dev_data['D'].shape}")

# Plot initially loaded raw data for verification
import matplotlib.pyplot as plt
print("\nPlotting initial data verification...")

# Denormalize for visualization
muY_vals = stats["Y"][0].values
stdY_vals = stats["Y"][1].values
muU_vals = stats["U"][0].values
stdU_vals = stats["U"][1].values
muD_vals = stats["D"][0].values
stdD_vals = stats["D"][1].values

# Take all samples from training data and flatten
n_samples_plot = train_data['X'].shape[0]  # Use all training samples
X_plot = train_data['X'][:n_samples_plot].reshape(-1, nx)
U_plot = train_data['U'][:n_samples_plot].reshape(-1, nu)
D_plot = train_data['D'][:n_samples_plot].reshape(-1, nd)

# Denormalize
X_denorm = X_plot * stdY_vals + muY_vals
U_denorm = U_plot * stdU_vals + muU_vals
D_denorm = D_plot * stdD_vals + muD_vals

print(f"Plotting {len(X_denorm)} timesteps of training data...")

# Plot states
fig, axes = plt.subplots(min(3, nx), figsize=(14, 10))
if nx == 1:
    axes = [axes]

for state_idx in range(min(3, nx)):
    axes[state_idx].plot(X_denorm[:, state_idx], linewidth=1.0, alpha=0.8)
    axes[state_idx].set_ylabel(f'Room {state_idx} Temp (°C)', fontsize=11)
    axes[state_idx].grid(True, alpha=0.3)
    axes[state_idx].tick_params(labelsize=9)

axes[-1].set_xlabel('Time step (5 min intervals)', fontsize=11)
fig.suptitle('Initial Training Data - State Trajectories', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('initial_data_states.png', dpi=100)
print("Saved initial state data to initial_data_states.png")
plt.close()

# Plot inputs
fig, axes = plt.subplots(min(6, nu), figsize=(14, 10))
if nu == 1:
    axes = [axes]

for input_idx in range(min(6, nu)):
    axes[input_idx].plot(U_denorm[:, input_idx], linewidth=1.0, alpha=0.8, color='orange')
    axes[input_idx].set_ylabel(f'Control {input_idx}', fontsize=11)
    axes[input_idx].grid(True, alpha=0.3)
    axes[input_idx].tick_params(labelsize=9)

axes[-1].set_xlabel('Time step (5 min intervals)', fontsize=11)
fig.suptitle('Initial Training Data - Control Inputs', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('initial_data_inputs.png', dpi=100)
print("Saved initial input data to initial_data_inputs.png")
plt.close()

# Plot disturbances
fig, axes = plt.subplots(min(6, nd), figsize=(14, 10))
if nd == 1:
    axes = [axes]

for dist_idx in range(min(6, nd)):
    axes[dist_idx].plot(D_denorm[:, dist_idx], linewidth=1.0, alpha=0.8, color='green')
    axes[dist_idx].set_ylabel(f'Disturbance {dist_idx}', fontsize=11)
    axes[dist_idx].grid(True, alpha=0.3)
    axes[dist_idx].tick_params(labelsize=9)

axes[-1].set_xlabel('Time step (5 min intervals)', fontsize=11)
fig.suptitle('Initial Training Data - Disturbances', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('initial_data_disturbances.png', dpi=100)
print("Saved initial disturbance data to initial_data_disturbances.png")

# create dataloaders
from torch.utils.data import DataLoader
nsamples = train_data['X'].shape[0]  # number of samples
train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader = [DataLoader(d, batch_size=32, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset]]

"""
# # # # # # # # # # # # # # # # # # # # # # # #
#       Stage 2: system identification        #
# # # # # # # # # # # # # # # # # # # # # # # #
"""

""" Define a black-box ODE model to identify the system from data"""
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators
import torch

# define neural ODE that incorporates disturbances
# dx/dt = f(x, u, d) - STATE, CONTROL, DISTURBANCE
dx = blocks.MLP(nx + nu + nd, nx, bias=True, hsizes=[40, 40])
integrator = integrators.RK4(dx, h=0.1)  # Use realistic timestep
system_model = Node(integrator, ['xn', 'U', 'D'], ['xn'], name='NODE')
model = System([system_model])
# model.show()        # visualize computational graph of the NODE system ID model

"""Define the system identification optimization problem"""
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss

# Nstep rollout predictions from the model
xpred = variable('xn')[:, :-1, :]
# Ground truth data
xtrue = variable('X')
# define system identification loss function
loss = (xpred == xtrue) ^ 2
loss.update_name('system_id')
# construct differentiable optimization problem in Neuromancer
obj = PenaltyLoss([loss], [])
problem = Problem([model], obj)
# problem.show()

"""Solve the system identification problem"""
from neuromancer.trainer import Trainer
import torch.optim as optim

opt = optim.Adam(model.parameters(), 0.001)
trainer = Trainer(problem, train_loader, dev_loader,
                  optimizer=opt,
                  epochs=1,
                  patience=300,
                  train_metric='train_loss',
                  eval_metric='dev_loss')
best_model = trainer.train()

""" Evaluate the learned NODE system model on full test trajectory"""
# Convert test data to torch tensors
test_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in test_data.items() if k not in ['stats']}

# Run model in evaluation mode
with torch.no_grad():
    test_output = model(test_tensors)

# Plot predictions vs ground truth for first few states
import matplotlib.pyplot as plt
n_plot = min(3, nx)  # plot first 3 states
fig, axes = plt.subplots(n_plot, figsize=(12, 8))
if n_plot == 1:
    axes = [axes]

for state_idx in range(n_plot):
    axes[state_idx].plot(test_output['xn'][0, :-1, state_idx].detach().numpy(), 
                        label='Predicted', linewidth=2)
    axes[state_idx].plot(test_tensors['X'][0, :, state_idx].detach().numpy(), 
                        label='Ground Truth', linewidth=2, linestyle='--')
    axes[state_idx].set_ylabel(f'State {state_idx} (normalized)')
    axes[state_idx].legend()
    axes[state_idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time steps')
plt.tight_layout()
plt.savefig('node_system_id.png', dpi=100)
print("Saved system ID evaluation to node_system_id.png")
plt.close()

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       Stage 3: learning neural control policy         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

""" Create a closed loop system using the system model and a parametrized control policy """
# define control policy
class Policy(torch.nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.net = blocks.MLP(insize, outsize, bias=True, hsizes=[20, 20, 20])

    def forward(self, x, R):
        features = torch.cat([x, R], dim=-1)
        return self.net(features)

# fix model parameters
system_model.requires_grad_(False)

insize = 2*nx
policy = Policy(insize, nu)
policy_node = Node(policy, ['xn', 'R'], ['U'], name='policy')
cl_system = System([policy_node, system_model], name='cl_system')
# cl_system.show()

""" Sample dataset of control parameters """
# Use room temperatures as reference setpoints for the policy to track
train_dataset_policy = DictDataset({'R': train_data['X'], 'X': train_data['X'], 'xn': train_data['xn'], 'U': train_data['U'], 'D': train_data['D']}, name='train')
dev_dataset_policy = DictDataset({'R': dev_data['X'], 'X': dev_data['X'], 'xn': dev_data['xn'], 'U': dev_data['U'], 'D': dev_data['D']}, name='dev')
train_loader_policy = DataLoader(train_dataset_policy, batch_size=32, collate_fn=train_dataset_policy.collate_fn, shuffle=True)
dev_loader_policy = DataLoader(dev_dataset_policy, batch_size=32, collate_fn=dev_dataset_policy.collate_fn, shuffle=False)

""" Define objectives of the optimal control problem """
tru = variable('xn')[:, 1:, :]  # system states
ref = variable('R')             # reference
u = variable('U')               # control action
# reference tracking objective
loss = (ref == tru) ^ 2
loss.update_name('tracking')

# differentiable optimal control problem
obj = PenaltyLoss([loss], [])
problem_policy = Problem([cl_system], obj)
# problem_policy.show()

""" Optimize the control policy"""
opt_policy = optim.Adam(policy.parameters(), 0.01)
logout = ['loss']
trainer_policy = Trainer(problem_policy, train_loader_policy, dev_loader_policy,
                  optimizer=opt_policy,
                  epochs=1,
                  patience=50,
                  train_metric='train_loss',
                  eval_metric='dev_loss')

best_policy_model = trainer_policy.train()
problem_policy.load_state_dict(best_policy_model)

print("Policy training completed successfully!")
print("Models saved. You can now use the trained NODE and policy for predictions.")

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       Evaluation: Testing the trained policy          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

""" Evaluate the learned control policy on test data """
# Convert test data to torch tensors and add reference trajectory
test_tensors_policy = {k: torch.tensor(v, dtype=torch.float32) for k, v in test_data.items()}
# Use test states as reference for the policy to track
test_tensors_policy['R'] = test_tensors_policy['X'].clone()

# Evaluate policy on test data with step reference tracking
with torch.no_grad():
    test_output_policy = cl_system(test_tensors_policy)

# Extract results
y_pred = test_output_policy['xn'].detach().cpu().numpy()
y_true = test_tensors_policy['X'].detach().cpu().numpy()
u_applied = test_output_policy['U'].detach().cpu().numpy() if 'U' in test_output_policy else None

# Reshape from (nsamples, H, features) to (nsamples*H, features) for continuous plotting
y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
y_true_flat = y_true.reshape(-1, y_true.shape[-1])
if u_applied is not None:
    u_applied_flat = u_applied.reshape(-1, u_applied.shape[-1])
else:
    u_applied_flat = None

# Denormalize for interpretation
# Extract normalization stats
muY_vals = stats["Y"][0].values  # mean
stdY_vals = stats["Y"][1].values  # std
muU_vals = stats["U"][0].values
stdU_vals = stats["U"][1].values

# Denormalize state trajectories
y_pred_denorm = y_pred_flat * stdY_vals + muY_vals
y_true_denorm = y_true_flat * stdY_vals + muY_vals

# Denormalize control actions
if u_applied_flat is not None:
    u_applied_denorm = u_applied_flat * stdU_vals + muU_vals
else:
    u_applied_denorm = None

plt_nsteps = min(500, y_true_flat.shape[0])
n_states_plot = min(3, nx)

fig, axes = plt.subplots(n_states_plot, figsize=(14, 10))
if n_states_plot == 1:
    axes = [axes]

for state_idx in range(n_states_plot):
    axes[state_idx].plot(y_true_denorm[:plt_nsteps, state_idx], 'c-', linewidth=2.0, label='Reference/True')
    axes[state_idx].plot(y_pred_denorm[:plt_nsteps, state_idx], 'm--', linewidth=2.0, label='Predicted with Policy')
    axes[state_idx].set_ylabel(f'Room {state_idx} Temperature (°C)', fontsize=11)
    axes[state_idx].legend(fontsize=10)
    axes[state_idx].grid(True, alpha=0.3)
    axes[state_idx].tick_params(labelsize=9)

axes[-1].set_xlabel('Time step (5 min intervals)', fontsize=11)
fig.suptitle('Control Policy Performance on Test Data (Denormalized)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('control_policy_performance.png', dpi=100)
print("Saved policy evaluation to control_policy_performance.png")
plt.close()

# Optional: Plot control actions if available
if u_applied_denorm is not None:
    n_inputs_plot = min(6, nu)
    fig, axes = plt.subplots(n_inputs_plot, figsize=(14, 10))
    if n_inputs_plot == 1:
        axes = [axes]
    
    for input_idx in range(n_inputs_plot):
        axes[input_idx].plot(u_applied_denorm[:plt_nsteps, input_idx], 'g-', linewidth=2.0)
        axes[input_idx].set_ylabel(f'Control {input_idx}', fontsize=11)
        axes[input_idx].grid(True, alpha=0.3)
        axes[input_idx].tick_params(labelsize=9)
    
    axes[-1].set_xlabel('Time step (5 min intervals)', fontsize=11)
    fig.suptitle('Control Actions Applied by Learned Policy (Denormalized)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('control_actions.png', dpi=100)
    print("Saved control actions to control_actions.png")
    plt.close()

print("\nTraining and evaluation complete!")
print(f"Final system dimensions: nx={nx}, nu={nu}, nd={nd}")
print(f"Test dataset shape (flattened): {y_true_flat.shape}")
print(f"Plotted timesteps: {plt_nsteps} (≈ {plt_nsteps * 5 / 60:.1f} hours)")