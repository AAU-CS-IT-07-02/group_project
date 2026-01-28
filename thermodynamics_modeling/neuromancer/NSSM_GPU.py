import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import yaml
from neuromancer import BasicLogger
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset

# Neuromancer imports
from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable
from neuromancer.trainer import Trainer

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Hyperparameters
USE_GPU = True
BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.005
H = 288  # 24-hour prediction horizon
HIDDEN_DIM = 64

# Device selection
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device set to: {device}")

# ------------------------------------------------------------------------------
# 2. Data Processing
# ------------------------------------------------------------------------------
def process_dataframe(df_raw, config):
    """
    Preprocess the raw dataframe: handle missing values and compute derived features.
    Returns matrices for Outputs (Y), Controls (U), and Disturbances (D).
    """
    # Fill missing values
    df_raw = df_raw.ffill().bfill().fillna(0.0)

    # Calculate solar sum
    solar_cols = config.get('outdoor', [])
    solar_data = pd.DataFrame()

    for col in solar_cols:
        if col in df_raw.columns:
            solar_data[col] = df_raw[col]
        else:
            # Handle potential encoding mismatches
            alt = col.replace('façade', 'faÃ§ade')
            if alt in df_raw.columns:
                solar_data[col] = df_raw[alt]
            else:
                alt2 = col.replace('faÃ§ade', 'façade')
                if alt2 in df_raw.columns:
                    solar_data[col] = df_raw[alt2]
                else:
                    solar_data[col] = 0.0

    df_raw['solar_sum'] = solar_data.sum(axis=1)

    # Select columns based on configuration
    Y = df_raw[config['rooms_temp']].values
    U = df_raw[config['observations']].values
    D = df_raw[config['disturbances']].values

    return Y, U, D

def get_stats(Y, U, D):
    """
    Calculate Mean and Standard Deviation for normalization, ignoring NaNs.
    """
    def safe(arr):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std[std == 0] = 1.0 # Prevent divide by zero
        return np.nan_to_num(mean), np.nan_to_num(std)
    return safe(Y), safe(U), safe(D)

def normalize(data, stats):
    return (data - stats[0]) / stats[1]

def create_loader(Y, U, D, batch_size, nsteps, name='train'):
    """
    Create a Neuromancer DictDataset and DataLoader for the given sequences.
    """
    nm_data = {'Y': [], 'U': [], 'D': [], 'xn': []}
    L = len(Y)
    stride = 1

    for i in range(0, L - nsteps, stride):
        nm_data['Y'].append(Y[i:i+nsteps])
        nm_data['U'].append(U[i:i+nsteps])
        nm_data['D'].append(D[i:i+nsteps])
        nm_data['xn'].append(Y[i:i+1]) # Initial condition

    # Convert lists to tensors
    for k in nm_data:
        nm_data[k] = torch.tensor(np.array(nm_data[k]), dtype=torch.float32)

    ds = DictDataset(nm_data, name=name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=ds.collate_fn, num_workers=0)
    return loader, ds

# ------------------------------------------------------------------------------
# 3. Model Definitions
# ------------------------------------------------------------------------------
class DeviceProblem(Problem):
    """
    Custom Problem class that ensures input data is moved to the correct device 
    (GPU/CPU) before execution to prevent device mismatch errors.
    """
    def forward(self, data):
        target_device = next(self.parameters()).device
        gpu_data = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        return super().forward(gpu_data)

class SSM(nn.Module):
    """
    Neural State Space Model (NSSM) architecture.
    Dynamics: x_k+1 = f_x(x) + f_u(u) + f_d(d)
    """
    def __init__(self, fx, fu, fd, nx, nu, nd):
        super().__init__()
        self.fx, self.fu, self.fd = fx, fu, fd
        self.in_features, self.out_features = nx + nu + nd, nx

    def forward(self, x, u, d):
        return self.fx(x) + self.fu(u) + self.fd(d)

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Data Preparation ---
    train_path = os.path.join(BASE_DIR, config['train_data'].replace('./', ''))
    print(f"Processing data from: {train_path}")

    df = pd.read_csv(train_path, parse_dates=['timestamp'])
    Y, U, D = process_dataframe(df, config)

    # Normalization
    stats_Y, stats_U, stats_D = get_stats(Y, U, D)
    Y_n = normalize(Y, stats_Y)
    U_n = normalize(U, stats_U)
    D_n = normalize(D, stats_D)

    # Train/Dev Split (80/20)
    split_idx = int(len(Y_n) * 0.8)

    Y_train, Y_dev = Y_n[:split_idx], Y_n[split_idx:]
    U_train, U_dev = U_n[:split_idx], U_n[split_idx:]
    D_train, D_dev = D_n[:split_idx], D_n[split_idx:]

    # Create DataLoaders
    train_loader, _ = create_loader(Y_train, U_train, D_train, BATCH_SIZE, H, 'train')
    dev_loader, _ = create_loader(Y_dev, U_dev, D_dev, BATCH_SIZE, H, 'dev')

    # Sample batch for logging
    test_data_plot = next(iter(dev_loader))

    # --- Model Initialization ---
    nx = Y.shape[1]
    nu = U.shape[1]
    nd = D.shape[1]

    # Define networks
    fx = blocks.MLP(nx, nx, bias=True, linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU, hsizes=[HIDDEN_DIM, HIDDEN_DIM]).to(device)
    fu = blocks.MLP(nu, nx, bias=True, linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU, hsizes=[HIDDEN_DIM, HIDDEN_DIM]).to(device)
    fd = blocks.MLP(nd, nx, bias=True, linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU, hsizes=[HIDDEN_DIM, HIDDEN_DIM]).to(device)

    ssm_module = SSM(fx, fu, fd, nx, nu, nd).to(device)

    # Build System Graph
    model_node = Node(ssm_module, ['xn', 'U', 'D'], ['xn'], name='NSSM')
    dynamics_model = System([model_node], name='system', nsteps=H-1).to(device)

    # Define Loss (MSE)
    y = variable("Y")[:, 1:, :]
    yhat = variable('xn')[:, 1:, :]
    loss = PenaltyLoss([5. * (yhat == y)^2], [])

    # Compile Problem
    training_problem = DeviceProblem([dynamics_model], loss).to(device)
    optimizer = torch.optim.Adam(training_problem.parameters(), lr=LR)
    logger = BasicLogger(args=None, savedir=config["outdir"], verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    # --- Training ---
    trainer = Trainer(
        training_problem,
        train_loader,
        dev_loader,
        test_data_plot,
        optimizer,
        patience=20,
        epochs=EPOCHS,
        train_metric="train_loss",
        dev_metric="dev_loss",
        eval_metric="dev_loss",
        logger=logger
    )

    print("Starting training loop...")
    best_model_state = trainer.train()

    # --- Save Model Weights ---
    save_path = os.path.join(BASE_DIR, "best_model_weights.pth")

    if isinstance(best_model_state, dict):
        torch.save(best_model_state, save_path)
    else:
        torch.save(best_model_state.state_dict(), save_path)

    print(f"Model weights saved to: {save_path}")

    # --- Evaluation & Plotting (ADDED BACK) ---
    print("Generating evaluation plots...")

    # Load best weights back into the model
    training_problem.load_state_dict(best_model_state)
    training_problem.eval()

    # Prepare data for plotting (move sample batch to GPU)
    plot_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in test_data_plot.items()}

    # Run inference
    with torch.no_grad():
        output = training_problem(plot_batch)

    # Extract predictions and ground truth
    pred_key = 'dev_xn' if 'dev_xn' in output else 'xn'
    pred_traj = output[pred_key].cpu().numpy()  # Shape: (Batch, Time, Dim)
    true_traj = plot_batch['Y'].cpu().numpy()   # Shape: (Batch, Time, Dim)

    # Plot first sample, first room
    sample_idx = 0
    room_idx = 0

    plt.figure(figsize=(12, 6))
    plt.plot(true_traj[sample_idx, :, room_idx], label='True Temp (Norm)', color='black')
    plt.plot(pred_traj[sample_idx, :, room_idx], label='Predicted Temp (Norm)', color='red', linestyle='--')
    plt.title(f"Model Verification (Window {H}, Room {room_idx})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(BASE_DIR, "training_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")