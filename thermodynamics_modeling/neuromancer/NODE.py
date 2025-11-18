"""
Neuromancer NODE example adapted for AAU-BUILD dataset
"""

# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import neuromancer as nm
from neuromancer.dataset import DictDataset
from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.loggers import BasicLogger

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# ----------------------------
# System / performance setup
# - allow controlling DataLoader workers and CPU threading
# - read optional overrides from `config.yml` under `system`
# ----------------------------
system_cfg = config.get("system", {}) if isinstance(config, dict) else {}
# sensible defaults
cpu_count = os.cpu_count() or 1
DEFAULT_NUM_WORKERS = system_cfg.get("num_workers", max(0, cpu_count // 2))
DEFAULT_NUM_THREADS = system_cfg.get("num_threads", max(1, cpu_count))

# Set environment variables for common BLAS/OpenMP backends
os.environ.setdefault("OMP_NUM_THREADS", str(DEFAULT_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(DEFAULT_NUM_THREADS))

# Tell PyTorch how many intra-op threads to use (CPU only)
try:
    torch.set_num_threads(int(DEFAULT_NUM_THREADS))
    torch.set_num_interop_threads(max(1, int(DEFAULT_NUM_THREADS // 2)))
except Exception:
    # If torch isn't available at import time or doesn't support these calls,
    # just continue without crashing.
    pass

def get_colums(df):
    # === Combine solar columns ===
    df["solar_sum"] = df[config['outdoor']].sum(axis=1)

    # === Select subsets ===
    Y_df = df[config['rooms_temp']]
    U_df = df[config['observations']]
    D_df = df[config['disturbances']]

    return Y_df, U_df, D_df

# =======================================================
# 1. Data utilities (get_data, get_splits)
# =======================================================
def get_data(csv_path, dt_minutes=5, H=12):
    """Load, preprocess, normalize, and window building dataset.

    Args:
        csv_path (str): Path to the CSV file containing building data.
        dt_minutes (int): Resampling interval in minutes.
        H (int): Sequence length (horizon).

    Returns:
        dict: Dictionary containing sequences (xn, Y, U, D), timestep in seconds, and normalization stats.
    """    
    dt_sec = dt_minutes * 60.0

    # === Load and preprocess ===
    df = (pd.read_csv(csv_path, parse_dates=['timestamp'])
            .set_index('timestamp')
            .sort_index()
            .resample(f'{dt_minutes}min').mean()
            .interpolate(limit_direction='both'))

    Y_df, U_df, D_df = get_colums(df)

    # === Normalization ===
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
    
    print("There are Nans? ")
    print(np.isnan(Y).sum(), np.isnan(U).sum(), np.isnan(D).sum())
    print("There are Inf? ")
    print(np.isinf(Y).sum(), np.isinf(U).sum(), np.isinf(D).sum())


    # === Build sequence windows ===
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
        "xn": Xn,
        "Y": Ys,
        "U": Us,
        "D": Ds,
        "dt_sec": dt_sec,
        "stats": {"Y": (muY, stdY), "U": (muU, stdU), "D": (muD, stdD)}
    }


def get_splits(csv_path, dt_minutes=5, H=12, batch_size=64, split_ratio=0.5, num_workers=None):
    """Create train, dev, and test splits for the dataset.

    Args:
        csv_path (str): Path to the CSV file.
        dt_minutes (int): Resampling interval in minutes.
        H (int): Sequence length.
        batch_size (int): Batch size for DataLoader.
        split_ratio (float): Ratio for train/dev split.

    Returns:
        tuple: (train_loader, dev_loader, test_data, dt_sec, stats)
    """    
    data = get_data(csv_path, dt_minutes, H)
    Xn, Ys, Us, Ds = data["xn"], data["Y"], data["U"], data["D"]

    split = int(split_ratio * len(Ys))
    train = {"xn": Xn[:split], "Y": Ys[:split], "U": Us[:split], "D": Ds[:split]}
    dev   = {"xn": Xn[split:], "Y": Ys[split:], "U": Us[split:], "D": Ds[split:]}

    train_ds = DictDataset(train, name="train")
    dev_ds   = DictDataset(dev, name="dev")

    # Determine num_workers: prefer argument, then config/system default, then safe fallback
    if num_workers is None:
        num_workers = system_cfg.get("num_workers", DEFAULT_NUM_WORKERS)

    # For Windows, num_workers > 0 will spawn subprocesses for the DataLoader
    persistent = True if num_workers and num_workers > 0 else False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=int(num_workers),
        persistent_workers=persistent,
        pin_memory=False,
        prefetch_factor=2,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dev_ds.collate_fn,
        num_workers=int(max(0, int(num_workers)//2)),
        persistent_workers=(True if num_workers and num_workers > 1 else False),
        pin_memory=False,
        prefetch_factor=2,
    )

    test_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in dev.items()}
    test_data["name"] = "test"

    return train_loader, dev_loader, test_data, data["dt_sec"], data["stats"]


# =======================================================
# 2. Model definition
# =======================================================
def build_model(ny, nu, nd, H, dt_sec):
    """Construct NODE-based model including encoder, dynamics, and decoder.

    Args:
        ny (int): Number of output variables.
        nu (int): Number of input variables.
        nd (int): Number of disturbance variables.
        H (int): Prediction horizon.
        dt_sec (float): Integration timestep in seconds.

    Returns:
        tuple: (encode_sym, dynamics_model)
    """    
    torch.manual_seed(0)

    n_latent = 4  # latent state space dimension
    nx = n_latent

    # Latent encoder (observations -> latent initial state)
    encoder = blocks.MLP(ny, nx, bias=True,
                         linear_map=torch.nn.Linear,
                         nonlin=torch.nn.ReLU,
                         hsizes=[40])
    encode_sym = Node(encoder, ['Y'], ['xn'], name='encoder')

    # Latent ODE model (continuous-time)
    fx = blocks.MLP(nx + nu + nd, nx, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.Tanh,
                    hsizes=[40, 40])

    # Integrate ODE using RK4 with timestep = dt_sec
    fxRK4 = integrators.RK4(fx, h=dt_sec)
    model = Node(fxRK4, ['xn', 'U', 'D'], ['xn'], name='NODE')

    # Decoder (latent -> physical output)
    decoder = blocks.MLP(nx, ny, bias=True,
                         linear_map=torch.nn.Linear,
                         nonlin=torch.nn.ReLU,
                         hsizes=[40])
    decode_sym = Node(decoder, ['xn'], ['y'], name='decoder')

    # Full rollout system
    dynamics_model = System([model, decode_sym], name='system', nsteps=H)

    return encode_sym, dynamics_model


# =======================================================
# 3. Training setup
# =======================================================
def train_model(train_loader, dev_loader, test_data, encode_sym, dynamics_model):
    """Train Neuromancer NODE model using provided data loaders.

    Args:
        train_loader (DataLoader): Training data loader.
        dev_loader (DataLoader): Development data loader.
        test_data (dict): Test dataset.
        encode_sym (Node): Encoder node.
        dynamics_model (System): NODE dynamics system.

    Returns:
        Problem: Trained Neuromancer problem instance.
    """    

    # %% Constraints + losses:
    y = variable("Y")                      # observed
    yhat = variable('y')                   # predicted output

    # trajectory tracking loss
    reference_loss = 5.*(yhat == y)^2
    reference_loss.name = "ref_loss"

    # one step tracking loss
    onestep_loss = 1.*(yhat[:, 1, :] == y[:, 1, :])^2
    onestep_loss.name = "onestep_loss"


    # Assemble problem
    nodes = [encode_sym, dynamics_model]
    objectives = [reference_loss, onestep_loss]
    constraints = []

    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(nodes, loss)
    # problem.show()
    
    os.makedirs(config["outdir"], exist_ok=True)
    # Optimizer and trainer
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.003)
    logger = BasicLogger(args=None, savedir=config["outdir"], verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=config["model"]["patience"],
        warmup=config["model"]["warmup"],
        epochs=config["model"]["epochs"],
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
        logger=logger,
    )

    best_model = trainer.train()
    problem.load_state_dict(best_model)
    return problem


# =======================================================
# 4. Main script
# =======================================================
if __name__ == "__main__":
    CSV = config["train_data"]

    # === Load dataset and splits ===
    H = 16       
    batch_size = 64
    train_loader, dev_loader, test_data, dt_sec, stats = get_splits(
        CSV, dt_minutes=5, H=H, batch_size=batch_size
    )
    print(f"Integration step (dt_sec): {dt_sec}")

    # === Infer dimensions ===
    sample_batch = next(iter(train_loader))
    ny = sample_batch["Y"].shape[-1]
    nu = sample_batch["U"].shape[-1]
    nd = sample_batch["D"].shape[-1]

    # === Build model ===
    encode_sym, dynamics_model = build_model(ny, nu, nd, H, dt_sec)

    # === Train ===
    problem = train_model(train_loader, dev_loader, test_data,
                                  encode_sym, dynamics_model)
    
    # =====================
    # Parameter estimation results
    # =====================

    # Ensure rollout uses full test sequence length
    problem.nodes[1].nsteps = test_data['Y'].shape[1]

    # Run model in evaluation mode
    with torch.no_grad():
        test_outputs = problem(test_data)

    # --- Helper for denormalization
    def denormalize(x, mean, std):
        return (x * std) + mean

    # --- Extract normalization stats (already defined earlier in your script)
    muY_vals = stats["Y"][0].values  # mean
    stdY_vals = stats["Y"][1].values  # std
    muU_vals = stats["U"][0].values
    stdU_vals = stats["U"][1].values
    muD_vals = stats["D"][0].values
    stdD_vals = stats["D"][1].values


    # --- Extract arrays
    yhat = test_outputs['test_y'].detach().cpu().numpy()
    Y_test = test_data['Y']
    U_test = test_data['U']
    D_test = test_data['D']

    # --- Determine dimensions dynamically
    ny = Y_test.shape[-1]
    nu = U_test.shape[-1]
    nd = D_test.shape[-1]

    # --- Denormalize trajectories
    pred_traj = denormalize(yhat, muY_vals, stdY_vals).reshape(-1, ny).T
    true_traj = denormalize(Y_test, muY_vals, stdY_vals).reshape(-1, ny).T
    input_traj = denormalize(U_test, muU_vals, stdU_vals).reshape(-1, nu).T
    dist_traj = denormalize(D_test, muD_vals, stdD_vals).reshape(-1, nd).T

    # --- Plot rollout comparison
    plt_nsteps = min(500, true_traj.shape[1])
    figsize = 12
    fig, ax = plt.subplots(ny + nu + nd, figsize=(figsize, figsize))

    x_labels = [f'$y_{k}$' for k in range(ny)]
    for row, (t_true, t_pred, label) in enumerate(zip(true_traj, pred_traj, x_labels)):
        axe = ax[row]
        axe.plot(t_true[:plt_nsteps], 'c', linewidth=2.0, label='True output')
        axe.plot(t_pred[:plt_nsteps], 'm--', linewidth=2.0, label='Predicted output')
        axe.set_ylabel(label, rotation=0, labelpad=15, fontsize=10)
        axe.legend(fontsize=8)
        axe.tick_params(labelbottom=False, labelsize=9)

    # --- Inputs
    u_labels = [f'$u_{k}$' for k in range(nu)]
    for row, (u, label) in enumerate(zip(input_traj, u_labels)):
        axe = ax[row + ny]
        axe.plot(u[:plt_nsteps], linewidth=2.0, label='Input')
        axe.set_ylabel(label, rotation=0, labelpad=15, fontsize=10)
        axe.legend(fontsize=8)
        axe.tick_params(labelbottom=False, labelsize=9)

    # --- Disturbances
    d_labels = [f'$d_{k}$' for k in range(nd)]
    for row, (d, label) in enumerate(zip(dist_traj, d_labels)):
        axe = ax[row + ny + nu]
        axe.plot(d[:plt_nsteps], linewidth=2.0, label='Disturbance')
        axe.set_ylabel(label, rotation=0, labelpad=15, fontsize=10)
        axe.legend(fontsize=8)
        axe.tick_params(labelbottom=True, labelsize=9)

    ax[-1].set_xlabel('Time step', fontsize=10)
    plt.tight_layout()
    plt.show()
