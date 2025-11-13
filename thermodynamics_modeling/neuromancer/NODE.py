"""
Neuromancer NODE example adapted for AAU-BUILD dataset
"""

# ============================
# Imports
# ============================
import pandas as pd
import numpy as np
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


# =======================================================
# 1. Data utilities (get_data, get_splits)
# =======================================================
def get_data(csv_path, dt_minutes=5, H=12):
    """Load, process, normalize, and window the building dataset."""
    dt_sec = dt_minutes * 60.0

    # === Load and preprocess ===
    df = (pd.read_csv(csv_path, parse_dates=['timestamp'])
            .set_index('timestamp')
            .sort_index()
            .resample(f'{dt_minutes}min').mean()
            .interpolate(limit_direction='both'))

    # === Combine solar columns ===
    df["solar_sum"] = df[[
        "Outdoor:Solar__direct_radiation__east_façade",
        "Outdoor:Solar__direct_radiation__south_façade",
        "Outdoor:Solar__direct_radiation__west_façade"
    ]].sum(axis=1)

    # === Select subsets ===
    Y_df = df[["RoomA:Sensor__room_temperature"]]
    U_df = df[[
        "RoomA:Radiator__control_signal__motor_valve",
        "RoomA:Damper__position",
        "RoomA:AHU__active"
    ]]
    D_df = df[[
        "Outdoor:Temperature_air", "solar_sum",
        "RoomA_is_occupied", "RoomA:Window__opened_closed"
    ]]

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


def get_splits(csv_path, dt_minutes=5, H=12, batch_size=64, split_ratio=0.8):
    """Create train/dev/test loaders."""
    data = get_data(csv_path, dt_minutes, H)
    Xn, Ys, Us, Ds = data["xn"], data["Y"], data["U"], data["D"]

    split = int(split_ratio * len(Ys))
    train = {"xn": Xn[:split], "Y": Ys[:split], "U": Us[:split], "D": Ds[:split]}
    dev   = {"xn": Xn[split:], "Y": Ys[split:], "U": Us[split:], "D": Ds[split:]}

    train_ds = DictDataset(train, name="train")
    dev_ds   = DictDataset(dev, name="dev")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    dev_loader   = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=dev_ds.collate_fn)

    test_data = dev.copy()
    test_data["name"] = "test"

    return train_loader, dev_loader, test_data, data["dt_sec"], data["stats"]


# =======================================================
# 2. Model definition
# =======================================================
def build_model(ny, nu, nd, H, dt_sec):
    """Construct NODE-based model (encoder, dynamics, decoder)."""
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
    """Train Neuromancer NODE model."""
    

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

    # Optimizer and trainer
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.003)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=100,
        warmup=500,
        epochs=1000,
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
    CSV = "../../AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv"

    # === Load dataset and splits ===
    H = 12          # sequence length (1 hour horizon)
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
    muY_vals = stats.muY.values
    stdY_vals = stats.stdY.values
    muU_vals = stats.muU.values
    stdU_vals = stats.stdU.values
    muD_vals = stats.muD.values
    stdD_vals = stats.stdD.values

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
