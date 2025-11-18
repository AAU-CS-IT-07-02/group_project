import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import neuromancer
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from NODE import build_model, get_colums 

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# =============================
# 1. Load CSV and select columns
# =============================
CSV = config["test_data"]
df = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp')

Y_df, U_df, D_df = get_colums(df)

# =============================
# 2. Load normalization stats from training
# =============================
# Replace these with your saved stats or recompute
muY_vals = np.array([Y_df.mean().values])
stdY_vals = np.array([Y_df.std().replace(0, 1e-6).values])
muU_vals = np.array([U_df.mean().values])
stdU_vals = np.array([U_df.std().replace(0, 1e-6).values])
muD_vals = np.array([D_df.mean().values])
stdD_vals = np.array([D_df.std().replace(0, 1e-6).values])

# Normalize
Y = ((Y_df.values - muY_vals) / stdY_vals).astype(np.float32)
U = ((U_df.values - muU_vals) / stdU_vals).astype(np.float32)
D = ((D_df.values - muD_vals) / stdD_vals).astype(np.float32)

# =============================
# 3. Choose custom window
# =============================
start_idx = -500  # last 500 steps
end_idx = None    # or specify end index
H = 16            # horizon (same as training)

Y_seq = torch.tensor(Y[start_idx:end_idx], dtype=torch.float32).unsqueeze(0)
U_seq = torch.tensor(U[start_idx:end_idx], dtype=torch.float32).unsqueeze(0)
D_seq = torch.tensor(D[start_idx:end_idx], dtype=torch.float32).unsqueeze(0)
xn = Y_seq[:, :1, :]  # initial condition

test_data = {"xn": xn, "Y": Y_seq, "U": U_seq, "D": D_seq, "name": "custom_test"}

ny, nu, nd = Y_seq.shape[-1], U_seq.shape[-1], D_seq.shape[-1]
dt_sec = 5 * 60  # 5-minute sampling

# =============================
# 4. Rebuild model and load weights
# =============================
encode_sym, dynamics_model = build_model(ny, nu, nd, H, dt_sec)

# Create Problem object
y = variable("Y")
yhat = variable("y")
reference_loss = 5.*(yhat == y)^2
onestep_loss = 1.*(yhat[:, 1, :] == y[:, 1, :])^2
loss = PenaltyLoss([reference_loss, onestep_loss], [])
problem = Problem([encode_sym, dynamics_model], loss)

torch.serialization.add_safe_globals([neuromancer.problem.Problem])

problem = torch.load("./out_300/best_model.pth", map_location=torch.device('cpu'), weights_only=False)

problem.nodes[1].nsteps = Y_seq.shape[1]

# =============================
# 5. Run inference
# =============================
with torch.no_grad():
    outputs = problem(test_data)

yhat = outputs['custom_test_y'].cpu().numpy()
true_vals = Y_seq.cpu().numpy()

# Denormalize
pred_traj = (yhat * stdY_vals) + muY_vals
true_traj = (true_vals * stdY_vals) + muY_vals

# =============================
# 6. Compute metrics
# =============================
room_names = config["rooms"]
for i, room in enumerate(room_names):
    pred_flat = pred_traj[:, :, i].reshape(-1)
    true_flat = true_traj[:, :, i].reshape(-1)


# Remove NaNs from both arrays
    mask = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
    true_clean = true_flat[mask]
    pred_clean = pred_flat[mask]

# Compute metrics on cleaned data
    rmse = np.sqrt(((pred_clean - true_clean) ** 2).mean())
    mae = mean_absolute_error(true_clean, pred_clean)
    r2 = r2_score(true_clean, pred_clean)
    print(f"{room} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")



# =============================
# 7. Plot results
# =============================
plt.figure(figsize=(12, 6))
for i, room in enumerate(room_names):
    pred_flat = pred_traj[:,:, i].reshape(-1)
    true_flat = true_traj[:,:, i].reshape(-1)
    print(pred_traj.shape)
    print(true_traj.shape)

    plt.subplot(6, 1, i+1)
    plt.plot(pred_flat, label='True', color='cyan')
    plt.plot(true_flat, label='Predicted', color='magenta', linestyle='--')
    plt.title(f'{room} Temperature')
    plt.xlabel('Time step')
    plt.ylabel('Temperature')
    plt.legend()
plt.tight_layout()
plt.savefig('./out/multi_room_comparison.png')
plt.show()

# Error distribution
plt.figure(figsize=(8, 4))
colors = ['red', 'blue', 'orange', 'grey', 'yellow', 'purple']
for i, room in enumerate(room_names):
    errors = pred_traj[:, :, i].reshape(-1) - true_traj[:, :, i].reshape(-1)

    plt.subplot(6, 1, i+1)
    plt.hist(errors, bins=50, color=colors[i], edgecolor='black')
    plt.title(f'Prediction Error Distribution for {room}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('./out/custom_window_error_distribution.png')
plt.show()

