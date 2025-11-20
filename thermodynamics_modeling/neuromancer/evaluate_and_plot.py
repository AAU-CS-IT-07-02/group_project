#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained Neural ODE on test_dataset.csv and generate plots.

Outputs:
  - Time-series plots (Actual vs Predicted) for each room over selected window(s).
  - Scatter parity plots (Actual vs Predicted with y=x) for each room, aggregated
    across all evaluated windows.

Run examples:
  python evaluate_and_plot.py \
      --test ./dataset_split/test_dataset.csv \
      --out ./out \
      --H 16 \
      --mode last \
      --start_idx -500 \
      --solver rk4 \
      --windows 1

Requires:
  - torch
  - torchdiffeq
  - matplotlib
"""

import os
import csv
import json
import math
import random
import argparse

import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# -----------------------------
# 1) Default config (align with training)
# -----------------------------
DEFAULT_OUTDIR = "./out"
DEFAULT_TEST = "./dataset_split/test_dataset.csv"
DEFAULT_H = 16
DEFAULT_LATENT = 8

ROOMS_TEMP = [
    "RoomA:Sensor__room_temperature",
    "RoomB:Sensor__room_temperature",
    "RoomC:Sensor__room_temperature",
    "RoomD:Sensor__room_temperature",
    "RoomE:Sensor__room_temperature",
    "RoomF:Sensor__room_temperature",
]

CONTROL_FEATURES = [
    # Observations per room
    "RoomA:Radiator__control_signal__motor_valve", "RoomA:Damper__position", "RoomA:AHU__active",
    "RoomB:Radiator__control_signal__motor_valve", "RoomB:Damper__position", "RoomB:AHU__active",
    "RoomC:Radiator__control_signal__motor_valve", "RoomC:Damper__position", "RoomC:AHU__active",
    "RoomD:Radiator__control_signal__motor_valve", "RoomD:Damper__position", "RoomD:AHU__active",
    "RoomE:Radiator__control_signal__motor_valve", "RoomE:Damper__position", "RoomE:AHU__active",
    "RoomF:Radiator__control_signal__motor_valve", "RoomF:Damper__position", "RoomF:AHU__active",
    # Global observations
    "Heating:Control__setpoint_water_temperature__supply",
    "Ventilation:Sensor__air_temperature__supply",
    # Disturbances
    "Outdoor:Temperature_air", "solar_sum",
    "RoomA_is_occupied", "RoomA:Window__opened_closed",
    "RoomB_is_occupied", "RoomB:Window__opened_closed",
    "RoomC_is_occupied", "RoomC:Window__opened_closed",
    "RoomD_is_occupied", "RoomD:Window__opened_closed",
    "RoomE_is_occupied", "RoomE:Window__opened_closed",
    "RoomF_is_occupied", "RoomF:Window__opened_closed",
    # Outdoor solar (façades)
    "Outdoor:Solar__direct_radiation__east_façade",
    "Outdoor:Solar__direct_radiation__south_façade",
    "Outdoor:Solar__direct_radiation__west_façade",
]

# -----------------------------
# 2) CSV utilities & normalization
# -----------------------------
def read_csv_as_dicts(path):
    """
    Read CSV rows into a list of dicts: {column_name: float_or_numeric}.
    - Forward-fill missing values per column; first missing -> 0.0
    - Booleans mapped to 1.0/0.0
    - Non-numeric leftovers hashed to a small numeric code
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers = [h.strip().replace("\ufeff", "") for h in headers]
        rows = []
        prev_vals = [None] * len(headers)
        for row in reader:
            row_vals = {}
            for i, val in enumerate(row):
                v = (val or "").strip()
                if v == "" or v.lower() == "nan":
                    v = prev_vals[i] if prev_vals[i] is not None else "0.0"
                try:
                    fv = float(v)
                except ValueError:
                    if v.lower() in ("true", "yes", "on"):
                        fv = 1.0
                    elif v.lower() in ("false", "no", "off"):
                        fv = 0.0
                    else:
                        fv = float(abs(hash(v)) % 10)
                prev_vals[i] = str(fv)
                row_vals[headers[i]] = fv
            rows.append(row_vals)
    return headers, rows

def build_matrix(rows, selected_cols, device):
    """
    Extract matrix [T, D] for selected columns from list of row dicts.
    Missing columns are filled with zeros (warned).
    """
    T = len(rows)
    D = len(selected_cols)
    X = torch.empty(T, D, dtype=torch.float32, device=device)
    missing = []
    for j, col in enumerate(selected_cols):
        for t in range(T):
            if col in rows[t]:
                X[t, j] = float(rows[t][col])
            else:
                missing.append(col)
                X[t, j] = 0.0
    if missing:
        print(f"[WARN] Missing columns not found in CSV: {sorted(set(missing))}")
    return X

def normalize(X, mean, std):
    return (X - mean) / std

def denormalize(Xn, mean, std):
    return Xn * std + mean

# -----------------------------
# 3) Control interpolator
# -----------------------------
class ControlInterpolator:
    """
    Piecewise-linear interpolation of controls U(t) over discrete t_ref.
    U: [B, H, d_u] or [H, d_u] (broadcasted)
    """
    def __init__(self, t_ref, U):
        self.t_ref = t_ref
        self.U = U
        self.H = t_ref.shape[0]
        self.du = U.shape[-1]
        self.has_batch = (U.dim() == 3)

    def __call__(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=self.t_ref.dtype, device=self.t_ref.device)
        pos = torch.searchsorted(self.t_ref, t)
        left = torch.clamp(pos - 1, 0, self.H - 1)
        right = torch.clamp(pos, 0, self.H - 1)
        t_left = self.t_ref[left]
        t_right = self.t_ref[right]
        denom = torch.where((t_right - t_left) == 0, torch.ones_like(t_right), (t_right - t_left))
        alpha = (t - t_left) / denom

        if self.has_batch and t.dim() > 0:
            B = t.shape[0]
            U_left = self.U[torch.arange(B), left]
            U_right = self.U[torch.arange(B), right]
            return (1 - alpha.view(B, 1)) * U_left + alpha.view(B, 1) * U_right
        else:
            U_left = self.U[left] if not self.has_batch else self.U[0, left]
            U_right = self.U[right] if not self.has_batch else self.U[0, right]
            return (1 - alpha) * U_left + alpha * U_right

# -----------------------------
# 4) Neural ODE components (same as training)
# -----------------------------
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, control_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + control_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )
        self.control_interp = None

    def set_control(self, interp):
        self.control_interp = interp

    def forward(self, t, z):
        if self.control_interp is None:
            raise RuntimeError("Control interpolator not set on ODEFunc.")
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]
        if torch.is_tensor(t) and t.dim() == 0:
            t_batch = t.expand(B)
        elif torch.is_tensor(t) and t.dim() == 1:
            t_batch = t
        else:
            t_batch = torch.tensor([t]*B, dtype=z.dtype, device=z.device)
        u_t = self.control_interp(t_batch)                 # [B, d_u]
        return self.net(torch.cat([z, u_t], dim=-1))       # [B, latent_dim]

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x0):
        return self.net(x0)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z_t):
        return self.net(z_t)

class NeuralODEModel(nn.Module):
    def __init__(self, latent_dim, control_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim=output_dim + control_dim, latent_dim=latent_dim)
        self.odefunc = ODEFunc(latent_dim=latent_dim, control_dim=control_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=output_dim)

    def forward(self, y0, controls_seq, t_span, method='rk4'):
        B, H_local, d_u = controls_seq.shape
        assert H_local == t_span.shape[0], "controls_seq length must match t_span length"
        u0 = controls_seq[:, 0, :]
        enc_in = torch.cat([y0, u0], dim=-1)
        z0 = self.encoder(enc_in)

        interp = ControlInterpolator(t_ref=t_span, U=controls_seq)
        self.odefunc.set_control(interp)

        z_t = odeint(self.odefunc, z0, t_span, method=method)  # [H, B, latent]
        Hn, Bn, L = z_t.shape
        y_hat = self.decoder(z_t.reshape(Hn * Bn, L)).view(Hn, Bn, -1)  # [H, B, d_y]
        return y_hat

# -----------------------------
# 5) Metrics
# -----------------------------
def mae_rmse_per_room(y_true, y_pred, room_names):
    """
    y_true, y_pred: [N, H, d_y] (denormalized)
    """
    N, H, d_y = y_true.shape
    out = {}
    for i in range(d_y):
        err = y_pred[:, :, i] - y_true[:, :, i]
        mae = err.abs().mean().item()
        rmse = torch.sqrt((err**2).mean()).item()
        out[room_names[i]] = {"MAE": mae, "RMSE": rmse}
    return out

# -----------------------------
# 6) Plot helpers
# -----------------------------
def plot_time_series(y_true_HDy, y_pred_HDy, outdir, start_idx, dpi=120, room_names=None, title_suffix=""):
    """
    y_true_HDy, y_pred_HDy: [H, d_y] (denormalized)
    """
    os.makedirs(outdir, exist_ok=True)
    H, d_y = y_true_HDy.shape
    if room_names is None:
        room_names = [f"Room {i}" for i in range(d_y)]
    steps = list(range(H))

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    for i in range(d_y):
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(steps, y_true_HDy[:, i].cpu().numpy(), label="Actual", color="tab:blue", linewidth=2)
        ax.plot(steps, y_pred_HDy[:, i].cpu().numpy(), label="Predicted", color="tab:red", linestyle="--", linewidth=2)
        ax.set_title(room_names[i], fontsize=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle(f"Actual vs Predicted (window start={start_idx}){title_suffix}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(outdir, f"ts_window_{start_idx}.png")
    plt.savefig(fname)
    plt.close(fig)
    print(f"[Saved] {fname}")

def plot_parity_scatter(y_true_NHdY, y_pred_NHdY, outdir, dpi=120, room_names=None, title_suffix=""):
    """
    Aggregate scatter: Actual vs Predicted for each room across all evaluated windows.
    y_true_NHdY, y_pred_NHdY: [N, H, d_y] (denormalized)
    """
    os.makedirs(outdir, exist_ok=True)
    N, H, d_y = y_true_NHdY.shape
    if room_names is None:
        room_names = [f"Room {i}" for i in range(d_y)]

    # Flatten across N and H
    y_true = y_true_NHdY.reshape(N * H, d_y).cpu()
    y_pred = y_pred_NHdY.reshape(N * H, d_y).cpu()

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    for i in range(d_y):
        ax = plt.subplot(2, 3, i + 1)
        ax.scatter(y_true[:, i].numpy(), y_pred[:, i].numpy(),
                   s=8, alpha=0.5, color="tab:purple", edgecolors="none")
        # y=x line
        vmin = min(y_true[:, i].min().item(), y_pred[:, i].min().item())
        vmax = max(y_true[:, i].max().item(), y_pred[:, i].max().item())
        ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1, linestyle="--")
        ax.set_title(room_names[i], fontsize=10)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Parity (Actual vs Predicted){title_suffix}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(outdir, f"parity_scatter.png")
    plt.savefig(fname)
    plt.close(fig)
    print(f"[Saved] {fname}")

# -----------------------------
# 7) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot Neural ODE predictions vs actuals.")
    parser.add_argument("--test", type=str, default=DEFAULT_TEST, help="Path to test_dataset.csv")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Horizon length")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT, help="Latent dimension (match training)")
    parser.add_argument("--solver", type=str, default="rk4", help="ODE solver (rk4, dopri5, ...)")
    parser.add_argument("--mode", type=str, choices=["last", "random", "range"], default="last",
                        help="Window selection mode for time-series plots")
    parser.add_argument("--start_idx", type=int, default=-500, help="Start index for 'last'/'range'. Negative counts from end.")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive) for 'range'")
    parser.add_argument("--windows", type=int, default=1, help="Number of windows to plot in 'random' or to cap in 'range'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 'random' mode")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load scalers and model weights
    scalers_path = os.path.join(args.out, "scalers.pt")
    ckpt_path = os.path.join(args.out, "best_model.pt")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    scalers = torch.load(scalers_path, map_location=device)
    c_mean, c_std = scalers["c_mean"].to(device), scalers["c_std"].to(device)
    y_mean, y_std = scalers["y_mean"].to(device), scalers["y_std"].to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Load test CSV
    headers, rows = read_csv_as_dicts(args.test)
    controls = build_matrix(rows, CONTROL_FEATURES, device=device)  # [T, d_u]
    targets = build_matrix(rows, ROOMS_TEMP, device=device)         # [T, d_y]

    # Normalize
    controls_n = normalize(controls, c_mean, c_std)
    targets_n = normalize(targets, y_mean, y_std)

    # Rebuild model and load weights
    d_u = controls.shape[1]
    d_y = targets.shape[1]
    model = NeuralODEModel(latent_dim=args.latent_dim, control_dim=d_u, output_dim=d_y)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Time vector
    H = args.H
    t_span = torch.linspace(0, H - 1, H, dtype=torch.float32, device=device)

    # Select windows for *time-series* plots
    T = controls.shape[0]
    starts = []

    if args.mode == "last":
        tail_start = 0 if args.start_idx is None else (T + args.start_idx if args.start_idx < 0 else args.start_idx)
        tail_start = max(0, tail_start)
        region = list(range(tail_start, max(tail_start, T - H + 1)))
        starts = [region[-1]] if region else [max(0, T - H)]
    elif args.mode == "random":
        random.seed(args.seed)
        all_starts = list(range(0, max(0, T - H + 1)))
        # If a negative start_idx is specified, restrict to tail
        if args.start_idx is not None and args.start_idx < 0:
            tail_start = max(0, T + args.start_idx)
            all_starts = list(range(tail_start, max(tail_start, T - H + 1)))
        if len(all_starts) == 0:
            all_starts = [max(0, T - H)]
        starts = random.sample(all_starts, k=min(args.windows, len(all_starts)))
    elif args.mode == "range":
        s = 0 if args.start_idx is None else (T + args.start_idx if args.start_idx < 0 else args.start_idx)
        e = T if args.end_idx is None else args.end_idx
        s = max(0, s); e = min(T, e)
        starts = list(range(s, max(s, e - H + 1)))
        if args.windows is not None and args.windows > 0:
            starts = starts[:args.windows]

    if len(starts) == 0:
        starts = [max(0, T - H)]

    print(f"Time-series windows to plot: {starts}")

    # --- Generate time-series plots ---
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for start in starts:
            controls_seq = controls_n[start:start + H].unsqueeze(0)  # [1, H, d_u]
            y_seq_true = targets[start:start + H]                     # [H, d_y] (denorm)
            y0 = targets_n[start].unsqueeze(0)                        # [1, d_y]

            y_hat_seq_norm = model(y0, controls_seq, t_span, method=args.solver).squeeze(1)  # [H, d_y]
            y_hat_seq = denormalize(y_hat_seq_norm, y_mean, y_std)                           # [H, d_y]

            # Save for parity plot aggregation
            y_true_all.append(y_seq_true)
            y_pred_all.append(y_hat_seq)

            # Per-window time-series figure
            plot_time_series(y_seq_true, y_hat_seq,
                             outdir=args.out, start_idx=start, dpi=args.dpi,
                             room_names=ROOMS_TEMP,
                             title_suffix=f" | solver={args.solver}")

    # --- Parity scatter over *all* windows in the test slice ---
    # If you prefer to aggregate over the entire test set (all windows),
    # you can rebuild starts = range(0, T-H+1) and repeat the loop above.
    y_true_all = torch.stack(y_true_all, dim=0)  # [N, H, d_y]
    y_pred_all = torch.stack(y_pred_all, dim=0)  # [N, H, d_y]
    plot_parity_scatter(y_true_all, y_pred_all,
                        outdir=args.out, dpi=args.dpi,
                        room_names=ROOMS_TEMP,
                        title_suffix=f" | {len(starts)} window(s) | solver={args.solver}")

    # Optional: save quick metrics for reference
    metrics = {}
    for i, room in enumerate(ROOMS_TEMP):
        err = (y_pred_all[:, :, i] - y_true_all[:, :, i])
        mae = err.abs().mean().item()
        rmse = torch.sqrt((err**2).mean()).item()
        metrics[room] = {"MAE": mae, "RMSE": rmse}

    with open(os.path.join(args.out, "plot_metrics.json"), "w") as f:
        json.dump({"windows": starts, "H": H, "metrics": metrics}, f, indent=2)
    print(f"[Saved] {os.path.join(args.out, 'plot_metrics.json')}")
    print("Done.")

if __name__ == "__main__":
    main()
