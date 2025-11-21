#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test a trained Neural ODE model on test_dataset.csv (no plotting).

This script:
  - Loads saved scalers (c_mean/c_std, y_mean/y_std) from ./out/scalers.pt
  - Loads model weights from ./out/best_model.pt
  - Reads test_dataset.csv using Python's csv module
  - Normalizes controls/targets using training scalers
  - Rebuilds the same Neural ODE architecture
  - Evaluates all sliding windows of length H in the selected test range
  - Prints and saves per-room and aggregate MAE/RMSE

Run examples:
  python test_neural_ode.py --test ./dataset_split/test_dataset.csv --out ./out --H 16
  python test_neural_ode.py --start_idx -500 --solver rk4
  python test_neural_ode.py --mode range --start_idx 100 --end_idx 1000

Dependencies:
  - torch
  - torchdiffeq
"""

import os
import csv
import json
import argparse

import torch
import torch.nn as nn
from torchdiffeq import odeint

# -----------------------------
# 1) Default configuration (align with your training config)
# -----------------------------
DEFAULT_OUTDIR = "./out"
DEFAULT_TEST = "./dataset_split/test_dataset.csv"  # <-- requested filename
DEFAULT_H = 16

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
]

# -----------------------------
# 2) CSV utilities & normalization
# -----------------------------
def read_csv_as_dicts(path):
    """
    Read CSV rows into a list of dicts: {header: float_value}.
    - Forward-fill per column; first missing -> 0.0
    - Booleans to 1.0/0.0
    - Non-numeric residuals hashed to small numeric codes
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
    Missing columns are filled with zeros; a warning is printed.
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
    Piecewise-linear interpolation of controls U(t) over t_ref=[0..H-1].
    U can be [B, H, d_u] (batch) or [H, d_u] (single sample).
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
        denom = torch.where((t_right - t_left) == 0,
                            torch.ones_like(t_right),
                            (t_right - t_left))
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
            nn.Linear(128, latent_dim),
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
            nn.Linear(128, latent_dim),
        )
    def forward(self, x0):
        return self.net(x0)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
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
        """
        y0: [B, d_y] (normalized)
        controls_seq: [B, H, d_u] (normalized)
        t_span: [H] times (e.g., 0..H-1)
        """
        B, H_local, d_u = controls_seq.shape
        assert H_local == t_span.shape[0], "controls_seq length must match t_span length"
        u0 = controls_seq[:, 0, :]
        enc_in = torch.cat([y0, u0], dim=-1)
        z0 = self.encoder(enc_in)

        interp = ControlInterpolator(t_ref=t_span, U=controls_seq)
        self.odefunc.set_control(interp)

        z_t = odeint(self.odefunc, z0, t_span, method=method)   # [H, B, latent]
        Hn, Bn, L = z_t.shape
        y_hat = self.decoder(z_t.reshape(Hn * Bn, L)).view(Hn, Bn, -1)  # [H, B, d_y]
        return y_hat

# -----------------------------
# 5) Metrics
# -----------------------------
def mae_rmse_over_windows(y_true_seq, y_pred_seq):
    """
    y_true_seq, y_pred_seq: [N_windows, H, d_y] in physical units (denormalized)
    Return per-room MAE/RMSE and aggregate.
    """
    assert y_true_seq.shape == y_pred_seq.shape
    N, H, d_y = y_true_seq.shape

    per_room = {}
    for i in range(d_y):
        err = y_pred_seq[:, :, i] - y_true_seq[:, :, i]    # [N, H]
        mae = err.abs().mean().item()
        rmse = torch.sqrt((err ** 2).mean()).item()
        per_room[ROOMS_TEMP[i]] = {"MAE": mae, "RMSE": rmse}

    # Aggregate across rooms
    err_all = y_pred_seq - y_true_seq                     # [N, H, d_y]
    mae_all = err_all.abs().mean().item()
    rmse_all = torch.sqrt((err_all ** 2).mean()).item()
    aggregate = {"MAE": mae_all, "RMSE": rmse_all}
    return per_room, aggregate

# -----------------------------
# 6) Main: load, evaluate, report
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Neural ODE model on test_dataset.csv (no plots).")
    parser.add_argument("--test", type=str, default=DEFAULT_TEST, help="Path to test_dataset.csv")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Horizon length")
    parser.add_argument("--solver", type=str, default="rk4", help="ODE solver method")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent space dimensionality (must match training)")
    parser.add_argument("--mode", type=str, choices=["all", "last", "range"], default="all",
                        help="Which windows to evaluate: all (default), last, or range")
    parser.add_argument("--start_idx", type=int, default=-500,
                        help="For 'last'/'range': starting index (negative counts from end)")
    parser.add_argument("--end_idx", type=int, default=None, help="For 'range': end index (exclusive)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load scalers and checkpoint
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

    # Read test CSV
    headers, rows = read_csv_as_dicts(args.test)
    controls = build_matrix(rows, CONTROL_FEATURES, device=device)  # [T, d_u]
    targets = build_matrix(rows, ROOMS_TEMP, device=device)         # [T, d_y]

    # Normalize using training scalers
    controls_n = normalize(controls, c_mean, c_std)     # [T, d_u]
    targets_n = normalize(targets, y_mean, y_std)       # [T, d_y]

    # Rebuild model and load weights
    d_u = controls.shape[1]
    d_y = targets.shape[1]
    model = NeuralODEModel(latent_dim=args.latent_dim, control_dim=d_u, output_dim=d_y)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Time span (unit steps)
    H = args.H
    t_span = torch.linspace(0, H - 1, H, dtype=torch.float32, device=device)

    # Define window start indices based on mode
    T = controls.shape[0]
    if args.mode == "all":
        starts = list(range(0, max(0, T - H + 1)))
    elif args.mode == "last":
        tail_start = 0 if args.start_idx is None else (T + args.start_idx if args.start_idx < 0 else args.start_idx)
        tail_start = max(0, tail_start)
        region = list(range(tail_start, max(tail_start, T - H + 1)))
        starts = [region[-1]] if region else [max(0, T - H)]
    elif args.mode == "range":
        s = 0 if args.start_idx is None else (T + args.start_idx if args.start_idx < 0 else args.start_idx)
        e = T if args.end_idx is None else args.end_idx
        s = max(0, s)
        e = min(T, e)
        starts = list(range(s, max(s, e - H + 1)))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if len(starts) == 0:
        starts = [max(0, T - H)]

    # Evaluate windows
    y_true_seq = []
    y_pred_seq = []
    with torch.no_grad():
        for start in starts:
            controls_seq = controls_n[start:start + H].unsqueeze(0)   # [1, H, d_u]
            y_seq_true = targets[start:start + H]                      # [H, d_y] (denormalized)
            y0 = targets_n[start].unsqueeze(0)                         # [1, d_y]

            y_hat_seq_norm = model(y0, controls_seq, t_span, method=args.solver).squeeze(1)  # [H, d_y]
            y_hat_seq = denormalize(y_hat_seq_norm, y_mean, y_std)                           # [H, d_y]

            y_true_seq.append(y_seq_true)
            y_pred_seq.append(y_hat_seq)

    y_true_seq = torch.stack(y_true_seq, dim=0)   # [N, H, d_y]
    y_pred_seq = torch.stack(y_pred_seq, dim=0)   # [N, H, d_y]

    # Metrics
    per_room, aggregate = mae_rmse_over_windows(y_true_seq, y_pred_seq)

    print("\n=== Test Results ===")
    for room, vals in per_room.items():
        print(f"{room:40s}  MAE={vals['MAE']:8.4f}  RMSE={vals['RMSE']:8.4f}")
    print(f"\nAggregate over rooms/windows:  MAE={aggregate['MAE']:8.4f}  RMSE={aggregate['RMSE']:8.4f}")

    # Save metrics JSON
    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"per_room": per_room, "aggregate": aggregate,
                   "H": H, "N_windows": len(starts), "mode": args.mode,
                   "start_indices": starts}, f, indent=2)
    print(f"\n[Saved] {metrics_path}")
    print("Done.")

if __name__ == "__main__":
    main()
