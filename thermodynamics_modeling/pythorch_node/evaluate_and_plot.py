#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained Neural ODE on test_data.csv and generate plots.

Outputs:
  - Time-series plots (Actual vs Predicted) for each room over selected window(s).
  - Scatter parity plots (Actual vs Predicted with y=x) for each room, aggregated
    across all evaluated windows.

Run examples:
  python evaluate_and_plot.py \
      --test ./dataset_split/test_data.csv \
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
import matplotlib.pyplot as plt
import yaml
import runpy

# -----------------------------
# 1) Default config (align with training)
# -----------------------------
DEFAULT_OUTDIR = "./out"
DEFAULT_TEST = "./dataset_split/test_data.csv"
DEFAULT_H = 48
DEFAULT_LATENT = 16
# Feature lists (controls/targets) are loaded from `config.yml` at runtime.
ROOMS_TEMP = []
CONTROL_FEATURES = []

# The CSV utilities, interpolation and model definitions are provided by the
# training module `torchdiffeq.py` in this package. To avoid duplicating the
# model and utilities here we dynamically load those definitions at runtime
# inside `main()` (using `runpy.run_path`) and then bind the symbols we need
# such as `read_csv_as_dicts`, `build_matrix`, `normalize`, `denormalize` and
# `NeuralODEModel`.

# This keeps the evaluation script small and ensures the model implementation
# is maintained in a single place (the trainer module).
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

    # Load config and bind utilities/model from the training module
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "config.yml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yml not found in {base_dir}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Derive feature lists from config
    CONTROL_FEATURES = list(config.get("observations", [])) + list(config.get("disturbances", [])) + list(config.get("outdoor", []))
    ROOMS_TEMP = list(config.get("rooms_temp", []))

    # Dynamically load the training module's namespace (avoids circular imports)
    td_path = os.path.join(base_dir, "torchdiffeq_model.py")
    if not os.path.exists(td_path):
        raise FileNotFoundError(f"Trainer module not found: {td_path}")
    module_ns = runpy.run_path(td_path)

    # Bind utilities and model from the training module
    try:
        read_csv_as_dicts = module_ns["read_csv_as_dicts"]
        build_matrix = module_ns["build_matrix"]
        normalize = module_ns["normalize"]
        denormalize = module_ns["denormalize"]
        NeuralODEModel = module_ns["NeuralODEModel"]
    except KeyError as e:
        raise RuntimeError(f"Expected symbol missing in trainer module: {e}")

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
