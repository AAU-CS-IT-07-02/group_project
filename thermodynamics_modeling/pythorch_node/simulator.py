#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulator: Loop-based inference with optional controller integration and plotting.

Simulates the model over a dataset in window intervals, with optional real data overlay.

Run example:
  python simulator.py \
      --data ./dataset_split/test_data.csv \
      --out ./out \
      --H 16 \
      --start_idx 0 \
      --end_idx -1 \
      --show_real \
      --solver rk4
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import runpy


DEFAULT_OUTDIR = "./out"
DEFAULT_DATA = "./dataset_split/test_data.csv"
DEFAULT_H = 48
DEFAULT_LATENT = 16


def main():
    parser = argparse.ArgumentParser(description="Simulate model over dataset in window intervals.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to data CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Prediction horizon")
    parser.add_argument("--stride", type=int, default=None, help="Control horizon / step size (default: H)")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT, help="Latent dimension")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index in dataset (-1 = end)")
    parser.add_argument("--show_real", action="store_true", help="Overlay real data on plots")
    parser.add_argument("--solver", type=str, default="rk4", help="ODE solver method")
    parser.add_argument("--dpi", type=int, default=140, help="Plot DPI")
    args = parser.parse_args()

    # Default stride to H if not specified
    if args.stride is None:
        args.stride = args.H

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load config
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "config.yml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yml not found in {base_dir}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    CONTROL_FEATURES = list(config.get("observations", [])) + list(config.get("disturbances", [])) + list(config.get("outdoor", []))
    ROOMS_TEMP = list(config.get("rooms_temp", []))

    # Load utilities and model from trainer
    td_path = os.path.join(base_dir, "torchdiffeq_model.py")
    if not os.path.exists(td_path):
        raise FileNotFoundError(f"Trainer module not found: {td_path}")
    module_ns = runpy.run_path(td_path)

    read_csv_as_dicts = module_ns["read_csv_as_dicts"]
    build_matrix = module_ns["build_matrix"]
    normalize = module_ns["normalize"]
    denormalize = module_ns["denormalize"]
    NeuralODEModel = module_ns["NeuralODEModel"]

    # Load scalers and model
    scalers_path = os.path.join(args.out, "scalers.pt")
    ckpt_path = os.path.join(args.out, "best_model.pt")
    if not os.path.exists(scalers_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Scalers or checkpoint not found in {args.out}")

    scalers = torch.load(scalers_path, map_location=device)
    c_mean, c_std = scalers["c_mean"].to(device), scalers["c_std"].to(device)
    y_mean, y_std = scalers["y_mean"].to(device), scalers["y_std"].to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Load data
    headers, rows = read_csv_as_dicts(args.data)
    controls = build_matrix(rows, CONTROL_FEATURES, device=device)
    targets = build_matrix(rows, ROOMS_TEMP, device=device)

    # Normalize
    controls_n = normalize(controls, c_mean, c_std)
    targets_n = normalize(targets, y_mean, y_std)

    # Build and load model
    d_u = controls.shape[1]
    d_y = targets.shape[1]
    model = NeuralODEModel(latent_dim=args.latent_dim, control_dim=d_u, output_dim=d_y)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Determine simulation range
    T = controls.shape[0]
    start = 0 if args.start_idx is None else (max(0, T + args.start_idx) if args.start_idx < 0 else args.start_idx)
    end = T if args.end_idx is None or args.end_idx == -1 else (max(0, T + args.end_idx) if args.end_idx < 0 else args.end_idx)
    start = max(0, start)
    end = min(T, end)

    print(f"[INFO] Simulating from index {start} to {end} (total {end - start} steps)")
    print(f"[INFO] Prediction horizon H={args.H}, Control horizon (stride)={args.stride}")

    # Time vector for ODE
    t_span = torch.linspace(0, args.H - 1, args.H, dtype=torch.float32, device=device)

    # Simulation loop
    all_y_true = []
    all_y_pred = []
    all_idx = []

    current_idx = start
    window_count = 0

    with torch.no_grad():
        while current_idx + args.H <= end:
            # Extract window
            controls_seq = controls_n[current_idx:current_idx + args.H].unsqueeze(0)  # [1, H, d_u]
            y_seq_true = targets[current_idx:current_idx + args.H]                     # [H, d_y]
            y0 = targets_n[current_idx].unsqueeze(0)                                  # [1, d_y]

            # Inference
            y_hat_seq_norm = model(y0, controls_seq, t_span, method=args.solver).squeeze(1)  # [H, d_y]
            y_hat_seq = denormalize(y_hat_seq_norm, y_mean, y_std)                           # [H, d_y]

            # Store
            all_y_true.append(y_seq_true)
            all_y_pred.append(y_hat_seq)
            all_idx.append(current_idx)

            # Step forward by stride (control horizon)
            current_idx += args.stride
            window_count += 1

    print(f"[INFO] Simulated {window_count} windows")

    # Concatenate all predictions
    y_true_full = torch.cat(all_y_true, dim=0)  # [total_steps, d_y]
    y_pred_full = torch.cat(all_y_pred, dim=0)  # [total_steps, d_y]

    # Plot full trajectory
    plot_simulation(y_true_full, y_pred_full, args.out, ROOMS_TEMP, args.show_real, args.dpi, args.solver)

    # Save metrics
    metrics = {}
    for i, room in enumerate(ROOMS_TEMP):
        err = y_pred_full[:, i] - y_true_full[:, i]
        mae = err.abs().mean().item()
        rmse = torch.sqrt((err**2).mean()).item()
        metrics[room] = {"MAE": mae, "RMSE": rmse}

    metrics_path = os.path.join(args.out, "simulator_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"windows": window_count, "H": args.H, "stride": args.stride, "metrics": metrics}, f, indent=2)
    print(f"[INFO] Saved metrics to {metrics_path}")

    print("[INFO] Done.")


def plot_simulation(y_true, y_pred, outdir, room_names, show_real, dpi, solver):
    """
    Plot full simulation trajectory.
    y_true, y_pred: [total_steps, d_y]
    """
    os.makedirs(outdir, exist_ok=True)
    total_steps, d_y = y_true.shape

    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    for i in range(d_y):
        ax = plt.subplot(2, 3, i + 1)
        steps = list(range(total_steps))
        
        if show_real:
            ax.plot(steps, y_true[:, i].cpu().numpy(), label="Real", color="tab:blue", linewidth=2, alpha=0.7)
        ax.plot(steps, y_pred[:, i].cpu().numpy(), label="Predicted", color="tab:red", linestyle="--", linewidth=2)
        
        ax.set_title(room_names[i], fontsize=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    title_suffix = " (with real)" if show_real else " (model only)"
    plt.suptitle(f"Simulation Trajectory{title_suffix} | solver={solver}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fname = os.path.join(outdir, "simulation_trajectory.png")
    plt.savefig(fname)
    plt.close(fig)
    print(f"[INFO] Saved plot to {fname}")


if __name__ == "__main__":
    main()
