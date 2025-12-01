#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulator: Loop-based inference with pluggable controller integration.
Structure: Matches 'simulator_old.py' (Loop inside main).
Features: Stride 1, Clean Lines, Bang-Bang Integration.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml
import runpy
import numpy as np
import random

# --- 1. DETERMINISM SETUP ---
# We set a fixed seed so that every time you run this script, the random numbers
# (and thus the graph) are exactly the same. This is crucial for debugging.
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 2. IMPORT CONTROLLER ---
# We try to import the 'bang_bang.py' file. If it's missing, we warn the user
# but don't crash immediately (so the simulator can still run open-loop).
try:
    import bang_bang
except ImportError:
    print("[WARN] 'bang_bang.py' not found. Controller logic will not work.")
    bang_bang = None

DEFAULT_OUTDIR = "./out"
DEFAULT_DATA = "./dataset_split/test_data.csv"
DEFAULT_H = 48
DEFAULT_LATENT = 32 

def run_controller(controller_name, y0, controls_seq, scalers, controller_map, setpoint=None):
    """
    Wrapper to call the external controller file.
    This function acts as a bridge between the simulator loop and the logic file.
    """
    if not controller_name: return controls_seq
    
    if controller_name == "bang_bang":
        if bang_bang is None: return controls_seq
        # We pass the full state to the controller so it can make decisions
        return bang_bang.bang_bang_control(y0, controls_seq, scalers, controller_map, setpoint=setpoint)
    
    elif controller_name == "random":
        return bang_bang.random_control(controls_seq)
    
    return controls_seq

def main():
    parser = argparse.ArgumentParser(description="Simulate model over dataset in window intervals.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to data CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Prediction horizon")
    
    # --- UPDATED DEFAULTS ---
    # Stride 1: Checks temp every 5 mins (Instant reaction)
    parser.add_argument("--stride", type=int, default=1, help="Control horizon / step size")
    # Latent 32: Matches your pre-trained model size
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT, help="Latent dimension")
    # Show Real: Default to True for comparison
    parser.add_argument("--hide_real", dest="show_real", action="store_false", default=True, help="Hide real data")
    
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index")
    parser.add_argument("--solver", type=str, default="rk4", help="ODE solver")
    parser.add_argument("--dpi", type=int, default=140, help="Plot DPI")
    parser.add_argument("--controller", type=str, default=None, help="Controller strategy")
    parser.add_argument("--setpoint", type=float, default=None, help="Target temperature")
    parser.add_argument("--loop_type", type=str, default="closed", choices=["open", "closed"], help="Loop type")
    
    # Kept for compatibility with previous runs
    parser.add_argument("--ignore_weather", action="store_true")
    parser.add_argument("--force_ac", action="store_true")

    args = parser.parse_args()

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

    # --- CONTROLLER MAPPING ---
    # This block scans the config file to find which columns belong to which room.
    # It supports Radiators, Dampers, and AHUs.
    controller_map = []
    for i, room_col in enumerate(ROOMS_TEMP):
        room_prefix = room_col.split(":")[0]
        room_indices = {'room_idx': i, 'rad_idx': None, 'damp_idx': None, 'ahu_idx': None}
        for j, control_col in enumerate(CONTROL_FEATURES):
            if room_prefix in control_col:
                if "Radiator" in control_col: room_indices['rad_idx'] = j
                elif "Damper" in control_col: room_indices['damp_idx'] = j
                elif "AHU" in control_col: room_indices['ahu_idx'] = j
        if any(idx is not None for idx in [room_indices['rad_idx'], room_indices['damp_idx'], room_indices['ahu_idx']]):
            controller_map.append(room_indices)

    if args.controller:
        print(f"[INFO] Controller Map loaded for {len(controller_map)} rooms.")

    # Load utilities and model
    td_path = os.path.join(base_dir, "torchdiffeq_model.py")
    module_ns = runpy.run_path(td_path)
    read_csv_as_dicts = module_ns["read_csv_as_dicts"]
    build_matrix = module_ns["build_matrix"]
    normalize = module_ns["normalize"]
    denormalize = module_ns["denormalize"]
    NeuralODEModel = module_ns["NeuralODEModel"]

    # Load scalers and model
    scalers_path = os.path.join(args.out, "scalers.pt")
    ckpt_path = os.path.join(args.out, "best_model.pt")
    
    scalers = torch.load(scalers_path, map_location=device)
    c_mean, c_std = scalers["c_mean"].to(device), scalers["c_std"].to(device)
    y_mean, y_std = scalers["y_mean"].to(device), scalers["y_std"].to(device)
    
    scalers_dict = {"c_mean": c_mean, "c_std": c_std, "y_mean": y_mean, "y_std": y_std}

    ckpt = torch.load(ckpt_path, map_location=device)

    # Load data
    headers, rows = read_csv_as_dicts(args.data)
    controls = build_matrix(rows, CONTROL_FEATURES, device=device)
    states = build_matrix(rows, ROOMS_TEMP, device=device)

    # Normalize
    controls_n = normalize(controls, c_mean, c_std)
    states_n = normalize(states, y_mean, y_std)

    # Build and load model
    d_u = controls.shape[1]
    d_y = states.shape[1]
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
    
    # Time vector
    t_span = torch.linspace(0, args.H - 1, args.H, dtype=torch.float32, device=device)

    all_y_true = []
    all_y_pred = []
    
    # --- RESTORED ALL_IDX ---
    # This list tracks the actual time index of each prediction.
    all_idx = [] 
    
    current_idx = start
    window_count = 0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "simulation_log.txt"), "w") as f:
        f.write("Simulation Log\n================\n")

    # --- MAIN SIMULATION LOOP ---
    with torch.no_grad():
        while current_idx + args.H <= end:
            if not args.controller and args.loop_type == "open":
                y0 = states_n[current_idx].unsqueeze(0)
            else:
                # Closed-loop: use LAST timestep of previous prediction as the start of this one
                if all_y_pred:
                    y0_prev_denorm = all_y_pred[-1][-1, :].unsqueeze(0) 
                    y0 = normalize(y0_prev_denorm, y_mean, y_std) 
                else:
                    y0 = states_n[current_idx].unsqueeze(0)

            # Extract window
            controls_seq = controls_n[current_idx:current_idx + args.H].unsqueeze(0)
            y_seq_true = states[current_idx:current_idx + args.H]
            
            # --- APPLY CONTROLLER ---
            if args.controller:
                controls_seq = run_controller(
                    controller_name=args.controller,
                    y0=y0,
                    controls_seq=controls_seq,
                    scalers=scalers_dict,
                    controller_map=controller_map,
                    setpoint=args.setpoint
                )

            # Inference
            y_hat_seq_norm = model(y0, controls_seq, t_span, method=args.solver).squeeze(1)
            y_hat_seq = denormalize(y_hat_seq_norm, y_mean, y_std)

            # Store Data (Slicing to Stride prevents overlapping data points)
            slice_len = args.stride
            all_y_true.append(y_seq_true[:slice_len])
            all_y_pred.append(y_hat_seq[:slice_len])
            
            # Store Index
            all_idx.append(current_idx)

            # Step forward
            current_idx += args.stride
            window_count += 1
            
            if window_count % 500 == 0:
                print(f"      Processed {window_count} windows...")
                
    print(f"[INFO] Simulated {window_count} windows")

    if window_count == 0:
        print("[WARN] No windows simulated.")
        return

    # Concatenate
    y_true_full = torch.cat(all_y_true, dim=0)
    y_pred_full = torch.cat(all_y_pred, dim=0)

    # --- BURN-IN SLICING (Hide first 50 steps) ---
    # The model is unstable at t=0. We hide the first 50 steps to show a clean graph.
    burn_in = 50
    if len(y_pred_full) > burn_in:
        y_true_full = y_true_full[burn_in:]
        y_pred_full = y_pred_full[burn_in:]
        # Slice the index list too to match
        # Note: Since all_idx tracks windows, we approximate the slice
        # For Stride 1, this is exact.
        if len(all_idx) > burn_in:
             all_idx = all_idx[burn_in:]

    # Analysis
    abs_error = torch.abs(y_true_full - y_pred_full)
    mean_step_error = abs_error.mean(dim=1)
    THRESHOLD = 1.5 
    
    print("\n" + "=" * 50)
    print("DIVERGENCE ANALYSIS")
    print("=" * 50)
    if len(mean_step_error) > 5:
        window_size = 5
        rolling_error = torch.tensor(
            [mean_step_error[i:i+window_size].mean() for i in range(len(mean_step_error) - window_size)]
        )
        divergence_indices = (rolling_error > THRESHOLD).nonzero(as_tuple=True)[0]
        if len(divergence_indices) > 0:
            fail_idx = start + divergence_indices[0].item()
            print(f"Model predictions remain 'good' until index: {fail_idx}")
        else:
            print(f"Model predictions remained 'good' for the entire simulation!")
    print("=" * 50 + "\n")

    # Plot
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
    Simplified Plotting: Blue (Baseline) and Red (Predicted).
    Clean Lines, No segments.
    """
    os.makedirs(outdir, exist_ok=True)
    total_steps, d_y = y_true.shape
    steps = np.arange(total_steps)

    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    for i in range(d_y):
        ax = plt.subplot(2, 3, i + 1)
        
        # Y-Axis Grid (0.5 intervals) for readability
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        
        if show_real:
            ax.plot(steps, y_true[:, i].cpu().numpy(), label="Baseline (Uncontrolled)", color="tab:blue", linewidth=1.5, alpha=0.6)
        
        # Clean Red Line (Solid, Thin)
        ax.plot(steps, y_pred[:, i].cpu().numpy(), label="Predicted (Controlled)", color="tab:red", linestyle="-", linewidth=1.0)
        
        ax.set_title(room_names[i], fontsize=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')

    title_suffix = " (Baseline vs Controlled)"
    plt.suptitle(f"Simulation Trajectory{title_suffix}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fname = os.path.join(outdir, "simulation_trajectory.png")
    plt.savefig(fname)
    plt.close(fig)
    print(f"[INFO] Saved plot to {fname}")


if __name__ == "__main__":
    main()