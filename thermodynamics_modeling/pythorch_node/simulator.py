#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulator: Loop-based inference with pluggable controller integration.
Visualization: Clean Multi-Colored Lines.
Features: 
  - Tri-Actuator Support (Radiator, Damper, AHU).
  - Merged logic from teammate's AHU discovery.
  - Corrects defaults (Stride=1, Latent=32).

Run example:
  python simulator.py --controller bang_bang --setpoint 21.5
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import runpy
import numpy as np
import random

# --- 1. Determinism ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 2. Import Controller ---
try:
    import bang_bang
except ImportError:
    print("[WARN] 'bang_bang.py' not found. Controller logic will not work.")
    bang_bang = None

DEFAULT_OUTDIR = "./out"
DEFAULT_DATA = "./dataset_split/test_data.csv"
DEFAULT_H = 48
DEFAULT_LATENT = 32 

def run_controller_dispatcher(controller_name, y0, controls_seq, t_span, scalers, controller_map, setpoint=None):
    """Dispatcher: Decides which controller logic to apply."""
    if not controller_name:
        return controls_seq
        
    if controller_name == "bang_bang":
        if bang_bang is None:
            print("[WARN] bang_bang.py is missing.")
            return controls_seq
        
        # Use CLI setpoint if provided
        if setpoint is not None:
            return bang_bang.bang_bang_control(y0, controls_seq, scalers, controller_map, setpoint=setpoint)
        else:
            return bang_bang.bang_bang_control(y0, controls_seq, scalers, controller_map)
    
    elif controller_name == "random":
        if bang_bang is None: 
            return controls_seq
        return bang_bang.random_control(controls_seq)
    
    else:
        print(f"[WARN] Controller '{controller_name}' not implemented. Using default.")
        return controls_seq


def run_simulation_loop(model, controls_n, states_n, states_denorm, args, scalers, controller_map, device, normalize_fn, denormalize_fn, control_feature_names, use_controller=False):
    """
    Runs the simulation loop. Handles disturbance blocking if requested.
    """
    T = controls_n.shape[0]
    start = max(0, args.start_idx)
    end = min(T, args.end_idx) if args.end_idx != -1 else T
    
    t_span = torch.linspace(0, args.H - 1, args.H, dtype=torch.float32, device=device)

    all_y_pred = []
    all_controls = []
    
    current_idx = start
    window_count = 0
    
    y0 = states_n[current_idx].unsqueeze(0)

    total_steps = (end - start) // args.stride
    print(f"   -> Steps to simulate: {total_steps}")

    with torch.no_grad():
        while current_idx + args.H <= end:
            # Closed Loop Logic
            if all_y_pred:
                y0_prev_denorm = all_y_pred[-1][-1, :].unsqueeze(0) 
                y0 = normalize_fn(y0_prev_denorm, scalers['y_mean'], scalers['y_std']) 
            else:
                y0 = states_n[current_idx].unsqueeze(0)

            # Extract Window
            controls_seq = controls_n[current_idx:current_idx + args.H].unsqueeze(0)
            

            # Controller Logic
            if use_controller and args.controller:
                controls_seq = run_controller_dispatcher(
                    controller_name=args.controller, 
                    y0=y0, 
                    controls_seq=controls_seq, 
                    t_span=t_span, 
                    scalers=scalers, 
                    controller_map=controller_map,
                    setpoint=args.setpoint
                )
            
            all_controls.append(controls_seq.squeeze(0)[:args.stride])

            # Inference
            y_hat_seq_norm = model(y0, controls_seq, t_span, method=args.solver).squeeze(1)
            y_hat_seq = denormalize_fn(y_hat_seq_norm, scalers['y_mean'], scalers['y_std'])

            slice_len = args.stride
            all_y_pred.append(y_hat_seq[:slice_len])

            current_idx += args.stride
            window_count += 1
            
            if window_count % 500 == 0:
                print(f"      Processed {window_count} windows...")

    if len(all_y_pred) == 0:
        return None, None

    y_pred_full = torch.cat(all_y_pred, dim=0)
    controls_full = torch.cat(all_controls, dim=0)
    
    return y_pred_full, controls_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to data CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--H", type=int, default=DEFAULT_H, help="Prediction horizon")
    parser.add_argument("--stride", type=int, default=1, help="Control horizon / step size")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT, help="Latent dimension")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index")
    parser.add_argument("--hide_real", dest="show_real", action="store_false", default=True, help="Hide real data")
    parser.add_argument("--solver", type=str, default="rk4", help="ODE solver method")
    parser.add_argument("--dpi", type=int, default=140, help="Plot DPI")
    parser.add_argument("--controller", type=str, default=None, help="Controller strategy")
    parser.add_argument("--setpoint", type=float, default=None, help="Target temperature")
    parser.add_argument("--loop_type", type=str, default="closed", choices=["open", "closed"], help="Loop type")
    parser.add_argument("--ignore_weather", action="store_true", help="Block Solar/Outdoor data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "config.yml")
    with open(cfg_path, "r") as f: config = yaml.safe_load(f)

    CONTROL_FEATURES = list(config.get("observations", [])) + list(config.get("disturbances", [])) + list(config.get("outdoor", []))
    ROOMS_TEMP = list(config.get("rooms_temp", []))

    # --- UPDATED MAPPING (Includes AHU) ---
    controller_map = []
    for i, room_col in enumerate(ROOMS_TEMP):
        room_prefix = room_col.split(":")[0]
        # Map supports 3 types of actuators now
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

    # Load modules
    td_path = os.path.join(base_dir, "torchdiffeq_model.py")
    module_ns = runpy.run_path(td_path)
    read_csv_as_dicts = module_ns["read_csv_as_dicts"]
    build_matrix = module_ns["build_matrix"]
    normalize = module_ns["normalize"]
    denormalize = module_ns["denormalize"]
    NeuralODEModel = module_ns["NeuralODEModel"]

    # Load Data
    scalers = torch.load(os.path.join(args.out, "scalers.pt"), map_location=device)
    ckpt = torch.load(os.path.join(args.out, "best_model.pt"), map_location=device)
    
    scalers_dict = {
        "c_mean": scalers["c_mean"].to(device), "c_std": scalers["c_std"].to(device),
        "y_mean": scalers["y_mean"].to(device), "y_std": scalers["y_std"].to(device)
    }

    headers, rows = read_csv_as_dicts(args.data)
    controls = build_matrix(rows, CONTROL_FEATURES, device=device)
    states = build_matrix(rows, ROOMS_TEMP, device=device)
    
    controls_n = normalize(controls, scalers_dict['c_mean'], scalers_dict['c_std'])
    states_n = normalize(states, scalers_dict['y_mean'], scalers_dict['y_std'])

    d_u = controls.shape[1]
    d_y = states.shape[1]
    model = NeuralODEModel(latent_dim=args.latent_dim, control_dim=d_u, output_dim=d_y)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # --- RUNS ---
    print("\n[INFO] 1/2: Running Baseline Simulation (Original Controls)...")
    y_pred_base, _ = run_simulation_loop(
        model, controls_n, states_n, states, args, scalers_dict, controller_map, device, 
        normalize, denormalize, CONTROL_FEATURES, use_controller=False
    )

    print("\n[INFO] 2/2: Running Controlled Simulation (Active Controller)...")
    y_pred_ctrl, controls_ctrl = run_simulation_loop(
        model, controls_n, states_n, states, args, scalers_dict, controller_map, device, 
        normalize, denormalize, CONTROL_FEATURES, use_controller=True
    )

    if y_pred_base is None or y_pred_ctrl is None:
        print("[ERR] Simulation produced no data. Check indices.")
        return

    print(f"\n[INFO] Plotting comparisons...")
    plot_simulation(
        y_true=y_pred_base, 
        y_pred=y_pred_ctrl, 
        outdir=args.out, 
        room_names=ROOMS_TEMP, 
        show_real=True, 
        dpi=args.dpi, 
        solver=args.solver, 
        controls=controls_ctrl, 
        controller_map=controller_map
    )

    # Metrics
    metrics = {}
    for i, room in enumerate(ROOMS_TEMP):
        diff = y_pred_ctrl[:, i] - y_pred_base[:, i]
        avg_diff = diff.mean().item()
        metrics[room] = {"Avg_Temp_Change": avg_diff}

    with open(os.path.join(args.out, "simulator_metrics.json"), "w") as f:
        json.dump({"windows": len(y_pred_ctrl), "H": args.H, "metrics": metrics}, f, indent=2)
    print("[INFO] Done.")


def plot_simulation(y_true, y_pred, outdir, room_names, show_real, dpi, solver, controls=None, controller_map=None):
    os.makedirs(outdir, exist_ok=True)
    total_steps, d_y = y_true.shape
    steps = np.arange(total_steps)

    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    
    valve_indices = {}
    if controls is not None and controller_map is not None:
        for item in controller_map:
            if item['rad_idx'] is not None:
                valve_indices[item['room_idx']] = item['rad_idx']

    for i in range(d_y):
        ax = plt.subplot(2, 3, i + 1)
        
        if show_real:
            ax.plot(steps, y_true[:, i].cpu().numpy(), label="Baseline (Uncontrolled)", color="tab:blue", linewidth=1.5, alpha=0.6)
        
        y_pred_np = y_pred[:, i].cpu().numpy()
        
        if controls is not None and i in valve_indices:
            valve_idx = valve_indices[i]
            valve_signal = controls[:, valve_idx].cpu().numpy()
            threshold = (np.max(valve_signal) + np.min(valve_signal)) / 2
            
            on_mask = valve_signal > threshold
            off_mask = valve_signal <= threshold
            
            y_heating = y_pred_np.copy()
            y_cooling = y_pred_np.copy()
            y_heating[~on_mask] = np.nan
            y_cooling[~off_mask] = np.nan
            
            ax.plot(steps, y_pred_np, color='tab:red', linestyle='--', linewidth=1.0, alpha=0.3, label="Predicted")
            ax.plot(steps, y_cooling, color='firebrick', linestyle='-', linewidth=2.0, label="Mode: COOLING")
            ax.plot(steps, y_heating, color='limegreen', linestyle='-', linewidth=2.0, label="Mode: HEATING")
            
        else:
            ax.plot(steps, y_pred_np, label="Predicted", color="tab:red", linestyle="--", linewidth=2)

        ax.set_title(room_names[i], fontsize=10)
        if i >= 3: ax.set_xlabel("Time step")
        if i % 3 == 0: ax.set_ylabel("Temperature (°C)")
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