#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot multiple simulation runs together for comparison.

Loads CSV results from multiple simulator runs and plots them on the same figure
using the same style as simulator.py.

Usage example:
  python plot_multiple_runs.py \
      --runs ./out_run1/simulation_results.csv ./out_run2/simulation_results.csv \
      --labels "Run 1" "Run 2" \
      --out ./comparison_plot \
      --show_real

Or with a config file:
  python plot_multiple_runs.py --config runs_config.json --out ./comparison_plot
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def load_simulation_csv(csv_path):
    """
    Load simulation results from CSV.
    
    Args:
        csv_path: Path to simulation_results.csv
        
    Returns:
        df: DataFrame with columns [step_idx, {room}_real, {room}_pred, ...]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def extract_room_names(df):
    """
    Extract room names from DataFrame columns.
    
    Assumes columns are named: {room}_real, {room}_pred
    Returns unique room names.
    """
    room_names = set()
    for col in df.columns:
        if col.endswith("_real"):
            room_names.add(col[:-5])  # Remove "_real" suffix
        elif col.endswith("_pred"):
            room_names.add(col[:-5])  # Remove "_pred" suffix
    return sorted(list(room_names))


def plot_multiple_runs(runs_data, room_names, outdir, dpi=140, show_real=True):
    """
    Plot multiple simulation runs on the same figure for comparison.
    
    Args:
        runs_data: List of dicts with keys: {'df': DataFrame, 'label': str, 'color': str (optional)}
        room_names: List of room/output names
        outdir: Output directory
        dpi: Plot resolution
        show_real: Whether to plot real data (from first run only)
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Define colors for runs if not specified
    colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']
    for i, run_data in enumerate(runs_data):
        if 'color' not in run_data:
            run_data['color'] = colors[i % len(colors)]
    
    d_y = len(room_names)
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    
    for i, room in enumerate(room_names):
        ax = plt.subplot(2, 3, i + 1)
        
        # Plot real data from first run (if available and requested)
        if show_real and f"{room}_real" in runs_data[0]['df'].columns:
            df_first = runs_data[0]['df']
            steps = df_first['step_idx'].values
            real_data = df_first[f"{room}_real"].values
            ax.plot(steps, real_data, label="Real", color="tab:blue", linewidth=2, alpha=0.7)
        
        # Plot predicted data from all runs
        for run_data in runs_data:
            df = run_data['df']
            label = run_data['label']
            color = run_data['color']
            
            if f"{room}_pred" in df.columns:
                steps = df['step_idx'].values
                pred_data = df[f"{room}_pred"].values
                ax.plot(steps, pred_data, label=label, color=color, linestyle="--", linewidth=2)
        
        ax.set_title(room.split(":")[0], fontsize=10)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    run_labels = ", ".join([rd['label'] for rd in runs_data])
    plt.suptitle(f"Multi-Run Comparison | {run_labels}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fname = os.path.join(outdir, "multi_run_comparison.png")
    plt.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"[INFO] Saved comparison plot to {fname}")


def plot_metrics_comparison(runs_data, room_names, outdir, metric='mae', dpi=140):
    """
    Plot metrics (MAE or RMSE) for each room across all runs.
    
    Args:
        runs_data: List of dicts with keys: {'df': DataFrame, 'label': str}
        room_names: List of room names
        outdir: Output directory
        metric: 'mae' or 'rmse'
        dpi: Plot resolution
    """
    os.makedirs(outdir, exist_ok=True)
    
    metrics_dict = {run_data['label']: {} for run_data in runs_data}
    
    # Calculate metrics for each run
    for run_data in runs_data:
        df = run_data['df']
        label = run_data['label']
        
        for room in room_names:
            real_col = f"{room}_real"
            pred_col = f"{room}_pred"
            
            if real_col in df.columns and pred_col in df.columns:
                real_data = df[real_col].values
                pred_data = df[pred_col].values
                error = pred_data - real_data
                
                if metric.lower() == 'mae':
                    value = np.mean(np.abs(error))
                elif metric.lower() == 'rmse':
                    value = np.sqrt(np.mean(error**2))
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                metrics_dict[label][room] = value
    
    # Plot metrics
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    x = np.arange(len(room_names))
    width = 0.8 / len(runs_data)
    short_room_names = [room.split(":")[0] for room in room_names]
    
    for idx, run_data in enumerate(runs_data):
        label = run_data['label']
        values = [metrics_dict[label].get(room, 0) for room in room_names]
        offset = (idx - len(runs_data) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label)
    
    ax.set_xlabel("Room", fontsize=11)
    ax.set_ylabel(f"{metric.upper()} (°C)", fontsize=11)
    ax.set_title(f"Error Metrics ({metric.upper()}) Across Runs", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_room_names, rotation=0, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    fname = os.path.join(outdir, f"metrics_comparison_{metric}.png")
    plt.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"[INFO] Saved metrics plot to {fname}")


def main():
    parser = argparse.ArgumentParser(description="Plot multiple simulation runs for comparison.")
    parser.add_argument("--runs", nargs="+", help="Paths to simulation_results.csv files")
    parser.add_argument("--labels", nargs="+", help="Labels for each run (default: Run 1, Run 2, ...)")
    parser.add_argument("--config", type=str, help="JSON config file with run specifications")
    parser.add_argument("--out", type=str, default="./multi_run_comparison", help="Output directory")
    parser.add_argument("--show_real", action="store_true", help="Overlay real data on plots")
    parser.add_argument("--dpi", type=int, default=140, help="Plot DPI")
    parser.add_argument("--metrics", action="store_true", help="Also plot metrics comparison")
    args = parser.parse_args()

    runs_data = []

    if args.config:
        # Load from config file
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        for run_spec in config.get('runs', []):
            csv_path = run_spec['path']
            label = run_spec.get('label', os.path.basename(os.path.dirname(csv_path)))
            color = run_spec.get('color', None)
            
            df = load_simulation_csv(csv_path)
            runs_data.append({'df': df, 'label': label, 'color': color})
    else:
        # Load from command line arguments
        if not args.runs:
            raise ValueError("Must specify either --runs or --config")
        
        for idx, csv_path in enumerate(args.runs):
            label = args.labels[idx] if args.labels and idx < len(args.labels) else f"Run {idx + 1}"
            df = load_simulation_csv(csv_path)
            runs_data.append({'df': df, 'label': label})

    # Extract room names from first run
    room_names = extract_room_names(runs_data[0]['df'])
    print(f"[INFO] Found {len(room_names)} rooms: {room_names}")

    # Plot comparison
    plot_multiple_runs(runs_data, room_names, args.out, dpi=args.dpi, show_real=args.show_real)

    # Plot metrics if requested
    if args.metrics:
        plot_metrics_comparison(runs_data, room_names, args.out, metric='mae', dpi=args.dpi)
        plot_metrics_comparison(runs_data, room_names, args.out, metric='rmse', dpi=args.dpi)

    print(f"[INFO] All plots saved to {args.out}")


if __name__ == "__main__":
    main()
