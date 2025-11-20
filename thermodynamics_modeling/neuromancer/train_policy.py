import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuromancer.problem import Problem
from NODE import build_model, get_colums


def load_config(config_path='config.yml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_test_data(config):
    """Load and preprocess test data."""
    CSV = config["test_data"]
    df = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp')
    
    Y_df, U_df, D_df = get_colums(df)
    
    # Compute normalization stats
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
    
    norm_stats = {
        'muY': muY_vals, 'stdY': stdY_vals,
        'muU': muU_vals, 'stdU': stdU_vals,
        'muD': muD_vals, 'stdD': stdD_vals
    }
    
    return Y, U, D, norm_stats, Y_df, U_df, D_df


def load_model(config):
    """Rebuild and load the trained model."""
    ny, nu, nd = config.get("ny", 6), config.get("nu", 2), config.get("nd", 5)
    H = config.get("H", 16)
    dt_sec = config.get("dt_sec", 5 * 60)
    
    print(f"[INFO] Building model with ny={ny}, nu={nu}, nd={nd}, H={H}")
    encode_sym, dynamics_model = build_model(ny, nu, nd, H, dt_sec)
    
    # Load pre-trained weights
    model_path = config.get("model_path", "./out_300/best_model.pth")
    print(f"[INFO] Loading model weights from {model_path}")
    torch.serialization.add_safe_globals([Problem])
    problem = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    return problem, encode_sym, dynamics_model


def verify_model(problem, test_data, norm_stats):
    """Verify that the loaded model can produce predictions."""
    print("\n[VERIFY] Testing model inference...")
    
    Y_seq = test_data["Y"]
    test_name = test_data["name"]
    ny = Y_seq.shape[-1]
    
    # Run inference
    with torch.no_grad():
        outputs = problem(test_data)
    
    # Debug: print available output keys
    print(f"[DEBUG] Available output keys: {list(outputs.keys())}")
    
    # Construct expected output key (based on naming convention: {name}_y)
    expected_key = f"{test_name}_y"
    
    # Check if output exists
    if expected_key not in outputs:
        print(f"[ERROR] Expected output key '{expected_key}' not found in outputs!")
        print(f"[ERROR] Available keys: {list(outputs.keys())}")
        return False, None, None
    
    yhat = outputs[expected_key].cpu().numpy()
    print(f"[VERIFY] Model output shape: {yhat.shape}")
    print(f"[VERIFY] Expected shape: {Y_seq.shape}")
    
    # Check shape compatibility
    if yhat.shape != Y_seq.shape:
        print(f"[WARNING] Output shape {yhat.shape} does not match expected shape {Y_seq.shape}")
        print(f"[WARNING] This might be expected if the model has different output structure")
    
    # Denormalize and check values
    true_traj = (Y_seq.cpu().numpy() * norm_stats['stdY']) + norm_stats['muY']
    
    # Try to denormalize predictions if shapes are compatible
    try:
        pred_traj = (yhat * norm_stats['stdY']) + norm_stats['muY']
    except ValueError as e:
        print(f"[WARNING] Could not denormalize predictions: {e}")
        pred_traj = yhat
    
    print(f"[VERIFY] Predicted trajectory range: [{pred_traj.min():.2f}, {pred_traj.max():.2f}]")
    print(f"[VERIFY] True trajectory range: [{true_traj.min():.2f}, {true_traj.max():.2f}]")
    
    # Check for NaNs or Infs
    if np.isnan(pred_traj).any() or np.isinf(pred_traj).any():
        print("[ERROR] Predictions contain NaN or Inf values!")
        return False, None, None
    
    print("[VERIFY] ✓ Model inference completed!")
    return True, outputs, yhat


def plot_predictions(outputs, test_data, norm_stats, config, output_dir="./out_policy"):
    """Plot model predictions vs ground truth."""
    test_name = test_data["name"]
    Y_seq = test_data["Y"]
    expected_key = f"{test_name}_y"
    
    yhat = outputs[expected_key].cpu().numpy()
    true_vals = Y_seq.cpu().numpy()
    
    # Denormalize
    pred_traj = (yhat * norm_stats['stdY']) + norm_stats['muY']
    true_traj = (true_vals * norm_stats['stdY']) + norm_stats['muY']
    
    room_names = config.get("rooms", [f"Room_{i}" for i in range(Y_seq.shape[-1])])
    
    # Plot 1: Multi-room comparison
    print("\n[PLOT] Creating multi-room comparison plot...")
    plt.figure(figsize=(14, 10))
    for i, room in enumerate(room_names):
        pred_flat = pred_traj[:, :, i].reshape(-1)
        true_flat = true_traj[:, :, i].reshape(-1)
        
        plt.subplot(len(room_names), 1, i+1)
        plt.plot(true_flat, label='True', color='cyan', linewidth=1.5)
        plt.plot(pred_flat, label='Predicted', color='magenta', linestyle='--', linewidth=1.5)
        plt.title(f'{room} Temperature', fontsize=10)
        plt.xlabel('Time step')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/policy_multi_room_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[PLOT] Saved to {plot_path}")
    plt.close()
    
    # Plot 2: Error distribution
    print("[PLOT] Creating error distribution plot...")
    plt.figure(figsize=(14, 10))
    colors = ['red', 'blue', 'orange', 'grey', 'yellow', 'purple']
    for i, room in enumerate(room_names):
        errors = pred_traj[:, :, i].reshape(-1) - true_traj[:, :, i].reshape(-1)
        
        plt.subplot(len(room_names), 1, i+1)
        plt.hist(errors, bins=50, color=colors[i % len(colors)], edgecolor='black', alpha=0.7)
        plt.title(f'Prediction Error Distribution for {room}', fontsize=10)
        plt.xlabel('Error (°C)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    error_path = f"{output_dir}/policy_error_distribution.png"
    plt.savefig(error_path, dpi=150)
    print(f"[PLOT] Saved to {error_path}")
    plt.close()


def main(args):
    """Main entry point."""
    print("[INFO] Loading configuration...")
    config = load_config(args.config)
    
    print("[INFO] Loading test data...")
    Y, U, D, norm_stats, Y_df, U_df, D_df = load_test_data(config)
    
    print("[INFO] Loading trained model...")
    problem, encode_sym, dynamics_model = load_model(config)
    
    # Prepare test data for inference
    start_idx = -500
    end_idx = None
    H = config.get("H", 16)
    
    Y_seq = torch.tensor(Y[start_idx:end_idx], dtype=torch.float32).unsqueeze(0)
    U_seq = torch.tensor(U[start_idx:end_idx], dtype=torch.float32).unsqueeze(0)
    D_seq = torch.tensor(D[start_idx:end_idx], dtype=torch.float32).unsqueeze(0)
    xn = Y_seq[:, :1, :]
    
    test_data = {
        "xn": xn,
        "Y": Y_seq,
        "U": U_seq,
        "D": D_seq,
        "name": "policy_test"
    }
    
    print(f"[INFO] Test data prepared: Y={Y_seq.shape}, U={U_seq.shape}, D={D_seq.shape}")
    
    # Adjust model's sequence length to match test data
    problem.nodes[1].nsteps = Y_seq.shape[1]
    print(f"[INFO] Set model nsteps to {Y_seq.shape[1]}")
    
    # Verify model if flag is set
    if args.verify:
        success, outputs, yhat = verify_model(problem, test_data, norm_stats)
        if not success and args.verify:
            print("[ERROR] Model verification failed!")
            return 1
        
        # Plot predictions if verification passed
        if args.plot:
            plot_predictions(outputs, test_data, norm_stats, config)
    
    print("[INFO] ✓ Setup complete. Ready for policy training.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load trained neuromancer model and setup for policy control training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run model verification after loading"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot predictions vs ground truth after verification"
    )
    
    args = parser.parse_args()
    exit_code = main(args)
    exit(exit_code)
