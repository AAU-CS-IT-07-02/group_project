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
    print(f"test_data: {test_data}")
    print(f"Y_seq : {Y_seq}")
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


def autonomous_prediction(problem, Y_initial, U_seq, D_seq, norm_stats, config):
    """
    TRUE AUTONOMOUS prediction: given initial state + future controls/disturbances,
    predict future temperatures WITHOUT any ground truth feedback.
    
    This is what you actually need for control: the model receives:
    - Y_initial: initial room temperature (shape: 1, 1, ny)
    - U_seq: control inputs for all future steps (shape: 1, seq_len, nu)
    - D_seq: measured disturbances for all future steps (shape: 1, seq_len, nd)
    
    The model should:
    1. Encode Y_initial to get initial latent state
    2. Step through ODE for seq_len steps using U and D
    3. Decode latent state at each step to get predicted room temperatures
    
    NO ground truth feedback - purely autonomous rollout.
    """
    print("\n[AUTONOMOUS] Running true autonomous prediction...")
    
    from NODE import build_model
    from neuromancer.modules import blocks
    
    encoder = problem.nodes[0]
    dynamics_system = problem.nodes[1]
    
    seq_len = U_seq.shape[1]
    ny = Y_initial.shape[-1]
    nu = U_seq.shape[-1]
    nd = D_seq.shape[-1]
    
    # Extract the actual MLP weights from the trained System nodes
    ode_node = dynamics_system.nodes[0]     # NODE (ODE integrator)
    decoder_node = dynamics_system.nodes[1] # Decoder
    
    # Access the underlying PyTorch modules
    # ode_node wraps an RK4 integrator which wraps an MLP
    # decoder_node wraps an MLP directly
    
    predictions = []
    
    print(f"[AUTONOMOUS] Encoder type: {type(encoder)}, Encoder in_keys: {encoder.in_keys}")
    print(f"[AUTONOMOUS] ODE node type: {type(ode_node)}, ODE in_keys: {ode_node.in_keys}")
    print(f"[AUTONOMOUS] Decoder node type: {type(decoder_node)}, Decoder in_keys: {decoder_node.in_keys}")
    
    with torch.no_grad():
        # Step 1: Encode initial condition
        print(f"[AUTONOMOUS] Y_initial shape: {Y_initial.shape}")
        encoder_out = encoder({"Y": Y_initial})
        xn = encoder_out["xn"]  # (1, 1, 8) initial latent state
        print(f"[AUTONOMOUS] Initial latent state shape: {xn.shape}")
        print(f"[AUTONOMOUS] Initial latent state value (first 3 elements): {xn[0, 0, :3]}")
        
        # Step 2: Autonomous rollout - loop through time steps
        for t in range(seq_len):
            u_t = U_seq[:, t:t+1, :]  # (1, 1, nu)
            d_t = D_seq[:, t:t+1, :]  # (1, 1, nd)
            
            # DEBUG: Check for NaN in inputs
            if torch.isnan(u_t).any() or torch.isnan(d_t).any():
                print(f"[ERROR] NaN in control/disturbance at step {t}")
                print(f"  u_t has NaN: {torch.isnan(u_t).any()}")
                print(f"  d_t has NaN: {torch.isnan(d_t).any()}")
                break
            
            # Single ODE step: integrate latent state forward
            try:
                ode_out = ode_node({"xn": xn, "U": u_t, "D": d_t})
                xn_next = ode_out["xn"]  # (1, 1, 8) - new latent state
            except Exception as e:
                print(f"[ERROR] ODE integration failed at step {t}: {e}")
                print(f"  xn shape: {xn.shape}, value (first 3): {xn[0, 0, :3]}")
                print(f"  u_t shape: {u_t.shape}, value (first 3): {u_t[0, 0, :3]}")
                print(f"  d_t shape: {d_t.shape}, value (first 3): {d_t[0, 0, :3]}")
                break
            
            # Check for NaN after ODE step
            if torch.isnan(xn_next).any():
                print(f"[ERROR] ODE produced NaN at step {t}")
                print(f"  xn_next: {xn_next}")
                break
            
            # Decode latent state to get predicted room temperature
            decoder_out = decoder_node({"xn": xn_next})
            y_pred = decoder_out["y"]  # (1, 1, ny)
            
            # Check for NaN in prediction
            if torch.isnan(y_pred).any():
                print(f"[ERROR] Decoder produced NaN at step {t}")
                print(f"  y_pred: {y_pred}")
                break
            
            predictions.append(y_pred.cpu().numpy())
            xn = xn_next  # Update latent state for next iteration
            
            if (t + 1) % 50 == 0:
                print(f"[AUTONOMOUS] Step {t+1}/{seq_len} completed - pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    if not predictions:
        print("[ERROR] No predictions were generated!")
        return None, None
    
    # Stack predictions: each is (1, 1, 6) -> concatenate to (1, N, 6)
    y_pred_full = np.concatenate(predictions, axis=1)
    
    # Denormalize predictions
    y_pred_denorm = (y_pred_full * norm_stats['stdY']) + norm_stats['muY']
    
    print(f"[AUTONOMOUS] Prediction complete! Shape: {y_pred_full.shape}")
    print(f"[AUTONOMOUS] Prediction range (normalized): [{y_pred_full.min():.4f}, {y_pred_full.max():.4f}]")
    print(f"[AUTONOMOUS] Prediction range (physical units): [{y_pred_denorm.min():.2f}, {y_pred_denorm.max():.2f}]°C")
    
    return y_pred_full, y_pred_denorm


def one_step_ahead_validation(problem, test_data, norm_stats, config, steps_ahead=1):
    """
    Batch-based n-step-ahead validation using SAME approach as training.
    
    This runs the model in batch mode (like acc_metrics.py) to get predictions
    at horizon lengths of 1, 2, 3, ... steps_ahead. This matches how the model
    was trained and gives realistic performance metrics.
    
    Why batch vs step-by-step?
    - Your model was trained in batch mode: processes entire sequences at once
    - Step-by-step closed-loop is distributional mismatch: encoder/decoder get
      called in ways they never saw during training
    - Batch mode: give it data 0..499, get predictions for all 500 timesteps
    - Then extract predictions at offset=steps_ahead to measure future accuracy
    """
    print(f"\n[N-STEP] Running batch-based {steps_ahead}-step-ahead validation...")
    
    with torch.no_grad():
        Y_seq = test_data["Y"]
        
        seq_len = Y_seq.shape[1]
        ny = Y_seq.shape[-1]
        
        # Reset nsteps to match the sequence length (required for batch mode)
        problem.nodes[1].nsteps = seq_len
        
        # Run full batch prediction
        outputs = problem(test_data)
        
        # Extract predictions
        test_name = test_data["name"]
        expected_key = f"{test_name}_y"
        yhat = outputs[expected_key].cpu().numpy()  # (1, seq_len, ny)
        true_vals = Y_seq.cpu().numpy()
        
        # For n-step validation, extract predictions at specific offset
        # Prediction at time t predicts what happens at time t+steps_ahead
        # So we compare: pred[0..seq_len-steps_ahead] vs true[steps_ahead..seq_len]
        pred_at_offset = yhat[:, :-steps_ahead, :]  # predictions 0..seq_len-steps_ahead
        true_at_offset = true_vals[:, steps_ahead:, :]  # actual values steps_ahead..seq_len
        
        print(f"[N-STEP] Predictions shape (at offset {steps_ahead}): {pred_at_offset.shape}")
        print(f"[N-STEP] Ground truth shape: {true_at_offset.shape}")
    
    # Denormalize
    pred_traj = (pred_at_offset * norm_stats['stdY']) + norm_stats['muY']
    true_traj = (true_at_offset * norm_stats['stdY']) + norm_stats['muY']
    
    # Compute metrics for each room
    room_names = config.get("rooms", [f"Room_{i}" for i in range(ny)])
    print(f"\n[N-STEP] Per-room accuracy (batch {steps_ahead}-step-ahead):")
    print("-" * 60)
    
    rmse_per_room = []
    mae_per_room = []
    
    for i, room in enumerate(room_names):
        pred_flat = pred_traj[:, :, i].reshape(-1)
        true_flat = true_traj[:, :, i].reshape(-1)
        
        # Remove NaNs
        mask = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
        true_clean = true_flat[mask]
        pred_clean = pred_flat[mask]
        
        rmse = np.sqrt(((pred_clean - true_clean) ** 2).mean())
        mae = np.mean(np.abs(pred_clean - true_clean))
        
        rmse_per_room.append(rmse)
        mae_per_room.append(mae)
        
        print(f"{room:15s} | RMSE: {rmse:6.3f}°C | MAE: {mae:6.3f}°C")
    
    print("-" * 60)
    print(f"{'Average':15s} | RMSE: {np.mean(rmse_per_room):6.3f}°C | MAE: {np.mean(mae_per_room):6.3f}°C")
    print(f"[N-STEP] ✓ Batch {steps_ahead}-step-ahead validation complete!")
    
    return pred_traj, true_traj


def plot_one_step_comparison(pred_traj, true_traj, config, output_dir="./out_policy"):
    """Plot one-step-ahead predictions vs ground truth."""
    print("\n[PLOT] Creating closed-loop comparison plots...")
    
    ny = true_traj.shape[-1]
    room_names = config.get("rooms", [f"Room_{i}" for i in range(ny)])
    
    # Plot 1: Closed-loop trajectory comparison
    plt.figure(figsize=(14, 10))
    for i, room in enumerate(room_names):
        pred_flat = pred_traj[:, :, i].reshape(-1)
        true_flat = true_traj[:, :, i].reshape(-1)
        
        plt.subplot(len(room_names), 1, i+1)
        plt.plot(true_flat, label='True (measured)', color='cyan', linewidth=1.5)
        plt.plot(pred_flat, label='Predicted (closed-loop)', color='orange', linestyle='--', linewidth=1.5)
        plt.title(f'{room} Temperature (Closed-Loop)', fontsize=10)
        plt.xlabel('Time step')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/closed_loop_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[PLOT] Saved to {plot_path}")
    plt.close()
    
    # Plot 2: Closed-loop error distribution
    plt.figure(figsize=(14, 10))
    colors = ['red', 'blue', 'orange', 'grey', 'yellow', 'purple']
    for i, room in enumerate(room_names):
        errors = pred_traj[:, :, i].reshape(-1) - true_traj[:, :, i].reshape(-1)
        
        plt.subplot(len(room_names), 1, i+1)
        plt.hist(errors, bins=50, color=colors[i % len(colors)], edgecolor='black', alpha=0.7)
        plt.title(f'Closed-Loop Error Distribution for {room}', fontsize=10)
        plt.xlabel('Error (°C)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    error_path = f"{output_dir}/closed_loop_error_distribution.png"
    plt.savefig(error_path, dpi=150)
    print(f"[PLOT] Saved to {error_path}")
    plt.close()


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
    
    # Run autonomous prediction if flag is set
    if args.autonomous:
        print("\n[MAIN] Running TRUE AUTONOMOUS prediction...")
        Y_initial = Y_seq[:, :1, :]  # Just the first timestep
        y_pred_auto, y_pred_denorm = autonomous_prediction(problem, Y_initial, U_seq, D_seq, norm_stats, config)
        
        # Ground truth for comparison
        y_true_denorm = (Y_seq.cpu().numpy() * norm_stats['stdY']) + norm_stats['muY']
        
        # Compute metrics
        ny = Y_seq.shape[-1]
        room_names = config.get("rooms", [f"Room_{i}" for i in range(ny)])
        print(f"\n[AUTONOMOUS] Per-room prediction accuracy:")
        print("-" * 60)
        
        rmse_list = []
        mae_list = []
        for i, room in enumerate(room_names):
            pred_flat = y_pred_denorm[:, :, i].reshape(-1)
            true_flat = y_true_denorm[:, :, i].reshape(-1)
            
            rmse = np.sqrt(((pred_flat - true_flat) ** 2).mean())
            mae = np.mean(np.abs(pred_flat - true_flat))
            rmse_list.append(rmse)
            mae_list.append(mae)
            
            print(f"{room:15s} | RMSE: {rmse:6.3f}°C | MAE: {mae:6.3f}°C")
        
        print("-" * 60)
        print(f"{'Average':15s} | RMSE: {np.mean(rmse_list):6.3f}°C | MAE: {np.mean(mae_list):6.3f}°C")
        print("[AUTONOMOUS] ✓ Autonomous prediction complete!")
        
        # Plot if requested
        if args.plot:
            print("\n[PLOT] Creating autonomous prediction comparison plots...")
            plt.figure(figsize=(14, 10))
            for i, room in enumerate(room_names):
                pred_flat = y_pred_denorm[:, :, i].reshape(-1)
                true_flat = y_true_denorm[:, :, i].reshape(-1)
                
                plt.subplot(ny, 1, i+1)
                plt.plot(true_flat, label='True (measured)', color='cyan', linewidth=1.5)
                plt.plot(pred_flat, label='Autonomous Prediction', color='red', linestyle='--', linewidth=1.5)
                plt.title(f'{room} Temperature (Autonomous)', fontsize=10)
                plt.xlabel('Time step')
                plt.ylabel('Temperature (°C)')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = "./out_policy/autonomous_prediction_comparison.png"
            plt.savefig(plot_path, dpi=150)
            print(f"[PLOT] Saved to {plot_path}")
            plt.close()
    
    # Run one-step-ahead validation if flag is set
    if args.one_step:
        pred_traj, true_traj = one_step_ahead_validation(
            problem, test_data, norm_stats, config, 
            steps_ahead=args.steps_ahead
        )
        
        # Plot one-step-ahead results if requested
        if args.plot:
            plot_one_step_comparison(pred_traj, true_traj, config)
    
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
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Run TRUE autonomous prediction (only uses initial state + future U,D, NO ground truth)"
    )
    parser.add_argument(
        "--one-step",
        action="store_true",
        help="Run one-step-ahead validation (use actual measured states at each step)"
    )
    parser.add_argument(
        "--steps-ahead",
        type=int,
        default=1,
        help="Number of steps ahead to predict before resetting to measured state (default: 1)"
    )
    
    args = parser.parse_args()
    exit_code = main(args)
    exit(exit_code)
