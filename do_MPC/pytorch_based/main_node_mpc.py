import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# Try to import do_mpc, if fails, add relative path
try:
    import do_mpc
except ImportError:
    rel_do_mpc_path = os.path.join('..','..')
    sys.path.append(rel_do_mpc_path)
    import do_mpc

# Add path to find node_adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from node_adapter import load_models, create_node_model

def main():
    # ---------------------------------------------------------
    # 1. Setup Paths and Load Weights
    # ---------------------------------------------------------
    # Adjust this path to where the weights actually are
    # Using the path found in the workspace
    weights_path = r"c:\software\AAU\group_project_and_dataset\group_project\thermodynamics_modeling\neuromancer\out_300\best_model_state_dict.pth"
    
    if not os.path.exists(weights_path):
        print(f"Warning: Weights not found at {weights_path}")
        print("Please adjust 'weights_path' in the script.")
        return

    print("Loading PyTorch models...")
    fx_model, decoder_model = load_models(weights_path)
    
    # ---------------------------------------------------------
    # 2. Create do-mpc Model (Discrete, RK4 integrated)
    # ---------------------------------------------------------
    dt = 300.0 # 5 minutes in seconds
    model = create_node_model(fx_model, decoder_model, dt=dt)
    
    # ---------------------------------------------------------
    # 3. Setup MPC
    # ---------------------------------------------------------
    mpc = do_mpc.controller.MPC(model)
    
    setup_mpc = {
        'n_horizon': 10,
        't_step': dt,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    
    # Objective: Track setpoint for output 'y' (Room temperatures)
    # We assume normalized data, so 0.0 is the mean temperature.
    # Let's try to track 0.0 (mean).
    
    _y = model.aux['y']
    loss = ca.sumsqr(_y - 0.0) # Track 0 (mean)
    
    mpc.set_objective(mterm=loss, lterm=loss)
    
    # Input penalty (regularization)
    mpc.set_rterm(u=1e-2)
    
    # Constraints
    # Normalized inputs usually within [-3, 3] or similar.
    # Let's set some loose bounds.
    mpc.bounds['lower', '_u', 'u'] = -3.0
    mpc.bounds['upper', '_u', 'u'] = 3.0
    
    # We must provide the TVP (disturbances) for the prediction horizon
    # In a real scenario, this would be a weather forecast.
    # Here we setup a TVP function that returns zeros (mean weather).
    
    # IMPORTANT: get_tvp_template() returns a structure that must be filled.
    # The simulator and MPC might have different templates if they are different objects,
    # but here they share the model structure.
    
    tvp_template_mpc = mpc.get_tvp_template()
    def tvp_fun_mpc(t_now):
        # Fill with zeros (or your forecast data)
        # tvp_template_mpc['_tvp', :, 'd'] = 0.0 # Already initialized to 0 usually
        return tvp_template_mpc
    
    mpc.set_tvp_fun(tvp_fun_mpc)
    
    mpc.setup()
    
    # ---------------------------------------------------------
    # 4. Setup Simulator
    # ---------------------------------------------------------
    # We use the same model as the plant (Digital Twin)
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=dt)
    
    # Simulator also needs TVP
    # Note: Simulator TVP function is slightly different, it returns the TVP for the *current* step
    # whereas MPC returns for the *horizon*. However, do-mpc handles the structure check.
    # We must fetch the template FROM THE SIMULATOR instance to be safe.
    
    tvp_template_sim = simulator.get_tvp_template()
    def tvp_fun_sim(t_now):
        return tvp_template_sim
        
    simulator.set_tvp_fun(tvp_fun_sim)
    
    simulator.setup()
    
    # ---------------------------------------------------------
    # 5. Closed-loop Simulation
    # ---------------------------------------------------------
    # Initial state (latent)
    x0 = np.random.randn(4, 1) * 0.1 # Small random initial latent state
    
    mpc.x0 = x0
    simulator.x0 = x0
    
    mpc.set_initial_guess()
    
    n_steps = 20
    
    # Lists to store results
    y_history = []
    u_history = []
    
    print("Starting simulation...")
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        
        # Get current output y from simulator (auxiliary variable)
        # simulator.data['_aux', 'y'] contains history
        y_current = simulator.data['_aux', 'y'][-1]
        y_history.append(y_current)
        u_history.append(u0)
        
        print(f"Step {k}: y_mean={np.mean(y_current):.3f}")
        
    # ---------------------------------------------------------
    # 6. Plotting
    # ---------------------------------------------------------
    y_history = np.array(y_history).squeeze()
    u_history = np.array(u_history).squeeze()
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    
    ax[0].plot(y_history)
    ax[0].set_ylabel('Normalized Temp (y)')
    ax[0].set_title('Closed-loop Simulation with NODE Surrogate')
    ax[0].grid(True)
    
    ax[1].plot(u_history)
    ax[1].set_ylabel('Normalized Input (u)')
    ax[1].set_xlabel('Step')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
