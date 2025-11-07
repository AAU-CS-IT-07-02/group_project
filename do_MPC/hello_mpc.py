import numpy as np
import do_mpc
import casadi as ca
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

def setup_model():
    """
    This is our model, based on the PySINDy output
    (x0)' = -9.999 x0 + 9.999 x1 + 0.999 u0
    (x1)' = 27.988 x0 + -0.998 x1 + -1.000 x0 x2
    (x2)' = -2.666 x2 + -1.000 u1 + 1.000 x0 x1
    """
    model = do_mpc.model.Model('continuous')

    """
    States (from PySINDy x0, x1, x2)
    """
    x0 = model.set_variable(var_type='_x', var_name='x0')
    x1 = model.set_variable(var_type='_x', var_name='x1')
    x2 = model.set_variable(var_type='_x', var_name='x2')
    
    """
    Controls (from PySINDy u0, u1)
    """
    u0 = model.set_variable(var_type='_u', var_name='u0')
    u1 = model.set_variable(var_type='_u', var_name='u1')

    """
    Define the ODEs
    """
    x0_dot = -9.999*x0 + 9.999*x1 + 0.999*u0
    x1_dot = 27.988*x0 - 0.998*x1 - 1.000*x0*x2
    x2_dot = -2.666*x2 - 1.000*u1 + 1.000*x0*x1

    model.set_rhs('x0', x0_dot)
    model.set_rhs('x1', x1_dot)
    model.set_rhs('x2', x2_dot)

    model.setup()
    return model

def setup_controller(model):
    """
    Setup the MPC controller, goal, and rules
    """
    mpc = do_mpc.controller.MPC(model)
    
    """Stop the wall of text"""
    setup_mpc = {
        'n_horizon': 20,
        't_step': 0.1,
        'suppress_ipopt_output': True,
    }
    mpc.set_param(**setup_mpc)

    """
    Get the variables from the model
    """
    x0 = model.x['x0']
    x1 = model.x['x1']
    x2 = model.x['x2']
    
    """
    Set the Goal: get all states to 0
    """
    mterm = (x0**2) + (x1**2) + (x2**2)
    lterm = (x0**2) + (x1**2) + (x2**2)
    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    """
    Set the Rules: don't use too much power!
    """
    mpc.set_rterm(u0=1e-2, u1=1e-2)
    
    """
    Actuator limits
    """
    mpc.bounds['lower', '_u', 'u0'] = -20.0
    mpc.bounds['upper', '_u', 'u0'] = 20.0
    mpc.bounds['lower', '_u', 'u1'] = -20.0
    mpc.bounds['upper', '_u', 'u1'] = 20.0
    
    mpc.setup()
    return mpc

def setup_simulator(model):
    """
    Setup the sim to test our controller
    """
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=0.1)
    simulator.setup()
    return simulator

"""
--- Main Program ---
"""
if __name__ == '__main__':
    
    print("Setting up model and controller...")
    model = setup_model()
    controller = setup_controller(model)
    simulator = setup_simulator(model)

    """
    Set the initial state of the system
    Same as the control_input_ex.py example
    """
    x0 = np.array([-8.0, 8.0, 27.0])
    simulator.x0 = x0
    controller.x0 = x0
    
    """
    Lists to store the results
    """
    x0_history = [x0[0]]
    x1_history = [x0[1]]
    x2_history = [x0[2]]
    u0_history = []
    u1_history = []
    
    print("--- Running MPC Simulation (100 steps) ---")
    
    for i in range(100):
        """
        Get the best control command
        """
        u = controller.make_step(x0)
        
        """
        Simulate one step
        """
        x_next = simulator.make_step(u)
        
        """
        store results
        """
        x0_history.append(x_next[0, 0])
        x1_history.append(x_next[1, 0])
        x2_history.append(x_next[2, 0])
        u0_history.append(u[0, 0])
        u1_history.append(u[1, 0])
        
        """ if i % 10 == 0: """
        """    print(f"Step {i}, State x0: {x_next[0, 0]}") """
        
        """
        Update x0 for the next loop
        """
        x0 = x_next.flatten()

    print("...Simulation Complete. Now creating animation...")

    """
    --- Animate the Results ---
    """
    
    """
    Convert lists to numpy arrays
    """
    t = np.arange(len(x0_history))
    x0_hist = np.array(x0_history)
    x1_hist = np.array(x1_history)
    x2_hist = np.array(x2_history)
    u0_hist = np.array(u0_history)
    u1_hist = np.array(u1_history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    """
    Setup the plots
    """
    ax1.set_title('System States (Lorenz)')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax2.set_title('Control Inputs')
    ax2.set_ylabel('Value')
    ax2.set_xlabel('Time Steps')
    ax2.grid(True)

    """
    Set axis limits
    """
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax1.set_ylim(min(x0_hist.min(), x1_hist.min(), x2_hist.min()) - 1, 
                   max(x0_hist.max(), x1_hist.max(), x2_hist.max()) + 1)
    ax2.set_ylim(min(u0_hist.min(), u1_hist.min()) - 0.1, 
                   max(u0_hist.max(), u1_hist.max()) + 0.1)

    """
    Init empty lines for the animation
    """
    line_x0, = ax1.plot([], [], lw=2, label='x0')
    line_x1, = ax1.plot([], [], lw=2, linestyle='--', label='x1')
    line_x2, = ax1.plot([], [], lw=2, linestyle=':', label='x2')
    ax1.legend()

    line_u0, = ax2.plot([], [], lw=2, label='Control u0', color='g')
    line_u1, = ax2.plot([], [], lw=2, linestyle='--', label='Control uG1', color='m')
    ax2.legend()

    """
    Function to update the plot for each frame
    """
    def animate(i):
      
        if i > 0:
            line_u0.set_data(t[1:i+1], u0_hist[:i])
            line_u1.set_data(t[1:i+1], u1_hist[:i])
            
        line_x0.set_data(t[:i+1], x0_hist[:i+1])
        line_x1.set_data(t[:i+1], x1_hist[:i+1])
        line_x2.set_data(t[:i+1], x2_hist[:i+1])
        return line_x0, line_x1, line_x2, line_u0, line_u1

    """
    Create the animation
    """
    ani = animation.FuncAnimation(fig, animate, frames=len(x0_history), 
                                  interval=50, blit=True)

    """
    Save the animation as a GIF
    """
    save_file = 'mpc_lorenz_animation.gif'
    try:
        ani.save(save_file, writer='pillow', fps=15)
        print(f"All done! Saved animation to '{save_file}'")
    except Exception as e:
        print(f"\nError saving GIF. Do you have 'pillow' installed? (pip install pillow)")
        print(f"Full Error: {e}\n")

    # plt.show() 
    plt.close(fig)