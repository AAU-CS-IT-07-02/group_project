import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
# import os
# rel_do_mpc_path = os.path.join('..','..','..')
# sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# States struct (optimization variables):
X_s = model.set_variable('_x',  'X_s')
S_s = model.set_variable('_x',  'S_s')
P_s = model.set_variable('_x',  'P_s')
V_s = model.set_variable('_x',  'V_s')

# Input struct (optimization variables):
inp = model.set_variable('_u',  'inp')

# Certain parameters
mu_m  = 0.02
K_m   = 0.05
K_i   = 5.0
v_par = 0.004
Y_p   = 1.2

# Uncertain parameters:
Y_x  = model.set_variable('_p',  'Y_x')
S_in = model.set_variable('_p', 'S_in')

# Auxiliary term
mu_S = mu_m*S_s/(K_m+S_s+(S_s**2/K_i))

# Differential equations
model.set_rhs('X_s', mu_S*X_s - inp/V_s*X_s)
model.set_rhs('S_s', -mu_S*X_s/Y_x - v_par*X_s/Y_p + inp/V_s*(S_in-S_s))
model.set_rhs('P_s', v_par*X_s - inp/V_s*P_s)
model.set_rhs('V_s', inp)

# Build the model
model.setup()

mpc = do_mpc.controller.MPC(model)
do_mpc.controller.MPCSettings.supress_ipopt_output(mpc)

setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 1.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    # This solver cannot be used, it's proprietary
}

mpc.set_param(**setup_mpc)

mterm = -model.x['P_s'] # terminal cost
lterm = -model.x['P_s'] # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(inp=1.0) # penalty on input changes

# lower bounds of the states
mpc.bounds['lower', '_x', 'X_s'] = 0.0
mpc.bounds['lower', '_x', 'S_s'] = -0.01
mpc.bounds['lower', '_x', 'P_s'] = 0.0
mpc.bounds['lower', '_x', 'V_s'] = 0.0

# upper bounds of the states
mpc.bounds['upper', '_x','X_s'] = 3.7
mpc.bounds['upper', '_x','P_s'] = 3.0

# upper and lower bounds of the control input
mpc.bounds['lower','_u','inp'] = 0.0
mpc.bounds['upper','_u','inp'] = 0.2

Y_x_values = np.array([0.5, 0.4, 0.3])
S_in_values = np.array([200.0, 220.0, 180.0])

mpc.set_uncertainty_values(Y_x = Y_x_values, S_in = S_in_values)

# mpc.set_param(nlpsol='sqpmethod')
# mpc.set_param(nlpsol_opts={'qpsol': 'qrqp'})

mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 1.0
}

simulator.set_param(**params_simulator)

p_num = simulator.get_p_template()

p_num['Y_x'] = 0.4
p_num['S_in'] = 200.0

# function definition
def p_fun(t_now):
    return p_num

# Set the user-defined function above as the function for the realization of the uncertain parameters
simulator.set_p_fun(p_fun)

simulator.setup()

# Initial state
X_s_0 = 1.0 # Concentration biomass [mol/l]
S_s_0 = 0.5 # Concentration substrate [mol/l]
P_s_0 = 0.0 # Concentration product [mol/l]
V_s_0 = 120.0 # Volume inside tank [m^3]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

# Set for controller, simulator and estimator
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0
mpc.set_initial_guess()

import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
rcParams['text.usetex'] = True
#rcParams['text.latex.preamble'] = [r'\\usepackage{amsmath}',r'\\usepackage{siunitx}']
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(5, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics,mpc_graphics]:
    # Plot the state on axis 1 to 4:
    g.add_line(var_type='_x', var_name='X_s', axis=ax[0], color='#1f77b4')
    g.add_line(var_type='_x', var_name='S_s', axis=ax[1], color='#1f77b4')
    g.add_line(var_type='_x', var_name='P_s', axis=ax[2], color='#1f77b4')
    g.add_line(var_type='_x', var_name='V_s', axis=ax[3], color='#1f77b4')

    # Plot the control input on axis 5:
    g.add_line(var_type='_u', var_name='inp', axis=ax[4], color='#1f77b4')

"""
ax[0].set_ylabel(r'$X_s~[\si[per-mode=fraction]{\mole\per\litre}]$')
ax[1].set_ylabel(r'$S_s~[\si[per-mode=fraction]{\mole\per\litre}]$')
ax[2].set_ylabel(r'$P_s~[\si[per-mode=fraction]{\mole\per\litre}]$')
ax[3].set_ylabel(r'$V_s~[\si[per-mode=fraction]{\mole\per\litre}]$')
ax[4].set_ylabel(r'$u_{\text{inp}}~[\si[per-mode=fraction]{\cubic\metre\per\minute}]$')
ax[4].set_xlabel(r'$t~[\si[per-mode=fraction]{\minute}]$')"""

n_steps = 10
for k in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
plt.show()
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

make_animation = True

# The function describing the gif:
def update(t_ind):
    sim_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()

if make_animation:
    # Turn off interactive mode before saving animation
    plt.ioff()

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)

    writer = PillowWriter(fps=10)
    anim.save('anim_batch_reactor_final.gif', writer=writer)
    print("Animation saved as GIF.")
    plt.close(fig)