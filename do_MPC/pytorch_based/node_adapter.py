import torch
import torch.nn as nn
import casadi as ca
import numpy as np
import os
import sys

# Try to import do_mpc, if fails, add relative path
try:
    import do_mpc
except ImportError:
    # Assuming we are in do_MPC/pytorch_based/
    # and do_mpc package is in ../../ (root of repo or similar)
    # Adjust this based on actual location of do_mpc library
    rel_do_mpc_path = os.path.join('..','..')
    sys.path.append(rel_do_mpc_path)
    import do_mpc

# ---------------------------------------------------------
# 1. Define PyTorch Architectures (matching NODE.py)
# ---------------------------------------------------------

class DynamicsNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DecoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. Helper to load weights from the specific state_dict structure
# ---------------------------------------------------------
def load_models(path, nx=4, nu=20, nd=14, ny=6):
    # Dimensions
    input_dim_fx = nx + nu + nd
    hidden_dim = 40
    
    # Instantiate
    fx_model = DynamicsNet(input_dim_fx, hidden_dim, nx)
    decoder_model = DecoderNet(nx, hidden_dim, ny)
    
    # Load state dict
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
        
    sd = torch.load(path, map_location='cpu')
    
    # Map keys for Dynamics (fx)
    # nodes.1.nodes.0.callable.block.linear.0.weight -> net.0.weight
    fx_mapping = {
        'nodes.1.nodes.0.callable.block.linear.0.weight': 'net.0.weight',
        'nodes.1.nodes.0.callable.block.linear.0.bias':   'net.0.bias',
        'nodes.1.nodes.0.callable.block.linear.1.weight': 'net.2.weight', # Skip Tanh (1)
        'nodes.1.nodes.0.callable.block.linear.1.bias':   'net.2.bias',
        'nodes.1.nodes.0.callable.block.linear.2.weight': 'net.4.weight', # Skip Tanh (3)
        'nodes.1.nodes.0.callable.block.linear.2.bias':   'net.4.bias',
    }
    
    # Map keys for Decoder
    # nodes.1.nodes.1.callable.linear.0.weight -> net.0.weight
    decoder_mapping = {
        'nodes.1.nodes.1.callable.linear.0.weight': 'net.0.weight',
        'nodes.1.nodes.1.callable.linear.0.bias':   'net.0.bias',
        'nodes.1.nodes.1.callable.linear.1.weight': 'net.2.weight', # Skip ReLU (1)
        'nodes.1.nodes.1.callable.linear.1.bias':   'net.2.bias',
    }
    
    # Apply weights
    fx_sd = fx_model.state_dict()
    for k_old, k_new in fx_mapping.items():
        if k_old in sd:
            fx_sd[k_new] = sd[k_old]
    fx_model.load_state_dict(fx_sd)
    
    decoder_sd = decoder_model.state_dict()
    for k_old, k_new in decoder_mapping.items():
        if k_old in sd:
            decoder_sd[k_new] = sd[k_old]
    decoder_model.load_state_dict(decoder_sd)
    
    return fx_model, decoder_model

# ---------------------------------------------------------
# 3. Generic PyTorch -> CasADi Converter
# ---------------------------------------------------------
def pytorch_to_casadi(model, input_expr):
    """
    Converts a PyTorch Sequential model (or module with .net Sequential) 
    to a CasADi expression.
    """
    output_expr = input_expr
    
    # Handle both Sequential and custom modules wrapping Sequential
    layers = model.net if hasattr(model, 'net') else model
    
    for layer in layers:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()
            # CasADi mtimes: weight * input + bias
            # Ensure bias is column vector for broadcasting if needed, 
            # but CasADi handles 1D numpy array addition fine usually.
            output_expr = ca.mtimes(weight, output_expr) + bias
            
        elif isinstance(layer, nn.Tanh):
            output_expr = ca.tanh(output_expr)
            
        elif isinstance(layer, nn.ReLU):
            # ReLU(x) = max(0, x)
            output_expr = ca.fmax(0, output_expr)
            
        elif isinstance(layer, nn.Sigmoid):
            output_expr = 1 / (1 + ca.exp(-output_expr))
            
        else:
            raise NotImplementedError(f"Layer {layer} not supported in conversion.")
            
    return output_expr

# ---------------------------------------------------------
# 4. Create do-mpc Model
# ---------------------------------------------------------
def create_node_model(fx_model, decoder_model, dt=300.0):
    """
    Creates a do-mpc discrete model using the RK4 integrated dynamics.
    """
    model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')
    
    # Dimensions
    # We assume fx_model input is [x, u, d]
    # x: latent state (4)
    # u: control (20)
    # d: disturbance (14)
    
    nx = 4
    nu = 20
    nd = 14
    ny = 6
    
    # Define variables
    _x = model.set_variable(var_type='_x', var_name='x', shape=(nx, 1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(nu, 1))
    _d = model.set_variable(var_type='_tvp', var_name='d', shape=(nd, 1))
    
    # Concatenate inputs for fx
    # fx expects [x, u, d]
    fx_input = ca.vertcat(_x, _u, _d)
    
    # Define RK4 integration step
    # k1 = f(x, u, d)
    # k2 = f(x + h/2*k1, u, d)
    # k3 = f(x + h/2*k2, u, d)
    # k4 = f(x + h*k3, u, d)
    # x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Helper to evaluate fx symbolically
    # We can't reuse the same pytorch_to_casadi call efficiently if it creates new constants every time?
    # Actually, pytorch_to_casadi creates the expression graph. 
    # It's better to define a CasADi Function for fx first.
    
    # Create symbolic placeholders for Function definition
    sym_in = ca.SX.sym('in', nx + nu + nd)
    sym_out = pytorch_to_casadi(fx_model, sym_in)
    fx_func = ca.Function('fx', [sym_in], [sym_out])
    
    # RK4 steps
    h = dt
    
    k1 = fx_func(ca.vertcat(_x, _u, _d))
    k2 = fx_func(ca.vertcat(_x + 0.5*h*k1, _u, _d))
    k3 = fx_func(ca.vertcat(_x + 0.5*h*k2, _u, _d))
    k4 = fx_func(ca.vertcat(_x + h*k3, _u, _d))
    
    x_next = _x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    model.set_rhs('x', x_next)
    
    # Decoder output (auxiliary expression)
    y_expr = pytorch_to_casadi(decoder_model, _x)
    model.set_expression(expr_name='y', expr=y_expr)
    
    model.setup()
    return model

import os
