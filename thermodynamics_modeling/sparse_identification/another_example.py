"""
## Quadratic 2D ODE
 
This example shows a demonstration of SINDy on a quadratic two-dimensional damped harmonic oscillator.
 
We generate training data by integrating the following linear system of differential equations with initial condtion `(1,0)`.
 
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Quadratic damped SHO method is not defined and thus we have to create one
def quadratic_damped_SHO(t:float, x:np.ndarray, delta:float=0.1, alpha:float=1.0, beta:float=0.5) -> np.ndarray:
    """
    Computes the time derivatives for a **quadratically damped simple harmonic oscillator (SHO)**.

    The governing equations are \n
        dx1/dt = x2 \n
        dx2/dt = -delta * x2 - alpha * x1 - beta * (x1**2)
    
    Args:
        t: current time (not used explicitly)
        x: vector where x[0] = position and x[1] = velocity
        delta: linear damping coefficient.
        alpha: linear stiffness coefficient.
        beta: nonlinear quadratic stiffness coefficient.

    Returns:
        the time derivatives `[dx1dt, dx2dt]`
        
    """
    x1, x2 = x  # position, velocity
    dx1dt = x2
    dx2dt = -delta * x2 - alpha * x1 - beta * (x1**2)
    return np.array([dx1dt, dx2dt])



# Generate training data

dt = 0.01
t_train = np.arange(0, 25, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [1, 0]
x_train = solve_ivp(quadratic_damped_SHO, t_train_span, 
                    x0_train, t_eval=t_train, **integrator_keywords).y.T


# Next we fit a SINDy model to the training data, finding that it recovers the correct governing equations.

# Fit the model

poly_order = 5
threshold = 0.05

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)
model.fit(x_train, t=dt)
model.print()

import pysindy as ps
import sympy as sp
import jax.numpy as jnp
from jax import jit
import diffrax

import sympy as sp
import re

# 1. Get the PySINDy equation strings
eq_strings = model.equations()

# 2. Preprocess each string to make it SymPy-friendly
def fix_expr(expr: str):
    expr = expr.replace('^', '**')                      # power operator
    expr = re.sub(r'(?<=\d)\s*(?=x\d)', '*', expr)      # insert * between numbers and xN
    expr = re.sub(r'(?<=x\d)\s+(?=x\d)', '*', expr)     # insert * between xN and xM (e.g. "x0 x1")
    return expr

fixed_eqs = [fix_expr(eq) for eq in eq_strings]

# 3. Create proper SymPy symbols
n_states = len(fixed_eqs)
x = sp.symbols(f'x0:{n_states}')

# 4. Parse into SymPy expressions
eqs_sympy = [sp.sympify(eq, locals={f'x{i}': x[i] for i in range(n_states)}) for eq in fixed_eqs]


eqs = model.equations()
n_states = len(eqs)
x = sp.symbols(f'x0:{n_states}')
rhs_jax = sp.lambdify(x, eqs_sympy, modules='jax')

@jit
def f_jax(t, y, args=None):
    return jnp.array(rhs_jax(*y))

term = diffrax.ODETerm(f_jax)
solver = diffrax.Dopri5()
y0 = jnp.array([1.0] * n_states)
# Request outputs at the same time grid as the original training data
saveat = diffrax.SaveAt(ts=jnp.array(t_train))
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=float(t_train[0]),
    t1=float(t_train[-1]),
    dt0=float(dt),
    y0=y0,
    saveat=saveat,
)
# Print a concise, readable summary of the JAX/Diffrax solution
print("JAX solution summary:")
try:
    # diffrax returns an object with arrays for times and states when available
    ts = getattr(sol, 'ts', None)
    ys = getattr(sol, 'ys', None)

    if ts is not None:
        # convert to numpy for nicer printing
        ts_np = np.asarray(ts)
        print(f"  times: length={len(ts_np)}; first 10: {ts_np[:10]}")
    else:
        print("  times: (not available in solution object)")

    if ys is not None:
        ys_np = np.asarray(ys)
        print(f"  ys shape: {ys_np.shape}")
        # Print first few time steps and first few state values for quick inspection
        n_print_t = min(10, ys_np.shape[0])
        n_print_s = min(5, ys_np.shape[1] if ys_np.ndim > 1 else 1)
        print(f"  first {n_print_t} timesteps (first {n_print_s} states):")
        print(ys_np[:n_print_t, :n_print_s])
    else:
        # Fallback: print the whole object if no ts/ys attributes
        print("  solution object:", sol)

except Exception as e:
    print("  Could not pretty-print solution:", e)
    print(sol)

# Convert JAX solution to numpy array for plotting comparison (if available)
jax_sol = None
if hasattr(sol, 'ys') and sol.ys is not None:
    jax_sol = np.asarray(sol.ys)
    # If shape is (n_states, len(ts)) transpose to (len(ts), n_states)
    if jax_sol.ndim == 2 and jax_sol.shape[0] == n_states and jax_sol.shape[1] == len(t_train):
        jax_sol = jax_sol.T
    # If shape is (len(ts), n_states) that's ideal

# The learned model can be used to evolve initial conditions forward in time. Here we plot the trajectories predicted by the SINDy model against those of the true governing equations.

# Simulate and plot the results

x_sim = model.simulate(x0_train, t_train) # see how the training and the test data are organised within the aforementioned variables; see also how control data influences the model (how does pysindy deal with it)
plot_kws = dict(linewidth=2) # do we have to tell the model which atuator influences which sensor or can it learn by itself?

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="SINDy-model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")

# Overlay JAX/Diffrax solution if available
if jax_sol is not None:
    # Ensure jax_sol shape matches (len(t_train), n_states)
    if jax_sol.shape[0] == len(t_train) and jax_sol.shape[1] >= 2:
        axs[0].plot(t_train, jax_sol[:, 0], "g-.", label="JAX-Diffrax", linewidth=1.5)
        axs[0].plot(t_train, jax_sol[:, 1], "g-.")
    else:
        # Try to broadcast or slice if shapes slightly differ
        try:
            axs[0].plot(t_train[:jax_sol.shape[0]], jax_sol[:jax_sol.shape[0], 0], "g-.", label="JAX-Diffrax", linewidth=1.5)
            axs[0].plot(t_train[:jax_sol.shape[0]], jax_sol[:jax_sol.shape[0], 1], "g-.")
        except Exception:
            pass
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
if jax_sol is not None and jax_sol.shape[0] == len(t_train):
    axs[1].plot(jax_sol[:, 0], jax_sol[:, 1], "g-.", label="JAX-Diffrax", linewidth=1.0)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
plt.show()

