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


# The learned model can be used to evolve initial conditions forward in time. Here we plot the trajectories predicted by the SINDy model against those of the true governing equations.

# Simulate and plot the results

x_sim = model.simulate(x0_train, t_train)
plot_kws = dict(linewidth=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
fig.show()

