"""
## Unknown control input function example

This example illustrates how to use sindy when we do not know the control input function.

If you only have a vector of control input values at the times in t_test and do not know the functional form for u, the simulate function will internally form an interpolating function based on the vector of control inputs.
As a consequence of this interpolation procedure, simulate will not give a state estimate for the last time point in t_test.
This is because the default integrator, scipy.integrate.solve_ivp (with LSODA as the default solver), is adaptive and sometimes attempts to evaluate the interpolant outside the domain of interpolation, causing an error.

"""

import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import pysindy as ps
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control

if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = 0.04
    t_end_test = 0.04

data = (Path() / "../data").resolve()


@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters


if __name__ == "testing":
    import sys
    import os

    sys.stdout = open(os.devnull, "w")

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

# Control input


def u_fun(t):
    return np.column_stack([np.sin(2 * t), t**2])


# Generate measurement data
dt = 0.002

# Generate training data

t_train = np.arange(0, t_end_train, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27]
x_train = solve_ivp(
    lorenz_control,
    t_train_span,
    x0_train,
    t_eval=t_train,
    args=(u_fun,),
    **integrator_keywords,
).y.T
u_train = u_fun(t_train)

model = ps.SINDy()
model.fit(x_train, u=u_train, t=dt)
model.print()

# Generate testing data

t_test = np.arange(0, t_end_test, dt)
t_test_span = (t_test[0], t_test[-1])
u_test = u_fun(t_test)
x0_test = np.array([8, 7, 15])
x_test = solve_ivp(
    lorenz_control,
    t_test_span,
    x0_test,
    t_eval=t_test,
    args=(u_fun,),
    **integrator_keywords,
).y.T
u_test = u_fun(t_test)

x_test_sim = model.simulate(x0_test, t_test, u=u_test)

# Note that the output is one example short of the length of t_test
print("Length of t_test:", len(t_test))
print("Length of simulation:", len(x_test_sim))

# here we plot the graphs
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(12, 4))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test[:-1], x_test[:-1, i], "k", label="true simulation")
    axs[i].plot(t_test[:-1], x_test_sim[:, i], "r--", label="model simulation")
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

plt.show()