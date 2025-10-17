# PySINDy documentation + examples (summary)

The main class we'll make use of is called pySINDy. SINDy stands for
Sparse Identification of Non-linear Dynamical Systems, and, in a
nutshell, it uses sparse regression to learn a dynamical systems model
from measurement data. If we recall what it means for a vector to be
sparse, the concept of sparse regression becomes easier to grasp. A
vector is sparse if it has many elements equal to zero. Taking this into
consideration, sparse regression refers to a regression problem with the
additional requirement that the solution must be sparse. It is also
useful to know that dynamic system models generally represent systems
that have internal dynamics or memory of past states such as
integrators, delays, transfer functions, and state-space models
(https://bookdown.org/palomar/portfoliooptimizationbook/13.2-sparse-regression.html,
https://www.mathworks.com/help/control/ug/dynamic-system-models.html ).

------------------------------------------------------------------------

## Linear 2D ODE

In this example, we have a linear two-dimensional damped harmonic
oscillator. Here we start by generating the training data, using
`np.arange`, which gives us a numpy array of numbers between 0 and 25
with a 0.01 step value (time points). `t_train_span` is a tuple of the
first and the last element of that array (timespan). Using the initial
condition `(2,0)`, which means that when x is 2, y has to be 0, we train
using:

``` python
solve_ivp(linear_damped_SHO, t_train_span, x0_train,
          t_eval=t_train, **integrator_keywords).y.T
```

This function integrates a system of ordinary differential equations
given an initial condition. `linear_damped_SHO` is a function that
encodes the harmonic oscillator equations (giving us the derivatives),
`t_eval` tells the solver at which points we want the solution returned
and `**integrator_keywords` unpacks extra solver options. We have `.y.T`
because `solve_ivp` returns an array of points (`.t`) and an array of
solution values shaped `(n_vars, n_times)` (`.y`). Transposing (`.T`),
makes it `(n_times, n_vars)`.

As for the model itself, we use `ps.SINDy`, which uses sparse regression
to learn a dynamical systems model from measurement data. The
`optimizer` argument basically consists of a method used to fit the
model, in this case **STLSQ** (sequentially thresholded least squares
algorithm), the `feature_library` argument is used to specify candidate
right-hand side features, in this case the polynomial one. The others
are not relevant here. Regarding the `fit` function, we basically pass
the training data to the model and specify the timestep (which is 0.01).
As for the simulation, we use the initial condition from which we want
to simulate, which is `(2,0)`, and the array of points created with
`np.arange`. Finally, we plot it.

------------------------------------------------------------------------

## Cubic 2D ODE

This is another iteration of the previous example; the difference is
that we use `cubic_damped_SHO`.

------------------------------------------------------------------------

## Linear 3D ODE

In this example we have a linear 3D system that represents damped
rotational motion in the x-y plane and exponential decay in z. The
training data are generated in the same way as before, but here the time
range goes from 0 to 50. `t_train_span` is a tuple containing the first
and last time values. The initial condition is set to `[2, 0, 1]`,
meaning x = 2, y = 0, and z = 1 at the start. The data are generated
using:

``` python
solve_ivp(linear_3D, t_train_span, x0_train,
          t_eval=t_train, **integrator_keywords).y.T
```

This numerically solves the system of ordinary differential equations
defined in `linear_3D`. The `.y.T` transposes the solver output so that
each row corresponds to one time step and each column to one variable.

The SINDy model is fitted using a **fifth-order polynomial library** and
the **STLSQ optimizer** with a sparsity threshold of `0.01`. This
optimizer performs repeated least-squares fits while removing small
coefficients after each step, forcing the model to stay simple and
retain only the most important terms.

After fitting, the identified model is simulated again and compared with
the original data using time-series and 3D plots, showing that SINDy
accurately captures the true linear dynamics of the system.

------------------------------------------------------------------------

## Lorenz system (nonlinear ODE)

In this example we work with the Lorenz system, a nonlinear 3D system.
We start by generating the training data using `solve_ivp`, with a time
array created by:

``` python
np.arange(0, 100, dt)
```

and a small time step of `dt = 0.001`.

The system begins from the initial condition `[-8, 8, 27]`, and we use
the classic Lorenz parameters (σ = 10, ρ = 28, β = 8/3). This produces
the time evolution of x, y and z, as well as their derivatives, which
we'll use for model training.

The model is built using SINDy with a **fifth order polynomial feature
library**. The **STLSQ optimizer** is used with a threshold of `0.05`.
This means it first fits all possible terms using least squares, then
removes the ones with small coefficients, leaving only the dominant
terms that describe the main dynamics.

The model is fitted multiple times using different levels of noise added
to the data to see how noise affects its accuracy. After fitting, the
identified equations should ideally match the original linear system.

Finally, the model is simulated using the same initial conditions and
time span to check how well it reproduces the real dynamics. The results
are plotted both as time series (x, y, z vs. t) and as a 3D trajectory.
The dashed lines represent the SINDy model's simulation, while the solid
lines show the true solution. A good fit means the model has
successfully learned the underlying linear relationships between x, y,
and z.

## Another example of how to use SINDy
::: thermodynamics_modeling.sparse_identification.another_example
