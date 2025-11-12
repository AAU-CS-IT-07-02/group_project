import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Note that in the context of this example, we have
# thermal capacitance (C), which represents how much heat the room stores; thermal resistance (R), which determines how easily heat flows between inside and outside;
# heat input (Q_in) from HVAC or internal gains; and outdoor temperature (T_out) as a time-varying input.
# The differential equation describing the system is: C * dT_in/dt = (T_out - T_in)/R + Q_in.

# Data parameters
np.random.seed(42) 
n_steps = 288  # 24 hours * 60 minutes / 5 minutes (that is, a 5 min step)
dt = 300  # 5 minutes

# True building parameters for the data
R_true = 2.5  # K/W
C_true = 1e5  # J/K

# Generate outdoor temperature: daily sinusoidal variation
time = np.arange(n_steps)
T_out_series = 10 + 5*np.sin(2*np.pi*time/n_steps)  # 10°C average, +-5°C daily variation

# Generate heating input (Q_in)
Q_in_series = np.zeros(n_steps)
Q_in_series[50:150] = 5000  # heater ON for some period
Q_in_series[200:250] = 3000  # smaller heating later

def rc_model(T_in, T_out, Q_in, R, C, dt):
    dT = ((T_out - T_in)/R + Q_in) * dt / C
    return T_in + dT

T0 = 20  # initial indoor temperature
T_measured = [T0]
for k in range(1, n_steps):
    T_next = rc_model(T_measured[-1], T_out_series[k-1], Q_in_series[k-1], R_true, C_true, dt)
    T_measured.append(T_next)

T_measured = np.array(T_measured)

T_measured += np.random.normal(0, 0.3, size=n_steps)  # 0.3°C noise

data = pd.DataFrame({
    "T_measured": T_measured,
    "T_out": T_out_series,
    "Q_in": Q_in_series
})
data.to_csv("alternative_methods\synthetic_bms_data.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(T_measured, label="Indoor (measured)")
plt.plot(T_out_series, label="Outdoor", linestyle="--")
plt.xlabel("Time step")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.show()

def simulate(T0, T_out_series, Q_in_series, R, C, dt):
    T_sim = [T0]
    for k in range(1, len(T_out_series)):
        T_next = rc_model(T_sim[-1], T_out_series[k-1], Q_in_series[k-1], R, C, dt)
        T_sim.append(T_next)
    return np.array(T_sim)

def cost(params, T_measured, T0, T_out_series, Q_in_series, dt):
    R, C = params
    T_sim = simulate(T0, T_out_series, Q_in_series, R, C, dt)
    error = T_sim - T_measured
    return np.sum(error**2)  # sum of squared errors

# initial guesses
initial_guess = [2.0, 1e5]
res = minimize(cost, initial_guess, args=(T_measured, T0, T_out_series, Q_in_series, dt),
               bounds=[(0.1, 10), (1e3, 1e6)])

R_est, C_est = res.x
print("Estimated R:", R_est)
print("Estimated C:", C_est)

# simulate using estimated parameters
T_sim = simulate(T0, T_out_series, Q_in_series, R_est, C_est, dt)

plt.figure(figsize=(10,5))
plt.plot(T_measured, label="Measured")
plt.plot(T_sim, label="Simulated", linestyle="--")
plt.xlabel("Time step")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.show()
