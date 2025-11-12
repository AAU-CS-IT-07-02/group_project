import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Note that in the context of this example, we have
# thermal capacitance (C), which represents how much heat the room stores; thermal resistance (R), which determines how easily heat flows between inside and outside;
# heat input (Q_in) from HVAC or internal gains; and outdoor temperature (T_out) as a time-varying input.
# The differential equation describing the system is: C * dT_in/dt = (T_out - T_in)/R + Q_in.

def rc_model(t, T_in, R, C, T_out_func, Q_in_func):
    T_out = T_out_func(t)
    Q_in = Q_in_func(t)
    dTdt = (T_out - T_in)/(R*C) + Q_in/C
    return dTdt

# Parameters
R = 2.0       # K/W
C = 1e5       # J/K
T_in0 = 20.0  # Initial indoor temperature [°C]

# External conditions
def T_out_func(t):
    return 10 + 5 * np.sin(2 * np.pi * t / 24)  # daily outdoor temperature cycle

def Q_in_func(t):
    return 1000 if 8 <= (t % 24) <= 18 else 0  # HVAC on from 8:00 to 18:00

t_span = (0, 72)  # simulate 3 days
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(rc_model, t_span, [T_in0],
                args=(R, C, T_out_func, Q_in_func),
                t_eval=t_eval)

plt.figure(figsize=(10,5))
plt.plot(sol.t, sol.y[0], label="Indoor Temp")
plt.plot(sol.t, [T_out_func(t) for t in sol.t], '--', label="Outdoor Temp")
plt.xlabel("Time [hours]")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.title("1R1C Lumped Thermal Model")
plt.grid(True)
plt.show()
