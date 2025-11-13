"""
## RC Network
 
Here we model a simple room with: Thermal capacitance (C), which represents how much heat the room stores;
thermal resistance (R), which determines how easily heat flows between inside and outside;
heat input (Q_in) from HVAC or internal gains; and outdoor temperature (T_out) as a time-varying input.
The differential equation describing the system is: C * dT_in/dt = (T_out - T_in)/R + Q_in.
 
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Note that in the context of this example, we have
# thermal capacitance (C), which represents how much heat the room stores; thermal resistance (R), which determines how easily heat flows between inside and outside;
# heat input (Q_in) from HVAC or internal gains; and outdoor temperature (T_out) as a time-varying input.
# The differential equation describing the system is: C * dT_in/dt = (T_out - T_in)/R + Q_in.

def rc_model(t, T_in, R, C, T_out_func, Q_in_func):
    """
    Computes the rate of change of indoor temperature for a first-order RC thermal model.

    The governing equation is
        dT_in/dt = (T_out - T_in) / (R * C) + Q_in / C
    
    Args:
        t: current time.
        T_in: current indoor (or system) temperature.
        R: thermal resistance between indoor and outdoor environments.
        C: thermal capacitance of the indoor environment or system.
        T_out_func: function that returns the outdoor (ambient) temperature at time t.
        Q_in_func: function that returns the internal heat gain or input at time t.

    Returns:
        the time derivative of the indoor temperature dTdt
    """
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
    """
    Defines the outdoor temperature profile as a sinusoidal function 
    representing a daily temperature cycle.

    The governing equation is
        T_out(t) = 10 + 5 * sin(2πt / 24)
    
    Args:
        t: current time (in hours).

    Returns:
        the outdoor temperature T_out at time t
    """
    return 10 + 5 * np.sin(2 * np.pi * t / 24)  # daily outdoor temperature cycle

def Q_in_func(t):
    """
    Defines the internal heat gain profile as a step function representing
    an HVAC system operating schedule.

    The governing rule is
        Q_in(t) = 1000  if  8 ≤ (t mod 24) ≤ 18  
                  0     otherwise
    
    Args:
        t: current time (in hours).

    Returns:
        the internal heat gain Q_in at time t
    """
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
