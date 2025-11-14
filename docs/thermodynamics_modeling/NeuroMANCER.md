# Neuromancer NODE for Building Dynamics

This example demonstrates how to use Neuromancer to build and train a Neural Ordinary Differential Equation (NODE) model for predicting building dynamics based on sensor and actuator data.

### What is Neuromancer?
Neuromancer is a Python library for physics-informed machine learning, enabling the integration of neural networks with dynamical systems, constraints, and optimization problems.

### Why NODE?
Neural ODEs model continuous-time dynamics using differential equations parameterized by neural networks. They are ideal for:
- Capturing complex temporal dependencies.
- Handling irregular sampling.
- Providing interpretable latent dynamics.

#### What are latent dynamics
A latent space is a lower-dimensional representation of your system’s state that captures the essential dynamics without modeling every physical variable explicitly.
Instead of working in the original high-dimensional space (e.g., all sensor readings), we map the data into a compact latent representation.

>It acts like a compressed version of the system state, preserving the most important features for predicting future behavior.

### Workflow Overview
1. Data Preparation  
   - Load building dataset from CSV.
   - Normalize and create time-windowed sequences.
   - Split into train/dev/test sets.

2. Model Construction  
   - Encoder: Maps observed outputs to latent initial state.
   - NODE: Continuous-time latent dynamics integrated with RK4.
   - Decoder: Maps latent states back to physical outputs.

3. Training  
   - Define objectives (trajectory tracking, one-step prediction).
   - Optimize using Adam.
   - Validate and test on unseen data.

4. Evaluation  
   - Compare predicted vs. true trajectories.
   - Visualize inputs, disturbances, and outputs.

### Key Features
- Uses AAU-BUILD dataset (or similar building sensor data).
- Implements RK4 integration for NODE.
- Includes visualization of predictions.

---

### How Neuromancer Works with NODE

*   NODE represents system dynamics as a continuous-time ODE:
    $$
    \frac{dx}{dt} = f(x, u, d)
    $$
    where:
    *   (x): latent state
    *   (u): control inputs
    *   (d): disturbances
*   Neuromancer integrates this ODE using numerical methods (e.g., RK4) and wraps it in a System object for multi-step rollouts.
*   The workflow:
    1.  Encoder: Maps observed outputs (Y) to latent initial state (x\_0).
    2.  NODE Dynamics: Evolves latent state over time using learned ODE.
    3.  Decoder: Maps latent states back to physical outputs.

---

### Abbreviations and Symbols

| Symbol      | Meaning                                 |
| ----------- | --------------------------------------- |
| nx      | Dimension of latent state space         |
| ny      | Number of observed outputs              |
| nu      | Number of control inputs                |
| nd      | Number of disturbances                  |
| xn      | Initial latent state                    |
| Y       | Observed outputs (temperature, etc.)    |
| U       | Control inputs (valve position, etc.)   |
| D       | Disturbances (weather, occupancy, etc.) |
| H       | Prediction horizon (sequence length)    |
| dt\_sec | Integration timestep in seconds         |

---

# Implementation

::: thermodynamics_modeling.neuromancer.NODE
