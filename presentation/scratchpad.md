# System Characterisation

---

### System Characterisation

One of the main non-functional requirements for this semester's project in CS-IT-07 was security, which we had to let go of previously as a compromise when pivoting from failed modelling techniques.

In classical control theory, dynamical systems are often described in terms of differential equations, and one of the main benefits of this representation is that it gives us access to rigorous tools that let us analyse the modelled dynamics and by proxy, the underlying system.

With this methodology one can prove system properties and behaviours, and thus provide formal security guarantees like what is a safe operating envelope for system, unrecoverable errors, or how does the system respond to disturbances.

---

### Statistical Model Checking

As explained previously, in the pursuit of more accurate automatic modelling of system dynamics, we pivoted from white-box approaches like PySINDy, that generated systems of differential equations, to black-box neural networks. This way, we lost access to these mathematical analysis tools, and in turn the ability to provide formal guarantees.

To recover some of them without sacrificing model accuracy, we decided to make use of UPPAAL **Statistical Model Checking (SMC)**, a verification paradigm that shifts from formal proofs to empirical measurements. Rather than analytically proving properties, SMC helps us characterize system behavior through systematic simulation.

---

#### Experimental design

Our UPPAAL SMC integration works by:

1. Defining experimental scenarios using UPPAAL timed automata templates that specify initial conditions, disturbances, and control input sequences.
2. Formulating property queries in probabilistic logic (e.g., `Pr[≤T](<> property)`) that check with what probability does a property hold within a time horizon T.
3. Running extensive simulations where the NODE model is queried thousands of times under the specified scenario using Monte Carlo to introdue variations across runs. 
4. Computing statistical confidence from the results of experiments to extrapolate the probability of properties holding true.

With this approach we can assess black-box systems because it doesn't require interpretable equations. We just need the ability to simulate the system, a measurable property, and sufficient computational resources.

---

### Equilibrium

There are a variety of properties that are often verified in controlled dynamical systems, but this project limited the scope to four of them. The first of which is equilibrium.

It is defined as a point where system dynamics stabilize, meaning system states don't change over time. It is formalised as a combination of system states X and control inputs U for which the first derivative of the dynamics is zero.
We translated the property into an SMC query as: Pr[<=T] (<> total_t_derivative < threshold), where total_t_derivative is the numerically computed rate of change in system states. A threshold value is used to account for changes very close to zero and precision errors in floating point operations.

We found three scenarios were the property holds:
1. Active heating only: Where heating is continuously applied, room temperatures rise and settle at a warm equilibrium.

2. Passive cooling only: Where ventilation removes heat without active cooling, temperatures fall and settle at a cold equilibrium.

3. No control or disturbances: Without active heating or cooling, indoor temperatures gradually converge to the external environment.

---

### Stability

Stability addresses whether the system can recover when perturbed from a state of equilibrium. We formalise it in terms of Lyapunov stability, that states: if you displace the system to some nearby state, the trajectory should converge back to the equilibrium point over time.

To verify this with SMC, we start from the equilibrium points discovered in the previous analysis. The UPPAAL templates first induce each of those equilibrium states in the system, then apply a perturbation for a defined period of time, and finally allow the system to settle back to equilibrium by applying the original control input. We measure stability with the following query, which evaluates whether the system, after sufficient settling time, remains close to the previous average temperature.

The results confirmed that in fact all three states of equilibrium are stable.

---

### Reachability and Controllability

Reachability is defined as the set of states for which there is a valid sequence of control inputs between them and a given initial state. Formalised as follows: there exists a control input U over time T to take the system from x0 to xT.

To verify the reachable envelope with SMC, we use two scenarios. A "Warm" controller that maximizes heating to find the upper bound, and a "Cold" controller that maximizes cooling to find the lower bound. Each one runs until thermal saturation, and then we measure the temperatures achieved.

Results show that we can reach temperatures of +5 C degrees and -3 C degrees around outside temperature. This asymmetry reflects the building's physical design: powerful heating through radiators versus limited passive ventilation with no mechanical cooling.

Controllability extends the definition of reachability by requiring that the system can be steered from any initial state to any target state within the feasible envelope. Formally described as: ∀x0, xT : ∃u(t) : x0 → xT. For any initial and target states x0 and xT, there exists a control input U over time that connects them.

Under the assumption that the system dynamics are linear and the system is continuously controllable, we say that any state within the upper and lower bound can be taken as the initial state and there would exists a control sequence from it to any temperature within the +5 and -3 reachable envelope.
These key takeways can help us prevent impossible setpoints and establish the feasible state space for control objectives.

---

### Conclusions
These are the main conclussions we have reached throughout this project:

- The NODE model shows sufficient accuracy; and that forecasts of external conditions are critical for its performance
- SMC successfully recovers security guarantees lost with black-box models and confirmed expected dynamics: equilibrium, stability, and a reachable/controllable envelope of [+ 5°C, -3°C] around outside temperature.
- The UPPAAL REST API bridge enables integration with external ecosystems, most notably Pytorch, opening up a lot of posibilities
- With respect to the controllers, per-room control outperforms global control; but policy learning improves on all of them

And overall, that automatic System identification still requires domain knowledge and control theory expertise

---

### Further Work
We have outlined a variety of possible projects that could be implemented on top of our work, but we want to emphazise the two inmediate steps that we would've liked to take if given enough time, which is:
- Real-world deployment and validation on an actual building instead of simulations
- Expand the scope of the model and controller to include whole-building thermodynamics
- Enhance controller expressivity: move to a wider range of configurations for each actuator
- Expand optimization objectives beyond actuator wear and comfort to include energy cost and carbon footprint
- Assess transfer learning across buildings to evaluate generalization instead of retraining for each scenario
- Long-horizon planning incorporating known future states outside prediction horizon (e.g., scheduled occupancy patterns)
- Occupancy-aware control using booking systems for predictive pre-heating/cooling strategies

---