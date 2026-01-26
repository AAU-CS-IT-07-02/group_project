# System Characterisation


---

### System Characterisation Notes

One of the main non-functional requirements for this semester's project in CS-IT7 was security, which we had to let go of previously, as a compromise when pivoting from failed modelling techniques.

In classical control theory, dynamical systems are often described in terms of differential equations. One of the main benefits of this representation is that it gives us access to rigorous mathematical tools that let us analyse the modelled dynamics and by proxy, the underlying system.

With this methodology one can prove properties and system behaviour invariants, and consequently provide formal security guarantees like a safe operating envelope for system, unrecoverable errors, unreachable states or response characteristic for disturbances.

---

### Statistical Model Checking

As explained previously, in the pursuit of more accurate automatic modelling of system dynamics, we pivoted from white-box approaches like PySINDy, that generated systems of differential equations, to black-box neural networks. This way, we lost access to these mathematical analysis tools, and in turn the ability to provide formal security guarantees.

To recover these guarantees without sacrificing model accuracy, we decided to make use of UPPAAL **Statistical Model Checking (SMC)**, a verification paradigm that shifts from mathematical proof to empirical measurements. Rather than analytically proving properties hold, SMC helps us characterize system behavior through systematic simulation and statistical analysis.

---

#### Experimental design

Our UPPAAL SMC integration works by:

1. Defining experimental scenarios using UPPAAL timed automata templates that specify initial conditions, disturbances, and control input sequences.
2. Formulating property queries in probabilistic logic (e.g., `Pr[≤T](<> property)`) that check with what probability does a property hold within a time horizon T.
3. Running extensive simulations where the NODE model is queried thousands of times under the specified scenario using Monte Carlo to introdue variations across runs. 
4. Computing statistical confidence from the resulst of experiments to extrapolate the probability of properties holding true and with what confidence.

With this approach we can assess black-box systems because it doesn't require interpretable equations or analytical tractability. We just need the ability to simulate the system, a measurable property, and sufficient computational resources for statistical analysis.

---

### Equilibrium

There are a variety of properties that a often verified in controlled dynamical systems, but this project limited the scope to four of them. The first of which is equilibrium, that helps us find configurations were oscillations are reduced and there is no uncontrollable drift.

It is defined as a point where system dynamics stabilize, a combination of system states and control inputs that don't generate change. It is formalised as a combination of system states X and control inputs U for which the first derivative of the dynamics is zero.
We translated the property into an SMC query as: Pr[<=T] (<> total_t_derivative < threshold), where total_t_derivative is the numerically computed rate of change in system states. A threshold value is used to account for changes very close to zero and precision errors in floating point operations.

We found three scenarios were the property holds:
1. Active heating only: System reaches thermal saturation (warm). When heating is continuously applied, room temperatures rise until limited by insulation loss and system capacity, settling at a warm equilibrium.

2. Passive cooling only: System reaches thermal saturation (cool). When ventilation removes heat without active heating, temperatures fall until the cooling capacity is exhausted, settling at a cool equilibrium.

3. No control or disturbances: Without active heating or cooling inputs, indoor temperatures gradually converge to the external environment.

---

### Stability

Stability addresses whether the system can recover when perturbed from a state of equilibrium. We formalise it in terms of Lyapunov stability: if you displace the system to some nearby state, the trajectory should converge back to the equilibrium point over time. The mathematical formulation states that the distance between the current state and the equilibrium state should shrink to zero as time progresses.

To verify this with SMC, we start from the equilibrium points discovered in the previous analysis. The UPPAAL templates first induce each of those equilibrium states in the system, then apply a perturbation for a defined period of time, and finally allow the system to settle back to equilibrium by applying the original control input configuration. We measure stability by comparing the system's average temperature in the first half of the simulation period versus the second half with the following query:  Pr[<= T]((time > T/2)&&(|t_avg(T/2) − t_avg(T)| < margin)). It evaluates whether the system, after sufficient settling time, remains close to the previous equilibrium state. 
The verification results confirmed local Lyapunov stability with 95% confidence for the three scenarios.

---

### Reachability and Controllability

Reachability is defined as the set of states for which there is a valid sequence of control inputs between the initial and target states. Formalised as follows: ∃u(t):x0​→xT​, there exists a control input u over time to take the system from x0 to xT.

To verify the reachable envelope with SMC, we use two scenarios. A "Warm" controller that maximizes heating to find the upper bound, and a "Cold" controller that maximizes cooling to find the lower bound. Each one runs until thermal saturation, then we measure the extreme temperatures achieved achieved.

Results show that we can reach temperatures of +5 C degrees and -3 C degrees around outside temperature. This asymmetry reflects the building's physical design: powerful heating through radiators versus limited passive ventilation with no mechanical cooling.

Controllability extends the definition of reachability by requiring that the system can be steered from any initial state to any target state within the feasible envelope. Formally described as: ∀x0, xT : ∃u(t) : x0 → xT. For any initial and target states x0 and xT, there exists a control input U over time that connects them.

Under the assumption that the system dynamics are linear and the system is continuously controllable, any state within the upper and lower bound can be taken as the initial state and there would exists a control sequence from it to any temperature within the +5 and -3 reachable envelope.
These key takeways can help us prevent impossible setpoints and establish the feasible state space for all control objectives.

---

### Conclusions

By applying Statistical Model Checking to our NODE model, we recovered the security guarantees we sacrificed when pivoting from equation-based to data-driven approaches. Three fundamental properties—equilibrium, stability, and reachability—comprehensively characterize system behavior and enable confident deployment in a safety-critical building environment.

The results show:
- **Predictable behavior**: System settles into realistic equilibria
- **Disturbance resilience**: System bounces back from unexpected events  
- **Bounded operation**: System respects physical constraints and reachable limits

These guarantees are empirical rather than mathematical proofs, but they are statistically rigorous and directly applicable to real-world operation. For building control, this represents a pragmatic balance: we retain the accuracy of neural models while recovering formal verification capability.

---

### Further Work

- **Extended property verification**: Expand beyond core properties to energy efficiency, comfort maintenance, actuator wear patterns
- **Robustness analysis**: Characterize system behavior under sensor noise, actuator faults, and model uncertainty
- **Scenario expansion**: Test broader range of disturbances and operating conditions
- **Hybrid verification**: Combine SMC with classical methods on subsystems where interpretability is possible
- **Real-world validation**: Deploy on actual building and compare simulation predictions with operational data

---