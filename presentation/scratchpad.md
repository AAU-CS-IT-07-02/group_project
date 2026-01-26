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

**Definition**: The system's ability to recover from perturbations and return to equilibrium. A fundamental property for control systems: small disturbances should not cause large deviations.

**Why it matters**: Real buildings experience unexpected disturbances—open windows, sudden occupancy, sensor spikes, weather changes. We need confidence the system self-corrects rather than diverging.

**How we tested it**: We initialized the system at equilibrium, applied a bounded perturbation (e.g., sudden temperature spike), maintained it for a period, then released it. We measured whether the system returned to its original state within a tolerance.

**Results**: 
- System deviates momentarily when disturbed but **reliably returns to original state** with 95% confidence
- Response is proportional to disturbance magnitude (larger spikes take longer to recover)
- Local stability is robust across different control modes

**Security implication**: Justifies closed-loop control. Even if unexpected conditions arise, the system has proven capability to self-correct and maintain safe operation.

---

### Reachability and Controllability

**Definition**: The set of states reachable through control inputs; defines what operational conditions are actually achievable.

**Why it matters**: Operators need to know realistic bounds. Setting impossible targets (e.g., 22°C when outside is -10°C with only passive cooling) frustrates control and wastes energy. Understanding the reachable envelope prevents unrealistic demands.

**How we tested it**: We ran two control strategies: a "Warm" controller maximizing heating, and a "Cold" controller minimizing heating/maximizing ventilation. We measured the extreme temperatures reachable under each.

**Results**:
- **Maximum reachable**: Outside temperature + 5°C (reached under active heating)
- **Minimum reachable**: Outside temperature − 3°C (reached under passive cooling)
- **Asymmetry reflects physics**: Strong active heating, weak passive cooling
- All verified with 95% confidence

**Security implication**: Defines operational constraints. Provides realistic bounds for controller design and operator expectations. System can't violate physical limits—this is built-in safety.

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