# SMC System Characterization: Slide Blueprint

---

## Slide 1: Classical Analysis of Dynamical Systems

### ON THE SLIDE:
- **Headline:** Classical Analysis: The Foundation of Control Security
- **Visual Elements:**
  - Diagram: Control hierarchy (sensors → model → controller → actuators)
  - Highlight "Safety & Security" in red
- **Text to Display:**
  - "Building Control: Safety-Critical Infrastructure"
  - "Requirement: Formal Guarantees for System Behavior"
  - Three foundational properties (as icons/boxes):
    - Lyapunov Stability
    - Reachability Analysis
    - Invariant Sets

### SPEAKER NOTES:
- Open by emphasizing building systems affect occupant safety and comfort
- Classical control theory gives us mathematical proofs of system behavior
- Three key properties: stability (system bounces back), reachability (what's controllable), invariants (safe zones we never leave)
- This is the gold standard in control engineering—proven, trusted, industry-standard
- But there's a problem when we move to neural networks...

---

## Slide 2: The Challenge—Neural Models Lack Transparency

### ON THE SLIDE:
- **Headline:** The Accuracy-Verifiability Trade-Off
- **Text to Display:**
  - "Left side: Classical equation (`dT/dt = −α(T − T_ext) + β·u(t)`) with ✓ checkmark"
  - "Right side: Neural network diagram with "?" question mark"
  - Arrow between them labeled "Design Trade-off"
- **Text to Display (Body):**
  - "Classical: Interpretable and Verifiable"
  - "Neural: Accurate but Opaque"
  - "Selection depends on application requirements"

### SPEAKER NOTES:
- Classical methods require explicit mathematical equations we can analyze
- Example: simple thermal equation—every term has physical meaning, we can prove properties
- Neural ODE models learn from data, dramatically more accurate on real building data
- But the network doesn't give us equations; it's a black box
- We can't apply Lyapunov analysis, we can't formally verify reachability
- This is the core tension: accuracy vs. verifiability
- Leads to critical question: How do we deploy high-accuracy models safely?

---

## Slide 3: Statistical Model Checking (SMC)—A Pragmatic Bridge

### ON THE SLIDE:
- **Headline:** Statistical Model Checking: Simulation-Based Verification
- **Visual Elements:**
  - Flowchart (simple, clean):
    - Property Definition → Probabilistic Query → Run Simulations → Statistical Result
- **Text to Display:**
  - "Verification through empirical measurement"
  - Key metric: "95% Confidence" (vs classical "Mathematical Proof")
  - "Quantify system behavior without explicit model interpretation"

### SPEAKER NOTES:
- SMC is a different verification approach, not inferior, just different
- Instead of mathematical proof, we use simulation-based probabilistic verification
- Convert system properties to stochastic reachability questions
- Run thousands of simulations and compute probabilities
- Key insight: we don't need equations to measure behavior
- UPPAAL has SMC built-in, making this practical for our NODE model
- Think of it like statistical testing in science: we can't prove something with 100% certainty, but we can achieve high confidence
- This allows us to keep the accuracy of neural models while getting verifiable guarantees
- **Important note:** SMC is general—you can verify many different properties
  - Safety properties (does X always hold?)
  - Liveness properties (does Y eventually happen?)
  - Performance metrics (how often does Z occur?)
  - Response time, energy efficiency, comfort bounds, etc.
- Building control has many interesting properties to check
- We selected **three key ones** to demonstrate the approach and characterize our system
- These three give us a comprehensive picture but are just examples of what's possible

---

## Slide 3b (Optional Transition): Properties We Can Verify with SMC

### ON THE SLIDE:
- **Headline:** What Properties Can We Test?
- **Visual Elements:**
  - Large list/grid of property types:
    - Safety (constraints never violated)
    - Liveness (goals eventually achieved)
    - Performance (response times)
    - Efficiency (energy optimization)
    - Robustness (disturbance rejection)
    - And many more...
  - Highlight three selected: Equilibrium, Stability, Controllability
- **Text to Display:**
  - "SMC is General-Purpose"
  - "Many interesting properties in building control"
  - "We focused on three key ones"

### SPEAKER NOTES:
- SMC isn't limited to a fixed set of properties
- You can formulate any system behavior as a probabilistic query
- Safety properties: "Does this constraint always hold?"
- Liveness properties: "Does the system eventually reach this goal?"
- Performance: "What's the response time to reach setpoint?"
- Efficiency: "How often are actuators cycling?"
- Robustness: "How does the system handle disturbances?"
- Energy: "What's the energy consumption pattern?"
- The list goes on—basically any quantifiable system behavior
- For this project, building control raised certain critical questions
- We narrowed focus to three fundamental properties that characterize the system
- These three give us confidence about basic system behavior
- Later work could verify additional properties (energy efficiency, comfort maintenance, actuator wear, etc.)
- This demonstrates the flexibility of SMC approach

---

## Slide 4: Property 1—Equilibrium: Predictable Long-Term Behavior

### ON THE SLIDE:
- **Headline:** Does the System Settle Into Predictable States?
- **Key Visual:**
  - Results table (3 rows):
    | Control | Equilibrium | Confidence |
    |---|---|---|
    | Active Heating | Thermal Saturation (Warm) | 95% |
    | Passive Cooling | Thermal Saturation (Cool) | 95% |
    | No Control | Outside Temperature | 95% |
- **Text to Display:**
  - "Equilibrium = stable, predictable state"
  - "Three different equilibria for three control modes"
  - "All verified with 95% confidence"

### SPEAKER NOTES:
- Equilibrium is a fundamental concept in dynamics: where does the system settle?
- Mathematically: rate of change approaches zero (all temperatures stable)
- Why it matters: Building occupants need predictable conditions
- Our test measures the probability that temperature rates fall below a threshold
- Results show three distinct equilibria depending on control strategy:
  - When heating is active, rooms warm up to saturation
  - When only cooling/ventilation is active, rooms cool down to saturation
  - When nothing is controlled, temperature equalizes with outside
- All three achieve 95% confidence—this is what we'd expect physically
- This verification confirms the model responds realistically to different control modes

---

## Slide 5: Property 2—Stability: Resilience to Disturbances

### ON THE SLIDE:
- **Headline:** Does the System Recover After Unexpected Changes?
- **Visual Elements:**
  - Temperature trajectory plot showing:
    - Baseline state
    - Disturbance applied (marked region)
    - System deviation
    - Recovery back to baseline
  - Shade the "perturbation window" and "recovery" phases differently
- **Text to Display:**
  - "Stability = Bounces Back from Disturbances"
  - "Tested with: Sudden temperature spikes, then released"
  - "Result: Returns to original state with 95% confidence"

### SPEAKER NOTES:
- Stability is critical for safety: system should not diverge after unexpected events
- Classical Lyapunov stability: small perturbations don't cause large deviations
- Translates to building context: open window, sudden occupancy, sensor spikes—system should recover
- Our test methodology:
  - Start at equilibrium
  - Apply sudden disturbance (e.g., 10-degree temperature spike)
  - Maintain it for a period
  - Remove it and check if system returns to original state
- We measure convergence: does average temperature return within tolerance?
- Results show 95% confidence in local stability
- This gives occupants and operators confidence in closed-loop control
- Even if something unexpected happens, the system will self-correct

---

## Slide 6: Property 3—Controllability & Reachability: Operational Limits

### ON THE SLIDE:
- **Headline:** What's the Actual Range of Control?
- **Visual Elements:**
  - Temperature envelope diagram showing asymmetric band:
    - Center line = Outside temperature
    - Upper bound: +5°C (reachable with heating)
    - Lower bound: −3°C (reachable with passive cooling)
    - Shade reachable region, show constraints
- **Text to Display:**
  - "Reachable Envelope: Outside Temp ± (−3°C to +5°C)"
  - "Asymmetric = reflects real physics"
  - "Active heating available, No mechanical cooling"

### SPEAKER NOTES:
- Controllability answers: what states can we actually reach with available actuators?
- Key insight: if we can reach the extremes, we can reach everything in between (continuous dynamics)
- Why test bounds? Because it proves full controllability across the interval
- Our building has asymmetric capabilities:
  - Strong active heating (radiators with controlled valves)
  - No mechanical cooling (only passive ventilation)
  - Result: can warm up easily, cooling is harder
- Test approach:
  - "Warm" controller: maximize heating, find max reachable temperature
  - "Cold" controller: minimize heating/open ventilation, find min temperature
- Results:
  - Maximum: 5°C above outside (reaches thermal saturation with heating)
  - Minimum: 3°C below outside (passive cooling limit)
- This asymmetry is realistic and expected
- Practical value: Defines safe setpoints for controller design
- Can't command a 20°C room when outside is 5°C—not feasible
- These bounds guide realistic controller objectives

---

## Slide 7: Interpreting Results—Security & Operational Implications

### ON THE SLIDE:
- **Headline:** What Does This All Mean for Building Control?
- **Visual Elements:**
  - Three-column layout (one per property):
    | Equilibrium | Stability | Controllability |
    |---|---|---|
    | ✓ Predictable | ✓ Robust | ✓ Feasible |
    | No drift | Disturbance rejection | Realistic bounds |
- **Text to Display:**
  - "Verified Properties → Design Confidence"
  - "System behaves as expected"
  - "Safe for real-world deployment"

### SPEAKER NOTES - EQUILIBRIUM IMPLICATIONS:
- Confirms system reaches stable states under different modes
- No uncontrolled drift or runaway behavior
- Different equilibria appropriate for different seasons/times
- Comfort guarantee: temperatures won't keep rising/falling indefinitely

### SPEAKER NOTES - STABILITY IMPLICATIONS:
- System exhibits disturbance rejection
- Resilient to unexpected events: open windows, sudden occupancy spikes, weather changes
- Building maintains comfort despite perturbations
- Justifies closed-loop control strategies
- Operators and occupants can trust the system will self-correct

### SPEAKER NOTES - CONTROLLABILITY IMPLICATIONS:
- Defines operational boundaries (what's physically possible)
- Prevents unrealistic controller demands
- Asymmetry is realistic, not a limitation
- Guides controller design within feasible region
- Can set seasonal targets knowing what's achievable

### SPEAKER NOTES - OVERALL MESSAGE:
- All three properties verified → comprehensive system characterization
- These aren't just academic metrics; they directly impact controller design
- Next step: knowing these bounds, design controllers that exploit them safely

---

## Slide 8: Conclusion—SMC as Pragmatic Security

### ON THE SLIDE:
- **Headline:** Statistical Verification for Data-Driven Control
- **Visual Elements:**
  - Two-column comparison:
    | Classical Analysis | SMC Approach |
    |---|---|
    | Mathematical proof | 95% confidence |
    | Requires equations | Works with black boxes |
    | Limited by complexity | Scales with simulation |
- **Text to Display:**
  - "Accuracy + Verifiability"
  - "Not just proven—statistically verified"
  - "Ready for real-world deployment"
- **Closing Statement Box:**
  - "SMC provides verifiable guarantees for data-driven models in safety-critical systems."

### SPEAKER NOTES - THE TRADE-OFF:
- We chose high-accuracy neural models over interpretable equations
- Gave up: Classical mathematical proofs
- Gained: Working with real building data, superior predictions
- Question was: can we still provide security guarantees?
- Answer: Yes, through SMC

### SPEAKER NOTES - WHAT WE HAVE NOW:
- 95% statistical confidence in equilibrium, stability, controllability
- No need to understand how the neural network makes predictions
- Practical security: operational guarantees without mathematical proof
- This approach is applicable beyond buildings—any black-box system in UPPAAL

### SPEAKER NOTES - WHY THIS MATTERS:
- Real building dynamics are complex; sometimes neural models outperform physics-based
- Classical verification was impossible for our NODE model
- SMC bridges the gap: we get both accuracy AND verifiability
- Different kind of proof, but still rigorous
- Enables safe deployment of modern ML in safety-critical infrastructure

### SPEAKER NOTES - FUTURE POSSIBILITIES:
- Extend analysis to robustness (sensor noise, actuator faults)
- Other control properties (energy efficiency, response time)
- Hybrid approaches: classical on subsystems where possible
- Broader ML verification in critical systems

### SPEAKER NOTES - CLOSING:
- This isn't a compromise; it's an evolution of verification methods
- Real systems are too complex for pure classical analysis
- Data-driven models are here; we just needed the right verification tools
- SMC proves trustworthy control doesn't require full interpretability

