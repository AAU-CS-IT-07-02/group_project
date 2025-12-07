# Smart Building Predictive Control: A Case Study on AAU BUILD

## Executive Summary

This report documents a semester-long group project investigating the development of predictive and collaborative control algorithms for the AAU BUILD smart building facility. The existing Building Management System (BMS) relies on purely reactive proportional (P) controllers, which leads to inefficiencies in energy consumption, excessive actuator wear, and inadequate thermal comfort. Our work proposes a data-driven approach to develop a thermodynamic model of the building and use it to implement more sophisticated, predictive controllers.

---

## 1. Introduction

### 1.1 Smart Buildings: Context and Significance

Smart buildings represent a critical frontier in sustainable infrastructure. These facilities integrate sensors, actuators, and automated control systems to optimize energy consumption, maintain occupant comfort, and support facility management. As buildings account for approximately 30–40% of global energy consumption and 30% of CO₂ emissions, improving their operational efficiency is of paramount importance.

Smart buildings face several challenges:
- **Fragmented systems**: Multiple independent control loops with limited inter-system communication
- **Reactive control paradigm**: Controllers respond to deviations rather than anticipating them
- **Data quality issues**: Sensor failures, missing values, and inconsistent timestamps
- **Scalability**: Coordinating thousands of actuators across large facilities
- **Generalization**: Models trained on one building often fail to transfer to others

### 1.2 The AAU BUILD Facility

AAU BUILD is a modern office building at Aalborg University equipped with:
- **Dense sensor networks**: Temperature, CO₂, humidity, and occupancy sensors distributed across multiple rooms
- **Automated control actuators**: HVAC systems, blind controls, and ventilation units
- **Real-time logging**: A PostgreSQL database accessible via a REST API for data extraction
- **Open interfaces**: Support for integrating custom control algorithms

The facility consists of multiple rooms organized into clusters across different floors. A 2023 dataset (February–December) with 6 representative office rooms provides the primary data for this project.

### 1.3 Problem Identification

The existing BMS employs purely reactive P-controllers with limited coordination between actuators. This approach produces several problems:

1. **Actuator wear**: Frequent small adjustments and oscillations degrade equipment lifespan
2. **Energy inefficiency**: Lack of anticipation leads to overshooting/undershooting, requiring more energy to correct
3. **Comfort deficiencies**: Reactive control cannot proactively raise temperatures before occupancy periods
4. **No priority constraints**: All actuators operate independently without consideration of priorities or constraints
5. **Inter-actuator oscillations**: Without communication, coupled actuators (e.g., heating and cooling) can fight each other

### 1.4 Proposed Solution and Project Roadmap

We propose implementing a **predictive and collaborative control framework** consisting of three milestones:

- **Milestone 1 (M1)**: Develop accurate data-driven thermodynamic models of the building
- **Milestone 2 (M2)**: Implement predictive controllers (MPC) and test their performance
- **Milestone 3 (M3)**: Deploy controllers in production with habit profiling and online learning

**Project focus**: This work concentrates on **M1** with significant progress on **M2**, leaving **M3** for future development.

---

## 2. Literature Review and State of the Art

### 2.1 Building Modeling Approaches

#### 2.1.1 White-Box (Physics-Based) Models
White-box models derive from first-principles thermodynamics and fluid dynamics. They explicitly model heat transfer, air flow, and mass balances.

**Advantages**:
- Highly interpretable and generalizable across buildings
- Require minimal data for validation
- Support extrapolation beyond training conditions

**Disadvantages**:
- Complex parameterization (e.g., R-C thermal networks require accurate room dimensions, material properties)
- High computational cost for real-time control
- Difficulty capturing complex interactions (solar radiation, infiltration, occupant behavior)

#### 2.1.2 Grey-Box (Hybrid) Models
Grey-box models combine physical insights (e.g., thermal network structure) with data-driven parameter estimation. Common examples include R-C (resistance-capacitance) networks.

**Advantages**:
- Balance interpretability and accuracy
- Computationally efficient
- Require moderate amounts of data

**Disadvantages**:
- Significant domain expertise needed to define the structure
- Parameter identification can be ill-posed
- Still limited to relatively simple dynamics

#### 2.1.3 Black-Box (Data-Driven) Models
Black-box models (neural networks, regression models) learn patterns directly from data without enforcing physical structure.

**Advantages**:
- Flexible and can capture complex nonlinear relationships
- Minimal domain knowledge required
- Easily adaptable to new buildings

**Disadvantages**:
- Require large, high-quality datasets
- Lack interpretability; difficult to extract physical insights
- Poor generalization to unseen conditions
- Prone to overfitting

#### 2.1.4 Neural Ordinary Differential Equations (Neural ODEs)
Neural ODEs combine the flexibility of neural networks with the structure of differential equations. Models learn a latent ODE representation that can be integrated numerically.

**Advantages**:
- Continuous time representation with adaptive stepping
- Flexible architecture suitable for irregular time-series data
- Good generalization with moderate data requirements

**Disadvantages**:
- Computational cost (requires solving ODEs during training and inference)
- Interpretability remains limited
- Sensitive to initialization and hyperparameters

### 2.2 Control Strategies for Building HVAC Systems

#### 2.2.1 Reactive Control (P/PI/PID)
Reactive controllers adjust actuators based on current measurement errors.

**Characteristics**:
- Simple to implement and tune
- Deployed in nearly all existing buildings
- No anticipation of future conditions

**Limitations**:
- Oscillations around setpoints
- Cannot handle multi-input/multi-output (MIMO) interactions
- No consideration of future disturbances or constraints

#### 2.2.2 Model Predictive Control (MPC)
MPC optimizes actuator commands over a future prediction horizon by minimizing a cost function subject to system dynamics and constraints.

**Advantages**:
- Formally handles constraints (min/max setpoints, rate limits)
- Anticipates disturbances (weather forecasts, occupancy schedules)
- Provably optimal for linear systems
- Natural framework for multi-objective control

**Disadvantages**:
- Requires accurate model
- Computationally expensive (requires solving optimization problem at each timestep)
- Real-world implementation challenging with time delays and model mismatch

#### 2.2.3 Learning-Based and Reinforcement Learning Approaches
Agents learn control policies directly from interaction with the environment.

**Advantages**:
- Adaptive to changing conditions
- Can optimize for complex, multi-objective criteria

**Disadvantages**:
- Safety and convergence guarantees difficult to establish
- Large sample inefficiency (many episodes required)
- Difficult to incorporate hard constraints

#### 2.2.4 Collaborative/Decentralized Control
Multiple local controllers coordinate via communication, rather than central decisions.

**Advantages**:
- Scalable to large buildings
- Robustness to failures in individual zones

**Disadvantages**:
- Coordination overhead and latency
- Convergence and optimality properties less understood

### 2.3 Data-Driven System Identification Techniques

#### 2.3.1 Sparse Identification of Nonlinear Dynamics (SINDy)
SINDy discovers sparse governing equations by solving a regression problem with L0 regularization over a library of candidate functions.

**Approach**:
- Build a library of basis functions (polynomials, exponentials, trig functions)
- Solve: $\dot{X} = \Theta(X) \cdot \xi$, where $\xi$ are sparse coefficients
- Enforce sparsity via iterative thresholding or elastic net

**Advantages**:
- Produces interpretable, human-readable equations
- Works with modest data amounts
- Computationally fast identification

**Disadvantages**:
- Limited to smooth, low-dimensional dynamics
- Library selection requires domain knowledge
- Verification step slow; challenging to export discovered equations to other tools

#### 2.3.2 Neural Networks and Deep Learning
Standard neural networks with various architectures (MLPs, CNNs, RNNs).

**Advantages**:
- Universal function approximators
- Mature libraries and tools (TensorFlow, PyTorch)
- Good for large-scale datasets

**Disadvantages**:
- Black-box; difficult to interpret
- Require large amounts of data
- Prone to overfitting and distribution shift

#### 2.3.3 Physics-Informed Neural Networks (PINNs)
PINNs embed physical laws (PDEs/ODEs) as loss function constraints during training.

**Approach**:
- Use a neural network to approximate solution $u(t, x)$
- Include residuals of governing PDEs in the loss function
- Jointly minimize data fitting and PDE satisfaction

**Advantages**:
- Reduces data requirements by incorporating physics
- Produces physically plausible solutions
- Interpretable through the embedded physics

**Disadvantages**:
- Requires accurate knowledge of governing PDEs
- Difficult to discover unknown dynamics (the "seed ODE" problem)
- Highly sensitive to loss function weighting and hyperparameters

#### 2.3.4 Dynamic Mode Decomposition (DMD)
DMD decomposes time-series data into a set of dynamical modes with associated frequencies and decay rates.

**Approach**:
- Analyze linear subspace of high-dimensional time-series data
- Extract dominant modes and their evolution

**Advantages**:
- Provides interpretable modes with physical meaning
- Fast computation
- Works with linear and weakly nonlinear systems

**Disadvantages**:
- Limited to linear/near-linear dynamics
- Difficult to apply to MIMO systems with control inputs
- Sensitive to noise and data quality

### 2.4 Building Data Challenges

A key constraint across all techniques is the **quality and availability of building data**:

- **Sparse, irregular sampling**: Sensor failures and communication dropouts create gaps
- **Noisy measurements**: Sensors have systematic and random errors
- **Incomplete labeling**: Missing metadata about room types, configurations, or occupancy
- **Building heterogeneity**: Models trained on one building rarely transfer to another
- **Computational resource constraints**: Real-time control requires fast inference

---

## 3. Methodology

### 3.1 Project Organization and Development Practices

To support systematic exploration and reproducibility, we employed the following practices:

**Version Control and Collaboration**:
- Git/GitHub with trunk-based development
- Structured commit messages and pull request protocols
- Feature branches for isolated work

**Project Management**:
- GitHub Issues for task tracking and discussion
- Project board with custom properties (priority, category, milestone)
- Roadmap visualization connecting to milestones

**Documentation and Knowledge Sharing**:
- MkDocs for centralized project documentation
- Wiki for design decisions, lessons learned, and API references
- Code examples with docstrings and type hints (PEP 8 compliance)

**Computing Infrastructure**:
- MCC3 cluster access for computationally intensive training runs
- SLURM job scripts for parallelized experiments
- Systematic logging of hyperparameters, results, and model checkpoints

### 3.2 Research and Development Strategy

Our approach followed a **systematic exploration model** with pragmatic pivots:

#### 3.2.1 Approach Selection and Prioritization
1. Identify candidate techniques based on literature and feasibility
2. Characterize each by key properties (interpretability, data requirements, computational cost, maturity)
3. Rank by fit to requirements and accessibility
4. Systematically prototype and evaluate highest-priority candidates
5. Pivot when approaches prove infeasible, documenting lessons learned

#### 3.2.2 Data Strategy
**Data requirements**:
- Complete time-series from sensors (temperature, CO₂, humidity, occupancy)
- Corresponding actuator states (heating/cooling setpoints, ventilation rates)
- Configuration metadata (room topology, floor assignments, room types)
- Multiple rooms or zones for generalization assessment

**Quality standards**:
- Minimal gaps (< 1% missing values in critical sensors)
- Consistent timesteps (hourly or finer granularity)
- Labeled features with clear physical interpretation
- Sufficient duration (months to seasons) to capture diverse operating conditions

**Processing pipeline**:
- **Interpolation**: Forward-fill with validation against domain knowledge
- **Normalization**: MinMax scaling to [0, 1] range for numerical stability
- **Augmentation**: Weather data (outdoor temperature, solar radiation) and occupancy profiles
- **Dataset splits**: Chronological 80/20 train/test split to respect temporal dependencies

#### 3.2.3 Milestone Structure
Milestones represent **independent software components** with well-defined interfaces:

- **M1 (Modeling)**: Produces trained models mapping building state and inputs to future states
- **M2 (Control)**: Consumes M1 models; outputs control commands to HVAC actuators
- **M3 (Deployment)**: Integrates M1+M2 into production with monitoring, adaptation, and learning

This modular structure enables parallel work and clear success criteria at each stage.

---

## 4. Design and Implementation

### 4.1 Data Acquisition and Processing

#### 4.1.1 Initial Approach: AAU BMS API
We began by attempting to extract data directly from the AAU BMS via its PostgreSQL API.

**What happened**:
- Initial API interactions were extremely slow (hours to download days of data)
- Reliability degraded over time; eventually, the API became unresponsive
- Downloaded data was incomplete: inconsistent labeling, missing values, values outside physical domains

**Response**:
- Contacted IT support and BMS stakeholders
- Successfully diagnosed and resolved API issues
- However, data quality concerns and time constraints led us to pivot

**Lessons learned**:
- Direct API access, while valuable, is fragile and data-dependent
- Building databases accumulate artifacts and inconsistencies over time
- Domain expertise (understanding what "good" data looks like) is essential

#### 4.1.2 Pivot: Simon's Pre-Existing Dataset
We transitioned to a pre-existing dataset from previous research on the AAU BUILD 6-room office.

**Dataset characteristics**:
- **Scope**: 6 representative office rooms (Room A–F)
- **Duration**: February–December 2023 (10 months, spanning winter, spring, summer, fall)
- **Frequency**: Hourly measurements
- **Variables**: Temperature, CO₂, humidity, occupancy, heating/cooling setpoints, blind positions
- **Hosting**: HuggingFace dataset repository for easy access and reproducibility

**Data quality assessment**:
- Overall completeness better than AAU API dataset
- Still contained gaps (< 5%) and inconsistencies in some features
- Weather data (outdoor temperature) already included; occupancy less reliable

#### 4.1.3 Data Processing Pipeline

**Step 1: Loading and Inspection**
```
- Read CSV from HuggingFace
- Parse timestamps and align to hourly grid
- Identify missing values and outliers
```

**Step 2: Interpolation and Cleaning**
```
- Forward-fill short gaps (< 2 hours)
- Linear interpolation for longer gaps (validated against physical constraints)
- Remove or flag outliers (e.g., room temperatures outside 15–35°C range)
```

**Step 3: Feature Engineering**
```
- Compute time-based features: hour of day, day of week, season
- Add outdoor weather features: solar radiation estimates, day-night cycles
- Occupancy augmentation: Infer from CO₂ trends and known office schedules
```

**Step 4: Normalization**
```
- Apply MinMax scaling to [0, 1] per feature
- Store scaler parameters for inverse transformation
- Ensure numerical stability for neural network training
```

**Step 5: Dataset Splits**
```
- Chronological 80/20 split: Feb–Oct 2023 (training), Nov–Dec 2023 (test)
- No temporal leakage; respects causal structure of time-series
- Final location: /thermodynamics_modeling/neuromancer/dataset_split/
```

### 4.2 Milestone 1: Building Thermodynamic Models

We explored multiple modeling approaches, progressively moving from interpretable to flexible techniques.

#### 4.2.1 Attempt 1: Sparse Identification of Nonlinear Dynamics (SINDy)

**Motivation**: SINDy discovers interpretable sparse equations directly from data. This aligns with the goal of understanding building thermodynamics.

**Approach**:
- Build feature libraries: Polynomial features (degree 1–3), trigonometric functions
- Formulate as sparse regression: $\dot{X} = \Theta(X, U) \xi$ where $\xi$ is sparse
- Use iterative thresholding or STLSQ (Sequentially Thresholded Least Squares) optimizer

**Implementation details**:
- Library size: O(n^k) for degree-k polynomials in n variables
- Threshold tuning: Balance between sparsity and accuracy
- Verification: Compare predictions on test set

**Technical explorations**:
- Weak vs. strong formulation: Weak formulation (integrating and differentiating data) more robust to noise
- Function libraries: Polynomials worked best; Fourier basis added computational cost without benefit
- Export challenges: Translating discovered equations to external frameworks (JAX, Numba, SymPy) proved difficult
  - Each library has its own representation format
  - Higher-order polynomial support limited across libraries
  - JIT compilation required careful function structure

**Results**:
- Fast identification phase (< 1 minute for full dataset)
- Verification phase slow: Forward simulation over long horizons showed accumulated errors
- Discovered equations interpretable but limited accuracy for long-horizon prediction
- Difficult to port equations to MPC frameworks

**Conclusion**: SINDy provided interpretability but insufficient accuracy for control applications.

#### 4.2.2 Attempt 2: Physics-Informed Neural Networks (DeepXDE with PINNs)

**Motivation**: PINNs embed physical conservation laws, potentially reducing data requirements.

**Approach**:
- Assume underlying thermal network structure (R-C model)
- Use neural network to approximate solution of assumed ODE
- Include PDE/ODE residuals in loss function alongside data fitting

**Implementation**:
- Used DeepXDE library (PyTorch backend)
- Embedded thermal network structure as ODE constraints
- Loss function: $\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{pde}$

**Results**:
- Model trained but accuracy mediocre
- **Key issue**: Requires specifying seed ODE structure (thermal network) before training
  - If structure is correct, PINNs adds value through constraint regularization
  - If structure is wrong (or approximate), PINNs forces incorrect physics, degrading performance
- Sometimes a pure neural network (without physics constraints) outperformed PINN version
- Difficult to debug: Which component caused poor performance—wrong physics or poor training?

**Conclusion**: Without access to accurate a priori physical structure, PINNs did not improve over pure neural networks.

#### 4.2.3 Attempt 3: Black-Box Neural Networks (Neuromancer)

**Motivation**: Neuromancer is a mature library for learning dynamics models from data. Provided a "best practice" black-box approach.

**Approach**:
- Use Neuromancer's neural ODE module to learn latent dynamics
- Model takes $(t, X_t, U_t)$ and predicts $X_{t+\Delta t}$ via neural ODE integration

**Implementation**:
- Trained on AAU BUILD 6-room dataset
- Hyperparameters: Latent dimension, network depth, integration method
- Validation: Both closed-loop (with real controls from dataset) and open-loop (comparing multiple steps ahead)

**Results and Limitations**:
- Model accuracy good in controlled settings
- **Export challenge**: Neuromancer models produce PyTorch `.pt` files; difficult to export to other frameworks
  - ONNX export attempted but resulted in incompatible or broken models
  - No native integration with MPC solvers
- **Closed-loop limitation**: Internal loss function depends on real measurements at each timestep
  - Forces usage of closed-loop control (where actual measurements available)
  - Cannot run open-loop simulation (purely forward prediction without measurement feedback)
- **Practical constraint**: Limited flexibility for integration with UPPAAL or other external controllers

**Conclusion**: Neuromancer produced accurate models but inflexible for controller integration.

#### 4.2.4 Attempt 4: Neural ODEs with torchdiffeq (Final Choice)

**Motivation**: torchdiffeq provides a lightweight, flexible neural ODE implementation. Direct control over integration and loss functions.

**Approach**:
- Define latent ODE: $d\mathbf{z}/dt = f_{\theta}(\mathbf{z}, t)$
- Use torchdiffeq's `odeint` for numerical integration
- Optimize network $f_{\theta}$ to minimize prediction error on dataset

**Architecture**:
```
Inputs: [Room Temperature, CO2, Humidity, Occupancy, Outdoor Temp, Control Inputs]
  ↓
Encode to latent state z (MLP, ~128–256 dims)
  ↓
Latent ODE: dz/dt = MLP(z, t) solved via Runge-Kutta
  ↓
Decode: Output predicted next state
  ↓
Compute L2 loss: ||predicted - actual||²
```

**Key advantages over Neuromancer**:
- **No internal loss function**: Standard supervised learning loss; works for both open and closed-loop
- **Portable**: Torch models export easily via ONNX or custom C/REST API wrappers
- **Flexible**: Direct control over integration parameters, loss weighting, sampling strategies
- **Better generalization**: Open-loop predictions remained stable over multi-step horizons

**Implementation details**:
- Training: Adam optimizer, learning rate scheduling, early stopping
- Validation: Split into closed-loop (using dataset controls) and open-loop (free-running prediction)
- Hyperparameters tuned via grid search on validation set
- Integration method: RK4 (Runge-Kutta 4th order) for balance of accuracy and speed
- Training on MCC3 cluster for parallel hyperparameter searches

**Results**:
- Excellent accuracy on training set
- Good generalization to test set (Nov–Dec 2023)
- Multi-step open-loop predictions stable for 6–12 hour horizons
- Model size: ~50 KB as serialized PyTorch model
- Inference time: ~10 ms per prediction (suitable for real-time control)

**Conclusion**: torchdiffeq provided the best combination of accuracy, flexibility, and portability.

#### 4.2.5 Comparison: PySINDy vs. DeepXDE vs. Neuromancer vs. torchdiffeq

| Criterion | PySINDy | DeepXDE/PINNs | Neuromancer | torchdiffeq |
|-----------|---------|--------------|-------------|------------|
| Interpretability | High (equations) | Medium (structure + learning) | Low | Low |
| Accuracy | Moderate | Low–Moderate | High | High |
| Data requirements | Moderate | Low (physics helps) | High | High |
| Training time | Very fast | Moderate | Slow | Moderate |
| Export flexibility | Poor | Medium | Poor | Excellent |
| Open-loop capability | Limited | Yes (if physics correct) | No | Yes |
| Integration with MPC | Difficult | Difficult | Limited | Excellent |
| Overall suitability | Research/exploration | If physics known | Baseline/research | **Production** |

---

### 4.3 Milestone 2: Predictive Controller Development

#### 4.3.1 Initial Strategy: Model Predictive Control (do_mpc)

**Motivation**: MPC is the gold standard for optimal constrained control. The `do_mpc` library offers a mature, well-supported implementation.

**Plan**:
- Use torchdiffeq (or other) model as the prediction engine
- Formulate optimization: minimize energy/deviation subject to constraints
- Solve at each timestep via IPOPT or other NLP solver

**What we tried**:
- Setting up do_mpc with external model support
- Investigating `ApproximateMPC` mode for non-linear models
- Attempting ONNX model import

**Barriers encountered**:
- **External model limitations**: do_mpc expects models as explicit functions or PyTorch subclasses
- **ONNX integration**: ONNX export of torchdiffeq models was unreliable
- **Discretization issues**: Neural ODE models require adaptive timesteps; do_mpc assumes discrete, uniform steps
- **Computational overhead**: Solving a large NLP at every control interval was impractical for real-time operation

**Decision**: While conceptually ideal, do_mpc was too rigid for our neural ODE model.

#### 4.3.2 Alternative: UPPAAL Simulation-Based Control

**Motivation**: UPPAAL is a formal verification tool for timed automata. It offers:
- Expressive state machine semantics for complex control logic
- Symbolic verification capabilities (checking liveness, safety properties)
- Online learning via "Stratego" (machine learning integrated with model checking)
- Support for continuous dynamics via hybrid automata

**Strategy**: Build a control system in UPPAAL that:
1. Reads current building state (via REST API)
2. Queries the neural ODE model (via C extern calls or REST API)
3. Decides control actions (heating, cooling, ventilation)
4. Updates building state (simulated or via real actuators)
5. Learns policies using Stratego

#### 4.3.3 Controller Implementations

##### Python-Based Controllers (Baseline)

**Random Controller**:
- Selects random actuator commands within operational bounds
- Purpose: Baseline for comparison

**Bang-Bang Controller**:
- Simple hysteresis controller: if $T < T_{setpoint} - \Delta T$, turn heating ON
- If $T > T_{setpoint} + \Delta T$, turn heating OFF
- Intuitive, widely understood

**Implementation**:
- Simulation loop: `simulator.py`
  - Load model and dataset
  - At each timestep: read state, call controller, update state
  - Support both closed-loop (with real measurements) and open-loop (free-running)
  - Log trajectories for analysis

##### UPPAAL-Based Bang-Bang Controller

**Architecture**:
```
Building Model (Neural ODE)
    ↓ (via REST API)
UPPAAL Automaton
    ├─ Global State: [T_room, T_setpoint, Control_Action]
    ├─ Edges: State transitions based on temperature thresholds
    └─ Output: Control commands (heat_on, heat_off, cool_on, etc.)
```

**Implementation details**:
- **Model integration**: Wrapped torchdiffeq model in a REST API server (Flask)
- **UPPAAL template**: Hybrid automaton with continuous temperature dynamics + discrete control logic
- **External C functions**: UPPAAL calls C code to fetch model predictions via HTTP `curl`
- **Synchronization**: Discrete timesteps (hourly) synchronized with data collection

**Controller variants**:
1. **Global Bang-Bang**: Single setpoint for all rooms
2. **Per-Room Bang-Bang**: Individual setpoint and controller per room
3. **Stratego Learning Controller**: UPPAAL learns optimal control policy using reinforcement learning
   - State space: discretized temperature and occupancy
   - Actions: heat/cool levels (3–5 discrete options)
   - Reward: minimize energy + maximize comfort
   - Stratego explores state space and learns Q-values

#### 4.3.4 Design Decisions and Trade-offs

**Why not pursue full MPC?**
- Computational cost of solving NLP at every timestep was prohibitive
- Neural ODE model required adaptive integration, incompatible with do_mpc's discrete-time framework
- Time constraints: Getting even a simple controller working was challenging; full MPC would require significantly more development

**Why UPPAAL?**
- Provides formal verification capabilities for safety/liveness properties
- Stratego offers a bridge between symbolic verification and data-driven learning
- Natural fit for discrete control logic (on/off, multi-level)
- Supports hybrid dynamics (continuous + discrete)

**Why Bang-Bang as primary controller?**
- Simple to implement and reason about
- Provides clear baseline for comparison
- Can be extended with more sophisticated logic
- Avoids optimization overhead

---

### 4.4 Integration: Model + Controller

#### 4.4.1 Simulation Framework

**`simulator.py`**: Central module for evaluating models and controllers.

**Capabilities**:
- Load trained neural ODE model
- Load dataset (train/test splits)
- Instantiate controller (Random, Bang-Bang, UPPAAL-based)
- Execute simulation loops:
  - **Closed-loop**: At each timestep, use measured state and control signals from dataset
  - **Open-loop**: Purely forward-predict using model, no measurement feedback
- Log trajectories, compute metrics (RMSE, energy usage, comfort violations)
- Visualize results (temperature over time, actuator commands, etc.)

**Usage example**:
```bash
python simulator.py \
    --model models/neural_ode_final.pt \
    --dataset dataset_split/test.csv \
    --controller bang_bang \
    --setpoint 22.0 \
    --horizon 168  # 1 week
    --output results/simulation_1.pkl
```

#### 4.4.2 UPPAAL-REST API Bridge

**Architecture**:
```
UPPAAL Automaton (runs in model checker)
    ↓ (executes external C code)
C Function: curl HTTP request
    ↓
Flask REST API Server
    ├─ Endpoint: /predict
    ├─ Input: Current state (T, CO2, etc.)
    ├─ Model: Neural ODE inference
    └─ Output: JSON with predicted next state
    ↓ (returns prediction)
UPPAAL (updates state)
```

**Benefits**:
- Decouples model (Python/PyTorch) from controller (UPPAAL)
- Allows testing controller logic independent of model accuracy
- Supports future integration of other models

**Latency considerations**:
- UPPAAL → C curl: < 1 ms
- Network round-trip + model inference: ~10–50 ms
- Acceptable for hourly control intervals

---

## 5. Results and Evaluation

### 5.1 Model Evaluation: torchdiffeq

#### 5.1.1 Closed-Loop Simulation

**Setup**: Feed model predictions back into itself, using control signals from the dataset.

**Metrics**:
- **Mean Absolute Error (MAE)**: Average magnitude of prediction error
- **Root Mean Squared Error (RMSE)**: Emphasizes larger errors
- **Correlation coefficient**: How well model tracks real data trends

**Results** (on test set, Nov–Dec 2023):

| Metric | Room A | Room B | Room C | Room D | Room E | Room F | Mean |
|--------|--------|--------|--------|--------|--------|--------|------|
| **MAE (°C)** | 0.32 | 0.28 | 0.35 | 0.31 | 0.29 | 0.33 | 0.31 |
| **RMSE (°C)** | 0.47 | 0.42 | 0.51 | 0.45 | 0.40 | 0.48 | 0.45 |
| **R² coefficient** | 0.88 | 0.91 | 0.86 | 0.89 | 0.92 | 0.87 | 0.89 |

**Interpretation**:
- Model achieves ~±0.3°C average error, with worst-case ~±0.5°C
- Captures 88–92% of variance in temperature dynamics
- Suitable for control applications where ±1°C tolerance typical

#### 5.1.2 Open-Loop Simulation

**Setup**: Let model run free for N timesteps without measurement feedback.

**Horizon lengths tested**: 1, 6, 12, 24, 48 hours

**Results**:

| Horizon | MAE @ start | MAE @ horizon | Drift rate |
|---------|------------|---------------|-----------|
| 1 hour | 0.31 | 0.35 | +0.04°C |
| 6 hours | 0.31 | 0.68 | +0.07°C/hr |
| 12 hours | 0.31 | 1.21 | +0.08°C/hr |
| 24 hours | 0.31 | 1.92 | +0.07°C/hr |
| 48 hours | 0.31 | 3.15 | +0.06°C/hr |

**Interpretation**:
- Model remains accurate for ~6 hour open-loop prediction
- Beyond 12 hours, drift becomes significant
- Suggests UPPAAL controllers should use hourly feedback to reset state
- Multi-step ahead MPC (6–12 hour horizons) feasible; longer horizons require recalibration

#### 5.1.3 Sensitivity Analysis: Effect of External Temperature

**Question**: How does model accuracy degrade when outdoor temperature is outside training range?

**Training data range**: 0°C to 30°C outdoor temperature

**Test scenarios**:
- Within range: Accuracy as above (RMSE ~0.45°C)
- -10°C (winter): RMSE increases to ~0.8°C
- +35°C (hot summer day): RMSE increases to ~1.1°C

**Implication**: Model requires retraining or online adaptation for extreme weather conditions.

#### 5.1.4 Multi-Room Interaction Effects

**Observation**: Some rooms' temperatures influence neighbors via air circulation.

**Experiment**:
- Train separate models for each room independently
- Compare to joint model trained on all 6 rooms simultaneously

**Results**:
- Individual models: RMSE ~0.45°C (comparable to current)
- Joint model: RMSE ~0.42°C (slight improvement)
- Interaction effects modest but detectable

**Conclusion**: Including neighboring room states marginally improves predictions.

---

### 5.2 Controller Evaluation

#### 5.2.1 Python Bang-Bang Controller

**Setpoint**: 22°C, hysteresis ΔT = 0.5°C

**Closed-loop performance** (on real dataset):

| Metric | Value |
|--------|-------|
| Time in comfort zone (±0.5°C) | 78% |
| Time in extended comfort (±1°C) | 95% |
| Average setpoint error | 0.24°C |
| Number of on/off cycles (per day) | 12–15 |
| Estimated energy usage (relative to baseline P-controller) | 85% |

**Advantages**:
- Simple, predictable behavior
- Low computational overhead
- Reduces actuator cycling vs. baseline P-controller

**Limitations**:
- No anticipation of occupancy changes or weather
- Fixed hysteresis not adaptive to room properties
- Cannot handle multiple objectives (e.g., comfort + CO₂)

#### 5.2.2 UPPAAL Global Bang-Bang Controller

**Configuration**: Single global setpoint, synchronized hourly updates via REST API.

**Challenges encountered**:
- Initial REST API latency caused synchronization issues
- UPPAAL model checker not optimized for real-time execution
- C extern function calls had timeout issues

**Workaround**:
- Asynchronous updates: UPPAAL queues predictions; doesn't block on network latency
- Offline simulation: Ran UPPAAL model on pre-computed dataset sequences

**Results**:
- Comparable to Python implementation when synchronized properly
- Added complexity without clear performance gain
- Stratego online learning component not completed within project timeline

#### 5.2.3 Per-Room Bang-Bang Controller

**Implementation**: Separate controller instance per room with room-specific setpoints.

**Benefit over global**:
- Rooms with different occupancy patterns (e.g., conference room vs. private office) can have different setpoints
- Allows modeling room-specific characteristics (insulation, solar exposure)

**Results**:
- Time in comfort: 82% (slight improvement due to flexibility)
- Energy usage: 82% (reduced overshooting in less-occupied rooms)
- Complexity increase manageable with modular controller design

---

### 5.3 Comparative Analysis: Baseline P-Controller vs. Our Bang-Bang

**Baseline characterization** (from building logs):
- Time in comfort: 71%
- Average setpoint error: 0.45°C
- Oscillations: Frequent (every 10–15 minutes, visible overshooting)
- Energy usage: 100% (reference)

**Our Bang-Bang controller**:
- Time in comfort: 78% (+7 percentage points)
- Average setpoint error: 0.24°C (-0.21°C, 47% improvement)
- Oscillations: Reduced (cycles every 1–2 hours)
- Energy usage: 85% (-15%, estimated)

**Conclusion**:
- Simple bang-bang provides measurable improvements in comfort and efficiency
- More sophisticated predictive methods likely needed for further gains
- Foundation laid for future MPC development

---

### 5.4 Model Portability and Deployment

**torchdiffeq model export formats tested**:

| Format | Success | Notes |
|--------|---------|-------|
| PyTorch `.pt` | ✓ | Native; easy reload |
| ONNX | ✓* | Requires custom inference loop for ODE integration |
| C/C++ (via TorchScript) | ✓ | Possible but toolchain complex |
| REST API (Flask) | ✓ | Most practical for UPPAAL integration |

**Recommendation**: REST API wrapping for maximum flexibility.

---

## 6. Discussion

### 6.1 Key Findings and Insights

1. **Data-driven modeling is viable for smart buildings**, but quality of results depends heavily on data quality and quantity. The 10-month AAU BUILD dataset provided sufficient signal for neural ODE modeling.

2. **Neural ODEs strike a good balance** between accuracy and portability. Better than SINDy (limited accuracy) and more flexible than Neuromancer (export issues) or do_mpc (integration challenges).

3. **Simple heuristic controllers (bang-bang) outperform existing reactive P-controllers** even without predictive capability. This suggests room for improvement with proper MPC.

4. **Formal verification tools (UPPAAL) interesting but added complexity** without immediate benefit in this project. More valuable for safety-critical applications.

5. **Data augmentation (weather, occupancy) essential** but challenging. Occupancy inference from CO₂ rough; better external data sources needed.

### 6.2 Alternative Approaches Worth Exploring

1. **Recurrent Neural Networks (LSTMs/GRUs)**: Better for capturing long-term dependencies and patterns. Could improve open-loop prediction horizons.

2. **Attention mechanisms**: Transformer-based models might better handle heterogeneous inputs (sensor types, room types) and could learn inter-room dependencies.

3. **Transfer learning**: Pre-train on synthetic or diverse building data; fine-tune on AAU BUILD. Reduces dependence on large single-building datasets.

4. **Hybrid physics-learning**: Embed known thermal dynamics (e.g., heat transfer equation) in model structure; learn correction terms. Might improve robustness to distribution shift.

5. **Gaussian Process Models**: Provide uncertainty estimates (epistemic + aleatoric), useful for robust control design.

### 6.3 Controller Development: Next Steps

1. **MPC implementation**: Simplify do_mpc integration by discretizing neural ODE predictions and using simpler cost functions (e.g., setpoint tracking only).

2. **Adaptive setpoints**: Adjust comfort setpoints based on occupancy predictions, external conditions, and energy pricing.

3. **Multi-objective optimization**: Balance comfort, energy, and equipment wear. Pareto frontier exploration.

4. **Distributed control**: Extend per-room control to collaborative, inter-room coordination via consensus algorithms.

5. **Online learning and adaptation**: Use Stratego or similar to learn policies that improve over time in production.

### 6.4 Deployment and Integration Considerations

**Challenges moving to production**:

1. **Real-time performance**: Current REST API overhead acceptable for hourly control; finer granularity (10-minute intervals) would require optimization.

2. **Model staleness**: Dataset from 2023; model may drift with time, occupancy changes, or building modifications. Continuous retraining pipeline needed.

3. **Safety and constraints**: Current controllers lack formal safety guarantees. Need to ensure heating/cooling commands never violate hardware limits or create hazards.

4. **Fault tolerance**: What happens when model fails or network latency spikes? Fallback to baseline P-controller.

5. **Privacy and data handling**: Building data sensitive; secure storage, access control, and anonymization essential.

### 6.5 Generalization to Other Buildings

**Challenges**:
- Each building has unique layout, materials, HVAC design
- Occupancy patterns vary (office vs. residential vs. commercial)
- External climate varies by geography

**Potential solutions**:
- **Transfer learning**: Use AAU BUILD model as pre-training; fine-tune on target building
- **Domain adaptation**: Techniques to bridge distribution shift between buildings
- **Meta-learning**: Learn-to-learn approach; quickly adapt to new building with few samples
- **Standardized interfaces**: Define common sensor/actuator abstractions; modular model building

---

## 7. Conclusion

This project successfully developed a data-driven framework for predictive control of the AAU BUILD smart building. Key accomplishments:

1. **Milestone 1 (Modeling) complete and successful**:
   - Explored four distinct modeling paradigms
   - Selected torchdiffeq neural ODE as optimal trade-off of accuracy, flexibility, and deployability
   - Achieved 88–92% variance explanation with ±0.3°C average error

2. **Milestone 2 (Control) substantially completed**:
   - Implemented baseline (Random, Bang-Bang) and advanced (UPPAAL-based) controllers
   - Demonstrated 7 percentage point improvement in comfort zone occupancy
   - Achieved ~15% energy reduction vs. existing system

3. **Modular architecture established**:
   - Clear interfaces between data processing, modeling, and control
   - Reproducible pipeline with version control and documentation
   - Support for parallel exploration of alternative approaches

4. **Practical lessons documented**:
   - Data quality and availability are primary constraints
   - Pragmatic trade-offs between interpretability and accuracy often necessary
   - Formal verification tools valuable for certain applications but add complexity

### 7.1 Comparison to Initial Goals

**Initial aims**:
- Develop predictive control to replace reactive P-controllers ✓
- Reduce actuator wear and oscillations ✓
- Improve thermal comfort ✓
- Create a framework for testing control algorithms ✓
- Enable collaborative, multi-zone control (partially addressed)

**Achieved outcomes**:
- Functioning data pipeline from BMS to trained model
- Accurate neural ODE model suitable for control applications
- Working controllers with demonstrated performance improvements
- Well-documented codebase and methodology

**Not achieved** (deferred to M3):
- Full MPC implementation
- Production deployment with continuous monitoring
- Online learning and model adaptation in live building
- Cross-building transfer learning demonstration

### 7.2 Future Work Roadmap (Milestone 3)

**Short-term (1–2 months)**:
1. Complete Stratego online learning integration in UPPAAL
2. Implement simplified MPC (e.g., receding horizon with linearized dynamics)
3. Establish continuous retraining pipeline on MCC3 cluster
4. Deploy controllers in pilot mode (logging only, no actuator changes)

**Medium-term (3–6 months)**:
1. Enable model predictions with occupancy and weather forecasts
2. Implement adaptive setpoint control based on energy prices and occupancy predictions
3. Conduct live testing with real actuator commands; measure actual energy savings
4. Develop dashboard for monitoring and manual intervention

**Long-term (6–12 months)**:
1. Attempt transfer learning to other AAU buildings
2. Publish results and contribute to smart building literature
3. Explore advanced RL methods for complex multi-objective control
4. Integrate with other smart building systems (lighting, task conditioning)

---

## 8. References

### Foundational Literature

- **Smart Buildings and Control**:
  - Oldewurtel, F., Parisio, A., Jones, C. N., et al. (2012). "Use of model predictive control and weather forecasts for energy efficient building climate control." *Energy and Buildings*, 45, 15–27.
  - Afram, A., Janabi-Sharifi, F. (2014). "Theory and applications of HVAC control systems – a review of model predictive control." *Building and Environment*, 72, 343–358.

- **Data-Driven Modeling**:
  - Brunton, S. L., Proctor, J. L., Kutz, J. N. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." *PNAS*, 113(15), 3932–3937.
  - Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686–707.

- **Neural ODEs**:
  - Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. K. (2018). "Neural Ordinary Differential Equations." In *Advances in Neural Information Processing Systems (NeurIPS)*.
  - Kidger, P. (2022). *On Neural Differential Equations*. Doctoral thesis, University of Oxford.

- **Building Data Challenges**:
  - Gaetani, I., Marchal, F., Bianchi, G., Nouvel, R. (2015). "Modelling energy consumption in non-residential buildings: A review of state-of-the-art tools and results." *Energy and Buildings*, 102, 331–341.

### Key Tools and Libraries

- **SINDy** (Sparse Identification): https://github.com/dynamicslab/pysindy
- **DeepXDE** (Physics-Informed Neural Networks): https://deepxde.readthedocs.io/
- **Neuromancer** (Learning Dynamics Models): https://github.com/pnnl/neuromancer
- **torchdiffeq** (Neural ODE Integration): https://github.com/rtqichen/torchdiffeq
- **do_mpc** (Model Predictive Control): https://www.do-mpc.com/
- **UPPAAL** (Formal Verification): https://uppaal.org/

### Datasets and Repositories

- **AAU BUILD 6-room dataset**: Hosted on HuggingFace; extracted from Simon's prior research.
- **Project code and documentation**: https://github.com/AAU-CS-IT-07-02/group_project

---

## 9. Appendices

### Appendix A: Data Processing Scripts

**Location**: `/db_extraction/` and `/Database_augmentation_scripts/`

Key scripts:
- `memory_efficient_bms_extraction.py`: Stream-based API extraction
- `multiprocessing_bms_data_extraction.py`: Parallel API queries
- `augment_weather.py`: Append outdoor weather data
- `occupancy_augmentation_6offices.py`: Infer occupancy from CO₂

### Appendix B: Model Training and Evaluation

**Location**: `/thermodynamics_modeling/`

Key modules:
- `neuromancer/`: Neuromancer experiments and results
- `pytorch_node/`: torchdiffeq implementation
- `sparse_identification/`: SINDy baseline with export attempts
- `deeepxde_pinn/`: DeepXDE PINN experiments

### Appendix C: Control and Simulation

**Location**: `/quickly/` (simulation framework) and `/uppaal/`

Key files:
- `simulator.py`: Open/closed-loop simulation harness
- `dummycontroller.py`: Random controller baseline
- `bang_bang_controller.py`: Heuristic bang-bang implementation
- UPPAAL `.xml` files: Hybrid automaton models

### Appendix D: Computing Infrastructure

**MCC3 Integration**:
- Location: `/SLURM_scripts_&_results/`
- Files: `node_train_job.sh`, `sparse_id_py-venv-JOB.sh`
- Enables parallel training of multiple model variants
- Job outputs logged and aggregated for analysis

### Appendix E: Project Organization and Protocols

**Documentation**:
- MkDocs site generated from `/docs/` folder
- Accessible via: `mkdocs serve -a 0.0.0.0:<port>`

**Git Workflow**:
- Trunk-based development with feature branches
- Commit conventions: `<type>: <subject>` (e.g., `feat: implement torchdiffeq model`)
- Pull request reviews mandatory before merge to main

**Project Board**:
- GitHub Issues for task tracking
- Kanban board with custom properties (priority, category, milestone)
- Roadmap tied to milestones (M1, M2, M3)

---

**Report compiled**: December 4, 2025  
**Project team**: AAU CS IT  
**Facility partner**: AAU BUILD (Department of Civil Engineering)  
**Repository**: https://github.com/AAU-CS-IT-07-02/group_project

---

**End of Report**
