# Project Report Summary: Intelligent Building Management System through Data-Driven Thermodynamics Modeling and Control

## 1. Project Overview

**Project Title:** Intelligent Building Management System through Data-Driven Thermodynamics Modeling and Control

**Course:** Secure, Scalable and Useful Systems  
**Semester:** 7th Semester, 2025  
**Institution:** Aalborg University (AAU), Department of Computer Science

### Team Members
- Dragomir Matei Mihai
- Eduard Brahas
- Monda Rareș
- Pedro Felizardo Pedroso Carreira Lima
- Nasik Ali Khan
- Beltrán Aceves Gil

### Supervisors and Collaborators
- **Supervisor:** Marco Muñiz
- **Collaborators:** Rasmus Lund Jensen and Simon Pommerencke Melgaard

---

## 2. Project Context

The project is developed in collaboration with the **Department of Civil Engineering (BUILD)** at Aalborg University. The BUILD building serves as a Living Lab for building performance research.

### TMV 23 Building Characteristics
- **Location:** Thomas Manns Vej 23, Aalborg, Denmark
- **Size:** ~9,000 m² across five floors + one basement
- **Occupancy:** ~150 staff employees and 600 students
- **Functions:** Offices, meeting rooms, classrooms, workshops, laboratories

### Building Infrastructure
- **Sensor Network:** Light levels, temperature, CO₂, occupancy, humidity
- **HVAC Systems:** 14 ventilation systems, radiators, radiative panels, 3 cooling units
- **Building Management System (BMS):** Schneider EcoStruxure
- **Sustainable Features:** Photovoltaic panels, district heating

---

## 3. Problem Statement

Current building control systems face several challenges:
1. **Independent Operation:** Controllers work without collaboration or integration
2. **Reactive Control:** Systems respond only to current conditions without predictive capabilities
3. **System Oscillations:** Constant adjustments lead to energy waste and discomfort
4. **No Coordination:** Lack of coordinated control strategies causes inefficiencies

---

## 4. Proposed Solution

The project proposes a **collaborative and predictive control system** with three modular components:

### 4.1 Building Thermodynamics Simulation

**Primary Method: SINDy (Sparse Identification of Nonlinear Dynamics)**
- Approximates nonlinear systems as sparse combinations from a function library
- Implemented using [PySINDy](https://github.com/dynamicslab/pysindy)
- Supports both linear and nonlinear state-space representations

**Backup Method: DMD (Dynamic Mode Decomposition)**
- Approximates system dynamics through linearization of evolution modes
- Implemented using [PyDMD](https://github.com/PyDMD/PyDMD)

**Alternative Methods Explored:**
1. **Physics-Informed Neural Networks (PINNs)** - Using DeepXDE
2. **Neural ODEs (Neural Ordinary Differential Equations)** - Using NeuroMANCER and PyTorch/torchdiffeq
3. **RC Network Models** - Lumped parameter thermal models

### 4.2 Smart Predictive Controller (MPC)

**Implementation:** Model Predictive Control using [do-mpc](https://www.do-mpc.com/en/latest/)

**Features:**
- Predictive capability using thermodynamic model
- Multi-actuator coordination (HVAC, lighting, ventilation)
- Constraint handling for safety guarantees
- Configurable prediction and control horizons
- Weighted cost functions for energy, comfort, and actuator wear

### 4.3 Control Panel Interface (Future Work)

- Building profile selection mechanism
- Automated nighttime temperature adjustments
- Room booking-based comfort optimization
- Weather forecast integration
- Planned UPPAAL integration for verification

---

## 5. Technical Implementation

### 5.1 Repository Structure

```
group_project/
├── Database/                          # Database storage
├── Database_augmentation_scripts/     # Weather and occupancy augmentation
├── SLURM_scripts_&_results/          # HPC job scripts
├── alternative_methods/               # RC network and parameter estimation
├── db_extraction/                     # BMS data extraction tools
├── do_MPC/                           # MPC controller implementation
├── docs/                             # MkDocs documentation
├── thermodynamics_modeling/          # Core modeling components
│   ├── sparse_identification/        # PySINDy implementation
│   ├── dynamic_mode_decomposition/   # PyDMD implementation
│   ├── deeepxde_pinn/               # PINN implementation
│   ├── neuromancer_node/            # NeuroMANCER NODE
│   ├── pythorch_node/               # PyTorch NODE with torchdiffeq
│   └── data_fragmentation/          # Data preprocessing
└── uppaal/                           # UPPAAL integration
    └── http_to_uppaal/              # HTTP bridge for NN inference
```

### 5.2 Data Pipeline

**Data Sources:**
- AAU BUILD BMS API (4+ years of sensor data)
- Weather data augmentation (Aalborg hourly data)
- Occupancy data augmentation for 6 office rooms

**Data Processing:**
1. **Extraction:** Memory-efficient and multiprocessing BMS extraction
2. **Cleaning:** Handling missing values, timestamp synchronization
3. **Augmentation:** External weather and occupancy data
4. **Fragmentation:** Room-by-room, time-windowed datasets
5. **Normalization:** MinMax, Standard, and Robust scaling

### 5.3 Thermodynamics Modeling

**SINDy Configuration:**
- Feature libraries: Polynomial, Fourier, Identity
- Optimizer: STLSQ (Sequentially Thresholded Least Squares)
- Configurable sparsity threshold and regularization
- Support for control inputs (SINDy with control)

**Model Formulation:**
```
dX/dt = f(X, U)
```
Where:
- **X** = Building state (temperatures, CO₂ levels)
- **U** = Control inputs (heating/cooling setpoints, ventilation)
- **f** = Function discovered from data

### 5.4 Neural ODE Implementation

**PyTorch NODE Architecture:**
- **Encoder:** Maps observed state to latent initial state
- **ODE Function:** Neural network defining `dz/dt = f(z, u)`
- **Decoder:** Maps latent states to physical outputs
- **Integration:** RK4 or adaptive solvers (dopri5)

**Training Configuration:**
- Sequence horizon (H) for time windows
- Latent space dimensions
- Z-score normalization
- MSE loss with trajectory and one-step tracking

### 5.5 UPPAAL Integration

**Architecture:**
```
UPPAAL Model → C Wrapper (libcurl) → FastAPI Server → Neural ODE Model
```

**Components:**
1. **Python Server:** FastAPI/Uvicorn serving Neural ODE predictions
2. **C Wrapper:** libcurl-based HTTP client for UPPAAL
3. **JSON Communication:** cJSON parsing for request/response

---

## 6. Development Infrastructure

### 6.1 Documentation

- **Platform:** MkDocs with Material theme
- **Features:** 
  - Auto-generated API docs (mkdocstrings)
  - LaTeX math support (MathJax)
  - Code syntax highlighting
  - Admonitions and superfences

### 6.2 High-Performance Computing

- **Platform:** SLURM cluster (mcc3)
- **Partitions:** naples, dhabi, rome
- **Job Scripts:** Python virtual environment setup, dependency installation
- **Output Management:** Per-job output/error files

### 6.3 Dependencies

**Core Libraries:**
- mkdocs, mkdocs-material, mkdocstrings
- pysindy, pydmd
- torch, torchdiffeq
- neuromancer
- deepxde
- do-mpc
- fastapi, uvicorn
- pandas, numpy, scipy, scikit-learn
- matplotlib

---

## 7. Project Milestones

### Milestone 1: Building Thermodynamic Simulation (Weeks 1-6)
- [x] Data extraction pipeline from AAU BUILD BMS
- [x] Dataset cleaning and augmentation
- [x] Working SINDy model implementation
- [x] DMD backup implementation
- [x] Model validation and metrics

### Milestone 2: Smart Controller (Weeks 7-10)
- [x] MPC controller using do-mpc
- [x] Integration with thermodynamic model
- [x] Constraint and cost function definition
- [ ] Testing with simulated scenarios

### Milestone 3: Control Panel (Weeks 11-12)
- [x] UPPAAL integration architecture
- [x] HTTP bridge for NN inference
- [ ] Basic automation rules
- [ ] Interface for manual setpoints

### Milestone 4: Final Report and Presentation (Weeks 13-14)
- [ ] Project report completion
- [ ] Technical documentation
- [ ] Performance evaluation
- [ ] Final presentation

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary Modeling | SINDy | Interpretable, sparse models suitable for control |
| Backup Modeling | DMD | Well-supported, effective for linearized systems |
| Controller | MPC | Predictive, constraint-handling, multi-actuator |
| NN Alternative | Neural ODE | Continuous-time dynamics, irregular sampling |
| Integration | HTTP/REST | Language-agnostic, easy UPPAAL integration |
| Documentation | MkDocs | Auto-generated API docs, math support |

---

## 9. Files of Interest

### Core Implementation Files
- `thermodynamics_modeling/sparse_identification/dynamic_model_smart_building.py` - Main SINDy model
- `thermodynamics_modeling/pythorch_node/torchdiffeq_model.py` - PyTorch NODE
- `thermodynamics_modeling/neuromancer_node/NODE.py` - NeuroMANCER implementation
- `do_MPC/hello_mpc.py` - MPC controller example
- `uppaal/http_to_uppaal/server.py` - Inference server
- `uppaal/http_to_uppaal/uppaal_wrapper.c` - C bridge for UPPAAL

### Data Processing Files
- `db_extraction/memory_efficient_bms_extraction.py` - BMS data extraction
- `db_extraction/multiprocessing_bms_data_extraction.py` - Parallel extraction
- `Database_augmentation_scripts/augment_weather.py` - Weather augmentation
- `Database_augmentation_scripts/occupancy_augmentation_6offices.py` - Occupancy data

### Documentation Files
- `docs/literature_review/PROJECT_DOCUMENT.md` - Full project document
- `docs/thermodynamics_modeling/smart_building_modeling.md` - SINDy documentation
- `docs/thermodynamics_modeling/NeuroMANCER.md` - NODE documentation
- `docs/uppaal/integration_architecture.md` - UPPAAL architecture

---

## 10. References

1. Candanedo, J., et al. "Data-Driven Smart Buildings: State-of-the-Art Review"
2. Brunton, S. L., et al. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." PNAS
3. Afram, A., & Janabi-Sharifi, F. (2014). "Theory and applications of HVAC control systems–A review of model predictive control (MPC)." Building and Environment
4. Johra, H., et al. "What Metrics Does the Building Energy Performance Community Use to Compare Dynamic Models?"
5. Kutz, J. N., et al. (2016). "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems." SIAM
6. Chen, R. T. Q., et al. (2018). "Neural ordinary differential equations." NeurIPS

---

## 11. Coding Conventions

The project follows these guidelines:
- **Python Style:** PEP 8 formatting
- **Type Hints:** Used wherever possible
- **Documentation:** PEP 257 docstring conventions
- **Code Examples:** Fenced code blocks with language identifiers
- **Testing:** Validation through RMSE and prediction accuracy
