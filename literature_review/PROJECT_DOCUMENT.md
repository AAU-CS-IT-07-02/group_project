# Secure, Scalable and Useful Systems
### Aalborg University
## Semester Project - Computer Science Master's Program

---

**Project Title:** Intelligent Building Management System through Data-Driven Thermodynamics Modeling and Control

**Course:** [Secure, Scalable and Useful Systems]  
**Semester:** [7th Semester, 2025]  
**Team:** Dragomir Matei Mihai, Eduard Brahas, Monda Rareș, Pedro Felizardo Pedroso Carreira Lima, Nasik Ali Khan, Beltrán Aceves Gil  
**Supervisor(s):** [Marco Muñiz]  
**Collaborator(s):** Rasmus Lund Jensen and Simon Pommerencke Melgaard  
**Date:** October 3, 2025  

## Table of Contents

1. [Project Context](#1-project-context)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Scope and Milestones](#4-scope-and-milestones)
5. [Requirements](#5-requirements)
6. [References](#6-references)
7. [Amendments](#7-amendments)
---

## 1. Project Context

This project is developed in collaboration with the Department of Civil Engineering at Aalborg University, leveraging the unique opportunities provided by AAU BUILD. The building was constructed in 2016 and serves as both a functional academic facility and a cutting-edge Living Lab for building performance research and experimentation.

Located at Thomas Manns Vej 23 in Aalborg, Denmark, TMV 23 is a multi-storey office building spanning approximately 9,000 m² across five floors and one basement level. The building houses around 150 staff employees and 600 students, accommodating diverse functions including offices, meeting rooms, classrooms, workshops, and laboratories. As a Living Lab, TMV 23 enables real-world experimental setups and investigations to be conducted directly within the active working environment of staff and students.

The building is equipped with an extensive sensor network and building management system (BMS) that monitors various environmental parameters including light levels, temperature, CO2 concentrations, room occupancy, and humidity. The facility features 14 ventilation systems with varying capabilities, radiators, radiative panels, and three cooling units, all managed through a Schneider EcoStruxure BMS. Additionally, the building incorporates sustainable energy features such as photovoltaic panels and district heating connectivity.

The TMV 23 facility provides an ideal testbed for developing and validating intelligent building management solutions, offering both comprehensive sensor data and interfaces for deploying and testing automated control algorithms in a real-world environment. This project aims to leverage this unique infrastructure to develop advanced control methodologies that can integrate multiple data sources and coordinate building systems for improved energy efficiency and occupant comfort.

## 2. Problem Statement

Despite this high level of automation and sophisticated infrastructure, the current control systems operate independently without collaboration or integration. These simple controllers face significant challenges from dynamic environmental conditions including weather variations, heat generation from lighting and occupancy, and complex interactions between different building systems. The lack of coordinated control strategies represents a significant source of inefficiencies and oscillations. 

Furthermore, the existing controllers are purely reactive, responding only to current conditions without any predictive capabilities. This leads to energy waste, occupant discomfort, and poor performance because systems cannot prepare for predictable changes like daily schedules, weather forecasts, or occupancy patterns. Instead of smooth operation, the building systems are constantly inducing oscillations, wasting energy and creating uncomfortable conditions.

## 3. Proposed Solution
We propose the development of a collaborative and predictive control system for the entire building. This system is composed of three modular components with well-defined interfaces, enabling independent development and easy substitution of alternative implementations:

### **Building Thermodynamics Simulation**

As a prerequisite to building a predictive controller, the system must be able to model the thermodynamic behavior of the building to predict its state and simulate reactions to actuators. A systematic evaluation of available techniques reveals the following decision-making process:

*Black Box Approaches*: Despite having extensive data collected from AAU BUILD's sensors and actuators over 4 years, black box methods are ruled out due to their reduced explainability, interpretability, and predictability. Given the security focus of this semester project, sufficient guarantees cannot be provided using techniques such as:
- Classic neural networks (RNNs/LSTMs)
- Physics-informed Neural Networks  
- Neural ODEs

Additionally, the controller needs to calculate distant future states efficiently rather than computing each intermediate step, aiming to reduce computational overhead.

*Grey/White Box Approaches*: These considerations direct the approach toward grey/white box models that provide greater insights and guarantees:
- **Manual ODE definition**: Given AAU BUILD's size, complexity, and high-dimensional functional dependencies, manually defining ordinary differential equations is ruled out, especially considering the team's limited expertise in this domain.
- **UPPAAL finite-state automata**: While more approachable and well-supported by existing research, this approach is rejected due to the high barrier to entry for both the technique and tooling, given limited experience.
- **Data-driven ODE approximation**: Leveraging the extensive dataset, grey/white box models can be trained that better align with the requirements. Two promising techniques are identified:
  - **Sparse Identification of Nonlinear Dynamics (SINDy)**: Approximates nonlinear systems as sparse combinations of elements from a function library. This approach offers excellent research foundation and technical support through [PySINDy](https://github.com/dynamicslab/pysindy), with the flexibility to model both linear and nonlinear state-space representations.
  - **Dynamic Mode Decomposition (DMD)**: Approximates system dynamics through linearization of evolution modes. Well-supported academically and technically through [PyDMD](https://github.com/PyDMD/PyDMD), serving as a backup methodology.

SINDy is chosen as the primary system identification technique, with DMD as a backup method in case the current function library implementation struggles to model the system dynamics.

### **Smart Predictive and Collaborative Controller**

Building upon an accurate thermodynamic model, the project aims to implement a controller with predictive capabilities that can coordinate multiple actuators to achieve optimal building performance. The controller should consider the following features:
- **Predictive capability**: Anticipate future states and disturbances
- **Multi-actuator coordination**: Manage HVAC, lighting, and ventilation systems collaboratively
- **Safety guarantees**: Ensure system constraints are never violated

Various control approaches are considered based on their suitability for the multiple objectives:

**Traditional Control Methods**:
- **PID Control**: While popular and well-understood, PID controllers are inherently reactive and would require complex hierarchical inner/outer control loops to achieve predictive capabilities. This approach is rejected due to implementation complexity and limited coordination abilities.

**State-Based Control**:
- **Finite-State Automata (UPPAAL)**: Could provide formal verification capabilities and handle discrete system states effectively. However, achieving performance comparable to other control methods might prove difficult, particularly for the complex, multi-dimensional optimization required in smart building control.

**Advanced Control Strategies**:
- **Optimal Control**: While this approach could handle multi-objective optimization needs and constraint requirements perfectly, it's simply too computationally expensive for real-time building control. Running detailed system simulations at every control step would be impractical.

- **Model Predictive Control (MPC)**: Represents the optimal balance between performance and computational feasibility. MPC addresses all requirements through:
  - **Predictive capability**: Uses the thermodynamic model to forecast system behavior over a configurable prediction horizon
  - **Multi-actuator coordination**: Simultaneously optimizes control inputs across all building systems
  - **Constraint handling**: Ensures safety through hard constraints on system variables
  - **Computational efficiency**: Balances prediction accuracy with real-time performance through adjustable control and prediction horizons
  - **Objective prioritization**: Minimizes energy consumption, oscillations, and actuator wear through weighted cost functions

#### **Implementation Strategy**

The MPC controller will be implemented using the [Model Predictive Control Python toolbox](https://www.do-mpc.com/en/latest/), which provides:
- Robust optimization solvers for nonlinear MPC problems
- Flexible constraint definition capabilities
- Integration interfaces for SINDy-based building dynamics model
- Real-time performance optimization features

The controller will operate on a hierarchical structure where high-level comfort and energy objectives are translated into coordinated setpoints for individual building systems, ensuring both global optimization and local constraint satisfaction.

### **Control Panel Interface**

The final system component would be a control panel responsible for setting desired system states that the MPC controller achieves (analogous to defining system "attitude" or trajectory). This interface would work as a building profile selection mechanism with built-in automations for:
- Nighttime temperature adjustments
- Enhanced comfort conditions for booked rooms  
- Extreme weather forecast handling
- User-selectable comfort profiles (warm, cool, high occupancy scenarios)

While likely beyond this project's scope due to time constraints, this control panel would be implemented using UPPAAL, providing mechanisms to verify profile system validity and security while maintaining an easily extensible interface.

## 4. Scope and Milestones

### 4.1 Project Scope

The project has been designed a comprehensive building management system, but realistically not everything will be finished given the one semester time constraint. The main focus will be getting the thermodynamic simulation working well, and then attempting to implement a basic version of the controller.

**Primary focus areas:**
- Building a thermodynamic model using SINDy
- Getting data from AAU BUILD's BMS and cleaning it up
- Validating the model works properly
- Basic MPC controller implementation

**Secondary objectives (if time permits):**
- Testing the controller and evaluating performance
- Simple interface for changing setpoints manually
- Comparing the approach with existing methods

**Out of scope:**
- Full control panel with UPPAAL verification
- Complete system integration and real deployment
- Extensive real-world testing

### 4.2 Development Milestones

The work is split into three development milestones plus the final report:

#### **Milestone 1: Building Thermodynamic Simulation**
**Timeline:** Weeks 1-6  
**Deliverables:**
- Pipeline to extract data from AAU BUILD BMS
- Clean augmented dataset ready for training
- Working SINDy model
- Validation results and performance metrics
- DMD backup implementation if SINDy doesn't work

**Success criteria:**
- Model predicts building behavior accurately
- Fast enough to use for real-time control

#### **Milestone 2: Smart Controller**
**Timeline:** Weeks 7-10  
**Deliverables:**
- MPC controller using do-mpc
- Integration with the thermodynamic model
- Basic constraints and cost functions
- Testing with simulated scenarios

**Success criteria:**
- Controller can coordinate multiple actuator
- Works better than simple reactive controllers
- Respects safety constraints

#### **Milestone 3: Control Panel**
**Timeline:** Weeks 11-12 (if time permits)  
**Deliverables:**
- Basic interface for manual control
- Simple automation rules
- Documentation for future UPPAAL work

**Success criteria:**
- Working interface for manual setpoints
- Basic automated switching between profiles

#### **Milestone 4: Final Report and Presentation**
**Timeline:** Weeks 13-14  
**Deliverables:**
- Project report
- Technical documentation
- Performance results
- Final presentation

**Success criteria:**
- Complete documentation of achieved results
- Clear evaluation of system performance
- Cross-team knowledge of each area
- Plan for future work

<!-- ### 4.3 Risk Management

**Technical problems:**
- SINDy might not work well → Use DMD instead
- BMS data might be messy → Spend more time cleaning it
- MPC might be too slow → Use simpler model or shorter prediction horizon

**Timeline problems:**
- Running behind schedule → Focus on core features first
- Integration issues → Ensure interfaces are well-defined -->

## 5. Requirements

### 5.1 Functional Requirements
<!-- - Extract and process BMS data efficiently
- Implement SINDy algorithms for system identification
- Develop predictive control algorithms
- Evaluate performance against baseline methods
- Generate clear documentation and results -->

### 5.2 Non-functional Requirements

## 6. References

1. Candanedo, J., Blum, D., Chen, Z., Fierro, G., Gori, V., Johra, H., Madsen, H., Marszal-Pomianowska, A., O'Neill, Z., Pradhan, O., Rovas, D., Sacco, F., Stensson, S., Thilker, C., Vallianos, C., Wen, J., & White, S. Data-Driven Smart Buildings: State-of-the-Art Review. *Department of the Built Environment, The Faculty of Engineering and Science, Energy and Buildings Research Group*.

2. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932-3937.

3. Afram, A., & Janabi-Sharifi, F. (2014). Theory and applications of HVAC control systems–A review of model predictive control (MPC). *Building and Environment*, 72, 343-355.

4. Johra, H., Schaffer, M., Chaudhary, G., Kazmi, H. S., Le Dréau, J., & Petersen, S. What Metrics Does the Building Energy Performance Community Use to Compare Dynamic Models? *Department of Construction, Urban and Environmental Engineering, Faculty of Engineering and Science, Research Group for Energy in Buildings*.

5. PySINDy Documentation: https://pysindy.readthedocs.io/en/latest/

6. AAU BMS API Documentation (Internal)