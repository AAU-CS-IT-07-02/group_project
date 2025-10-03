
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

This project is developed in collaboration with the Department of Civil Engineering at Aalborg University, leveraging the unique opportunities provided by AAU BUILD. It was constructed in 2016, serves as both a functional academic facility and a cutting-edge Living Lab for building performance research and experimentation.

Located at Thomas Manns Vej 23 in Aalborg, Denmark, TMV 23 is a multi-storey office building spanning approximately 9,000 m² across five floors and one basement level. The building houses around 150 staff employees and 600 students, accommodating diverse functions including offices, meeting rooms, classrooms, workshops, and laboratories. As a Living Lab, TMV 23 enables real-world experimental setups and investigations to be conducted directly within the active working environment of staff and students.

The building is equipped with an extensive sensor network and building management system (BMS) that monitors various environmental parameters including light levels, temperature, CO2 concentrations, room occupancy, and humidity. The facility features 14 ventilation systems with varying capabilities, radiators, radiative panels, and three cooling units, all managed through a Schneider EcoStruxure BMS. Additionally, the building incorporates sustainable energy features such as photovoltaic panels and district heating connectivity.

The TMV 23 facility provides an ideal testbed for developing and validating intelligent building management solutions, offering both comprehensive sensor data and interfaces for deploying and testing automated control algorithms in a real-world environment. This project aims to leverage this unique infrastructure to develop advanced control methodologies that can integrate multiple data sources and coordinate building systems for improved energy efficiency and occupant comfort.

## 2. Problem Statement

Despite this high level of automation and sophisticated infrastructure, the current control systems operate independently without collaboration or integration. These simple controllers face significant challenges from dynamic environmental conditions including weather variations, heat generation from lighting and occupancy, and complex interactions between different building systems. The lack of coordinated control strategies represents a significant source of inefficiencies and oscillations. 

Furthermore, the existing controllers are purely reactive, responding only to current conditions without any predictive capabilities. This leads to energy waste, occupant discomfort, and poor performance because systems cannot prepare for predictable changes like daily schedules, weather forecasts, or occupancy patterns. Instead of smooth operation, the building systems are constantly inducing oscillations, wasting energy and creating uncomfortable conditions.

## 3. Proposed Solution

### 3.1 Proposed draft 1
We propose the develpment of a collaborative and predictive control system for the entire building. This system is made up of three components (well-defined?, isolated?, i just want to say that there are clear interfaces between them making it easy to swap them for alternatives):
- Building simulation: as a prerequisite to building a predictive controller we must be able to model the behaviour/thermodynamic of the building to predict its state and simulate the reactions to our controller. There were multiple techniquest at our disposal and we followed this decision-making process:
    - Black box approach: given the extensive amount of data available to us, collected from sensors and actuators of AAU BUILD throughout 4 years, a variery of black box aproaches where availabel to us but we decided against them because it would reduce the explainability, interpretability and  predictability of the simulation. Keepint in mind the security theme of the semester project, we felt we could not give any guarantees using one of these techniques:
        - Classis NNs like RNNs/LSTMs
        - Physics-informed Neural Networks
        - Neural ODEs
    - We also wanted our controller to be able to calculate far states of our system to aid in its predictive capabilities, rather than compute each step of the way, in the hopes of reducing the computational cost.
    - Grey/White box approach: given this information, we switches our attention to grey/white models that would give us more control and insight
        - Manually define a system of ordinary differential equations: given the size, complexity and high cardinality of functional dependencies of AAU BUILD, we quickly ruled out this method, specially taking into account the lack of expertise in this area from our group members
        - Manually define a finite-state automata in UPPAAL: while more aproachable and highly supported by previous reseach, we decided against it for the high barrier of entry to both the technique and tool given our lack of experience
        - Data-driven approximation of ODEs: given the bast amounts of data at our disposal we can also train grey/white box models  that more closely align with our requirements. We have identified two potential techniques:
            - Sparse identification of non linear dynamics: which aproximates non linear systems as a sparse combination of elements in a function library. Really good research and tech stack support through [PySINDy](https://github.com/dynamicslab/pysindy). We chose to use this because we have the option to model both linear and non-linear state-space representations.
            - Dynamic mode decomposition: aproximation of system dynamics through linearization of evolution modes. Well supported academically and tech stack through [PyDMD](https://github.com/PyDMD/PyDMD). This will be our backup method.  

- Smart predictive and collaborative controller: building on top of an accurate model we wanted to implement a controller with predictive capabilities that would be able combine multiple actuators. We considered the following options:
    - Finite-state automata: using a tool like UPPAAL we would build a state machine to control our system, but we doubted our ability to produce a controller with performance on par with more accesible techniques.
    - PID: a very popular approach in control theory that we quickly rejected due to the need of an inner an outer hierarchical control loops to add predictive capabilities.
    - Optimal control: while more in line with our requirements of security, prediction and colaboration; the computational cost of simulating our system so thouroughly discualified this method
    - MPC: model predictive control represents a good middle ground between the other methods that balances security and scalabily thanks to its tweakable prediction/control horizon. This technique can make good use of our thermodynamics model and predicting its state, combine the actions of multiple actuators to achieve our desired state, minimize energy use, oscillations, actuator wear, etc thanks to its priority system and help us guarantee the safety of the system and its actions thanks to the constraint system. We will be pursuing this path, using the [Model predictive control python toolbox](https://www.do-mpc.com/en/latest/)  

- Control panel: the final element of our solution would be a control panel in charge of setting the desired state of the system that the MPC controll would be in charge of achieving (can be thought of as the "attitude" or trajectory of the system). It can be thought as a mechanism for choosing a profile for the building, but with a small set of automations built in to take into account desired behaviour like changing the temperature for the night, improved comfort conditions for booked rooms, handling extreme weather forecasts, providing end-users a set of comfort profiles for warm, cool, or high ocupancy situations. While probably out of scope for this project due to time limitations, we would use UPPAAL to model this control panel, providing mechanisms to verify the validity and security of the profile system while maintaining an easy-to-expand interface.
### 3.2 Proposed draft 2

We propose the development of a collaborative and predictive control system for the entire building. This system is composed of three modular components with well-defined interfaces, enabling independent development and easy substitution of alternative implementations:

**Building Simulation Component**
As a prerequisite to building a predictive controller, we must be able to model the thermodynamic behavior of the building to predict its state and simulate reactions to our controller. Our systematic evaluation of available techniques led us through the following decision-making process:

*Black Box Approaches*: Despite having extensive data collected from AAU BUILD's sensors and actuators over 4 years, we decided against black box methods due to their reduced explainability, interpretability, and predictability. Given the security focus of this semester project, we cannot provide sufficient guarantees using techniques such as:
- Classic neural networks (RNNs/LSTMs)
- Physics-informed Neural Networks  
- Neural ODEs

Additionally, we require our controller to calculate distant future states efficiently rather than computing each intermediate step, hoping to reduce computational overhead.

*Grey/White Box Approaches*: These considerations directed us toward grey/white box models that provide greater control and insight:
- **Manual ODE definition**: Given AAU BUILD's size, complexity, and high-dimensional functional dependencies, we ruled out manually defining ordinary differential equations, especially considering our team's limited expertise in this domain.
- **UPPAAL finite-state automata**: While more approachable and well-supported by existing research, we decided against this due to the high barrier to entry for both the technique and tooling, given our limited experience.
- **Data-driven ODE approximation**: Leveraging our extensive dataset, we can train grey/white box models that better align with our requirements. We identified two promising techniques:
  - **Sparse Identification of Nonlinear Dynamics (SINDy)**: Approximates nonlinear systems as sparse combinations of elements from a function library. This approach offers excellent research foundation and technical support through [PySINDy](https://github.com/dynamicslab/pysindy), with the flexibility to model both linear and nonlinear state-space representations.
  - **Dynamic Mode Decomposition (DMD)**: Approximates system dynamics through linearization of evolution modes. Well-supported academically and technically through [PyDMD](https://github.com/PyDMD/PyDMD), serving as our backup methodology.

**Smart Predictive and Collaborative Controller**
Building upon an accurate thermodynamic model, we aim to implement a controller with predictive capabilities that can coordinate multiple actuators. Our evaluation of control strategies included:

- **Finite-state automata**: Using UPPAAL to build a state machine controller, though we questioned our ability to achieve performance comparable to more accessible techniques.
- **PID control**: A popular control theory approach that we rejected due to the complexity of implementing hierarchical inner/outer control loops for predictive capabilities.
- **Optimal control**: While better aligned with our security, prediction, and collaboration requirements, the computational cost of thorough system simulation disqualified this approach.
- **Model Predictive Control (MPC)**: Represents an optimal balance between other methods, offering security and scalability through adjustable prediction/control horizons. MPC can effectively utilize our thermodynamic model for state prediction, coordinate multiple actuators to achieve desired states, minimize energy consumption, oscillations, and actuator wear through its priority system, and guarantee system safety through constraint mechanisms. We will pursue this approach using the [Model Predictive Control Python toolbox](https://www.do-mpc.com/en/latest/).

**Control Panel Interface**
The final system component would be a control panel responsible for setting desired system states that the MPC controller achieves (analogous to defining system "attitude" or trajectory). This interface functions as a building profile selection mechanism with built-in automations for:
- Nighttime temperature adjustments
- Enhanced comfort conditions for booked rooms  
- Extreme weather forecast handling
- User-selectable comfort profiles (warm, cool, high occupancy scenarios)

While likely beyond this project's scope due to time constraints, we would implement this control panel using UPPAAL, providing mechanisms to verify profile system validity and security while maintaining an easily extensible interface.

## 4. Scope and Milestones
## 5. Requirements

### 5.1 Functional Requirements
- Extract and process BMS data efficiently
- Implement SINDy algorithms for system identification
- Develop predictive control algorithms
- Evaluate performance against baseline methods
- Generate clear documentation and results

### 5.2 Non-functional Requirements

## 6. References

### Core Methodology
1. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932-3937.

### Building Control
2. Afram, A., & Janabi-Sharifi, F. (2014). Theory and applications of HVAC control systems–A review of model predictive control (MPC). *Building and Environment*, 72, 343-355.

### Technical Documentation
3. PySINDy Documentation: https://pysindy.readthedocs.io/en/latest/
4. AAU BMS API Documentation (Internal)

### Additional References
*[To be expanded during literature review phase]*


##  7. Amendments
---

**Document Version:** 1.1  
**Last Updated:** October 3, 2025  
---