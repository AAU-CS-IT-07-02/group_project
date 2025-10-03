
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
- Smart predictive and collaborative controller:
- Control panel: 
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