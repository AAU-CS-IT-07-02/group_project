
# Edit region
## Report structure (draft)
- Introduction
    - Soft narrative preface about smart buildings, use/benefits/technical challenges (maintenance, design, upgrades/improvements difficulties)/how common they are, why they are important
    - Introduce TMV 23, use their paper
    - Context Subsection:
        - Current state of TMV
            - Hardware
            - Current BMS
            - Limitations
            - Problems caused by limitations
            - Previous developments and paths explored if any (past projects, iniciatives from Simon/Rasmus)

- Problem statement and proposed solution
    - Out of all the shortcommings in TMV 23, which one do we care about
        - Lack of predictive and collaborative controllers (vs the existing reactive P controller) cause:
            - actuator wear
            - inneficiencies in energy usage
            - lack of comfort (with only reactions the actuators cannot take the room/cluster/building to the correct state fast enough (ie. no active mechanical cooling))
            - diffculties having priorities or constraints in existing controllers
            - oscillations due to lack of communication between actuators and their changes of 
        - [TODO]: add the rest of those identified during the first meeting and the initial presentation
    - Explain conceptially how we would solve it and how we came to that decisions
        - What things need to change in order to improve the situation
        - What measures one can take to make those changes
        - This defines the hard and soft requirements for actually implemented the solution
            - Here we can mention the forces that shape our decision making, requirements, objectives of the semester, nice-to-haves, etc
        - What a full solution would look like: basically our roadmap with M1/M2/M3
            - It must be very clear in this section our focus is M1, with M2 being a nice to have, and leaving M3 for future work

- Literature Review & State of the Art
    
   {* 
    - Smart Buildings: Context & Requirements
        - Definition: Automated, monitored, integrated systems optimizing energy, comfort, sustainability
        - Core challenges: Fragmentation, reactive control, data quality, scalability
    
    - Building Modeling Approaches
        - White-box (physics-based): Interpretable but complex; hard to parameterize
        - Grey-box (hybrid): Balance interpretability + flexibility; requires domain knowledge
        - Black-box (data-driven): Flexible but needs large, quality datasets; less interpretable
        - Neural ODE approaches: Flexible time-series modeling; computational cost & portability concerns
    
    - Control Strategies for Building HVAC
        - Reactive (P/PI/PID): Simple, robust, deployed widely; no anticipation → oscillations & inefficiency
        - MPC: Optimal over horizon; requires accurate model & computational resources
        - Learning-based: Adaptive; safety & convergence concerns
        - Collaborative/decentralized: Scalable but coordination overhead
    
    - Data-Driven Identification Techniques
        - PySINDy: Discovers interpretable sparse equations; limited to smooth, low-dimensional dynamics
        - Neural networks: General-purpose; needs lots of data; black-box
        - PINNs: Physics-aware; reduces data; requires good understanding of governing equations
        - Key constraint for buildings: Data sparse, noisy, incomplete; generalization across buildings difficult
    
   *}


- Methodology
    - How we organize ourselves and our work (systems, processes, protocols)
        - Git/Github
        - Commit structure, PR protocol
        - Branch organization (trunk based dev)
        - Wiki
        - Issues creation, organization and assignment
        - Project
            - Roadmap
            - Custom properties
        - Mkdocs
    - How we will organize the research/development process
        - How we identify the available approaches for our desired solution
        - Characterize/define them
        - Sort them by priority and explain selection criteria
            - Fit the most requirements; drop requirements for progress when needed
        - Systematically explore these options until we find suitable ones
            - This sets the stage for our failures/pivots/success
        - Data strategy:
            - Data requirements: Complete time-series from sensors, actuators, configurations for combinations of rooms (floors, clusters, individual)
            - Quality standards: Minimal gaps, consistent timesteps, labeled features
            - Processing pipeline: Interpolation, normalization, augmentation (weather, occupancy)
        - Milestones for the project
            - Explain that milestones correspond with independent software elements with well-defined interfaces to encourage modularity and independent work
    
- Design and Implementation of explored solutions
    - Data: What actually happened
        - AAU BMS API: Slow/unreliable; resolved with IT support but pivoted to pre-existing dataset
        - Simon's Dataset: 6-room office (RoomA–F), Feb–Dec 2023; HuggingFace hosted;
        - Processing: Interpolation, MinMax normalization, weather/occupancy augmentation
        - Final: `/thermodynamics_modeling/neuromancer/dataset_split/`, 80/20 chronological split
        - Limitations: Since some of the discrete actuators data (particularly AHU_active) contain values that do not appear in the documentation (see "Detailed operational building data for six office rooms in Denmark: Occupancy, indoor environment, heating, ventilation, lighting and room control monitoring with sub-hourly temporal resolution"), they introduce ambiguity and inconsistency. This was discovered while implementing the random controller in python (it selects exactly one value from the set of documented values).
    
    - M1 Models:
            - Attempt type (classic ML with standard training, )
                - Black/White/Grey/Etc box? 
                - Here or inside each attempt but, what requirements does it fulfil and which it does not
                - Attempt name 1
                    - Technique explanation
                    - Design
                    - MCC3 considerations
                        - What elements 
                    - Implementation
                    - Outcome
                - Attempt name n
    - M2 Controllers:
        - This one is trickier because the limiting factor was not getting controller libraries to work, but rather having them be compatible with models
        - do_mpc:
            - Technique explanation
            - Why it would be suitable
            - What we tried
            - Why we cannot use it
        - Pure python:
        - Uppal:
            - Technique explanation
            - Why its suitable
            - How we made it work
            - Usage
                - Classic UPPAAL
                - TiGA
                - Stratego 
                - Int/Double discretization
- Results and Evaluation of successful approach (torchdiffeq and UPPAAL)
    - Gantt chart (NEEDS TO BE MOVED TO AN APPROPRIATE PLACE)
    - Model evaluation
        - Using real data to ascertain accuracy and horizon limit
            - Closed vs open loop simulation
            - Long vs short prediction horizons
            - Multiple scenarios
        - Using synthetic initial conditions
            - Agai

- Conclusion
- Future work















































# Introduction

Here is the introduction. The next chapter is chapter 2.

## Timeline

1. Intro to AAU build BMS and challenges
2. Logging platform for sensors, actuators and configurations of the BMS. In a PostgreSQL DB available through an API.
3. Identification and formalization of the problem: the control system is purely reactive, not predictive, and manages the actuators individually. (we identified more problems than this but we focused on this one)
4. We proposed a solution to the main problem mentioned above. It consists of finding a model for the thermodynamics of AAU Build, and developing a series of controllers working towards predictive and collaborative control for such a model.
5. We analyzed the available data (BMS system, building blueprints, log DB data).
   5.1. Analyzing log data required us to download it using the provided API. This was problematic because it was very slow at the start and ended up not working at all. **Data was fairly incomplete and unreliable. Coupled with the download issues we decided to pivot to a pre-existing dataset.
       5.1.1. We managed to get in contact with IT support and the correct stakeholders of the platform and resolve the issue, even though we never used it again.
   5.2. Used the blueprint and BMS information to write down the structure of the building (e.g., which floor a room is on, which cluster a room belongs to, room types).
   5.3. Analyzed the content of the partially downloaded dataset. Found out: data is not consistent, poorly labeled, lots of missing values, values outside expected domains.
   5.4. Evaluated the possibility of performing data augmentation to enrich the dataset (weather and occupancy augmentation).
6. We anticipated we would need to run many experiments, so we got access to the MCC3 computing platform and developed a pipeline for it.
7. Explored and read about the available techniques for data-driven modelling of building thermodynamics. Decided to start with PySINDy.
8. Spent considerable time diving into PySINDy to understand its features/limitations and implementation details.
   8.1. Explored solvers, function libraries and combinations, weak vs strong formulation, and PDE support.
   8.2. Fast identification but slow verification. Tried to speed things up with JIT/SymPy/JAX/Numba; got partial success. The core limitation was translating PySINDy's equation representation into each library's format, which limited support to lower-order polynomials (with a few JAX exceptions).

In parallel:

9. Switched to the dataset from Simon's paper.
   9.1. Analyzed it — better than the previous dataset, but still had missing values.
   9.2. Weather data was already present; focus moved to occupancy analysis.
10. Even before having a working model, we started exploring controller development. We anticipated using MPC, so we began with the `do_mpc` library.
11. Since a thermodynamics model was not yet available, we explored alternatives:
   11.1. Dynamic Mode Decomposition (DMD): tried but we struggled to apply it and interpret results.
   11.2. DeepXDE with Physics-Informed Neural Networks (PINNs): produced a first 'working' model, but it was inaccurate because it required a seed ODE; a pure NN sometimes worked better.
   11.3. Neuromancer: a black-box NN approach that produced promising results. Exporting models for use by external controllers was hard (ONNX/export didn't work well); the models also rely on an internal loss function for the ODE pass, which requires real measurements and limits usage to closed-loop simulation.
   11.4. `torchdiffeq`: similar to Neuromancer in using a latent-NN ODE approximation and numerical integration. The model was more portable and did not rely on an internal ODE loss, making open- and closed-loop simulations viable. This became our final and most accurate model.
12. We tried to port all models to `do_mpc`, but limited external model support made this difficult. We attempted ApproximateMPC and ONNX but didn't succeed.
13. Implemented a simulation loop supporting open and closed-loop calls to models, allowing us to use dataset-based control inputs, compare to baseline, or plug in any controller implementation.
14. Implemented Random and Bang-Bang controllers in Python.
15. Connected the NN-based model to UPPAAL using external C function calls plus a REST API wrapper and `curl`.
   15.1. Implemented a global Bang-Bang (BB) controller.
   15.2. Implemented a per-room BB controller.
   15.3. Implemented a UPPAAL system template suitable for Stratego.
      15.4. Implemented online policy learning control without moving the system/horizon.
      15.5. Pending: finalize policy learning and evaluate.

---

### After the report
- Slides for presentation (Marco)
- Documentation
- Demo

### Before the report
- Decide whether to hand in the repository. If yes, determine the expected state.


## TODOs
- Where to put all information regarding dataset
<!-- - Where to put literature review and state of the art -->
- Limitations for overall project and each technique/step?



## Notes
- How to discuss deviations that shaped the outcome of the project
- How AAU data is handled (management, security)

## Option 1 – Narrative and structure
### Introduction / Context
### Background
- Basic knowledge for context
  - ODEs
  - Controllers
  - Thermodynamics concepts

### Problem Statement
- Clearly define the reactive vs predictive control issue
- Mention why this is critical for building energy efficiency

### Proposed Solution

### Literature Review / State of the Art
- High-level description of predictive and collaborative control approaches

### Methodology
Choose a narrative style:
- Option 1: Systematic exploration — justify final choice (show a decision tree or timeline of choices)
- Option 2: Focus on pivots and concessions — explain why black-box modeling was necessary (a pragmatic choice due to constraints)
- Option 3: Mistakes and lessons learned — highlight practical challenges and how they were addressed

> EDUARD: I prefer a combination of Option 1 and Option 2 for transparency.

Notes for methodology:
1. We reviewed available techniques and assessed them against our problem, accessibility and support.
2. We framed pivoting to less explainable models as pragmatic concessions.
3. Sometimes we prioritized implementation speed over exhaustive evaluation of alternatives.

~~Alternative: talk about mistakes explicitly and why/when they occurred instead of focusing on implementation.~~

## Design
- M1
  - PySINDy baseline
  - Classic trained models (DeepXDE, Neuromancer)
  - NODE model using `torchdiffeq`
- M2
  - `simulator.py` as an interface between model and controller
  - UPPAAL -> C -> HTTP integration; discretizations
    - Stratego
    - Classic
    - Tiga
- M3

Code parameterization and pipeline from training to controller deployment.

## Implementation
Note: Implementation can be used to explain trade-offs, or focus on 'what' was implemented.
Example: Neuromancer produced a `.pt` model that was hard to port between frameworks.

For each technique (PySINDy, Neuromancer, `torchdiffeq`), add:
- What it is
- Why we tried it
- Why it worked or failed

For each pivot, explain the reasoning (e.g., data issues -> switch to Simon's dataset).
For controllers, explain why we started with simple ones (Random, Bang-Bang) before pursuing MPC.

## Results and Evaluation
Model:
- Model characterization (effect of outside temperature, occupancy, and room topology)
- Model size/performance comparisons
- Datasets used for training & testing
- Data subset impacts on model performance

Controllers:
- Random vs Bang-Bang
- Controller plots
- Different setpoints (temperature and controller setpoints)
- Time until controllers drift from setpoint
- Potential UPPAAL learning integration

## Discussion
Topics:
- Alternative evaluation metrics for models/controllers
- Other model architectures worth exploring
- Other controller approaches
- Possible UPPAAL developments
- Dashboard for controller profile management
- Integrating pipeline into production

Also: Habit profiling and occupancy-based model learning.

## Conclusion and Future Work
- Compare initial aims vs final outcomes
- Suggest next steps (e.g., dashboard, habit profiling)
- Personal notes about implementations and pipeline utility
- Review recent research updates

## References + Appendix

## Side notes
- Data issues and how they shaped the project
- Why certain techniques were abandoned
- How deviations influenced the final design

# End edit region



## Report structure (draft)
- Introduction
- Problem statements
- Problem description
- Explored solutions
- Implementation
- Evaluation
- Conclusion
- Future work

## Notes
- How to discuss deviations that shaped the outcome of the project
- How AAU data is handled (management, security)

## Option 1 – Narrative and structure
### Introduction / Context
### Background
- Basic knowledge for context
  - ODEs
  - Controllers
  - Thermodynamics concepts

### Problem Statement
- Clearly define the reactive vs predictive control issue
- Mention why this is critical for building energy efficiency

### Proposed Solution

### Literature Review / State of the Art
- High-level description of predictive and collaborative control approaches

### Methodology
Choose a narrative style:
- Option 1: Systematic exploration — justify final choice (show a decision tree or timeline of choices)
- Option 2: Focus on pivots and concessions — explain why black-box modeling was necessary (a pragmatic choice due to constraints)
- Option 3: Mistakes and lessons learned — highlight practical challenges and how they were addressed

> EDUARD: I prefer a combination of Option 1 and Option 2 for transparency.

Notes for methodology:
1. We reviewed available techniques and assessed them against our problem, accessibility and support.
2. We framed pivoting to less explainable models as pragmatic concessions.
3. Sometimes we prioritized implementation speed over exhaustive evaluation of alternatives.

~~Alternative: talk about mistakes explicitly and why/when they occurred instead of focusing on implementation.~~

## Design
- M1
  - PySINDy baseline
  - Classic trained models (DeepXDE, Neuromancer)
  - NODE model using `torchdiffeq`
- M2
  - `simulator.py` as an interface between model and controller
  - UPPAAL -> C -> HTTP integration; discretizations
    - Stratego
    - Classic
    - Tiga
- M3

Code parameterization and pipeline from training to controller deployment.

## Implementation
Note: Implementation can be used to explain trade-offs, or focus on 'what' was implemented.
Example: Neuromancer produced a `.pt` model that was hard to port between frameworks.

For each technique (PySINDy, Neuromancer, `torchdiffeq`), add:
- What it is
- Why we tried it
- Why it worked or failed

For each pivot, explain the reasoning (e.g., data issues -> switch to Simon's dataset).
For controllers, explain why we started with simple ones (Random, Bang-Bang) before pursuing MPC.

## Results and Evaluation
Model:
- Model characterization (effect of outside temperature, occupancy, and room topology)
- Model size/performance comparisons
- Datasets used for training & testing
- Data subset impacts on model performance

Controllers:
- Random vs Bang-Bang
- Controller plots
- Different setpoints (temperature and controller setpoints)
- Time until controllers drift from setpoint
- Potential UPPAAL learning integration

## Discussion
Topics:
- Alternative evaluation metrics for models/controllers
- Other model architectures worth exploring
- Other controller approaches
- Possible UPPAAL developments
- Dashboard for controller profile management
- Integrating pipeline into production

Also: Habit profiling and occupancy-based model learning.

## Conclusion and Future Work
- Compare initial aims vs final outcomes
- Suggest next steps (e.g., dashboard, habit profiling)
- Personal notes about implementations and pipeline utility
- Review recent research updates

## References + Appendix

## Side notes
- Data issues and how they shaped the project
- Why certain techniques were abandoned
- How deviations influenced the final design

### Option 2 – Alternate structure
- Introduction and Motivation
- What we found in the AAU BMS
- Starting goals and ideas
- How the project evolved
  - Working with the data
  - Setting up experiments (MCC3)
  - Trying different modeling methods
  - First controllers
  - UPPAAL integration
- Final approach
- Results
- What worked, what didn't, and what we learned
- Conclusion and next steps

---

## Scratchpad
Group implementations by paradigm (e.g., NODEs: `DeepXDE`, `Neuromancer`, `torchdiffeq`).