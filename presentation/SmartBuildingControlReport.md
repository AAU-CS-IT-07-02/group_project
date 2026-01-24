# Smart Building Control

**Data-driven modelling and control of AAU Build**\
Beltrán José Aceves Gil, Eduard Brahas, Matei-Mihai Dragomir, Nasik Ali Khan, Rares , Monda, Pedro Felizardo Pedroso Carreira Lima\
Computer Science (IT), CS-IT7-02, 2026-26\
Master’s Project

STUDENT REPORT

![Cover image](images/cover-1.png)

**Smart Building Control**\
Data-driven modelling and control of AAU Build\
Beltrán José Aceves Gil, Eduard Brahas, Matei-Mihai Dragomir, Nasik Ali Khan, Rares , Monda, Pedro Felizardo Pedroso Carreira Lima\
Computer Science (IT), CS-IT7-02, 2026-26\
Master’s Project\
\
STUDENT REPORT

© Aalborg University 2015

![AAU logo or page image](images/cover-2.png)

Computer Science\
Aalborg University\
https://www.cs.aau.dk/

**Title:** Data-driven modelling and control of AAU Build\
**Theme:** Scientific Theme\
**Project Period:** Fall Semester 2025\
**Project Group:** CS-IT07-02\
**Participant(s):** Beltrán José Aceves Gil; Eduard Brahas; Matei-Mihai Dragomir; Nasik Ali Khan; Rares , Monda; Pedro Felizardo Pedroso Carreira Lima\
**Supervisor(s):** Marco Antonio Muñiz Rodriguez\
**Copies:** 1\
**GitHub Organization:** https://github.com/AAU-CS-IT-07-02\
**Page Numbers:** 25\
**Date of Completion:** January 24, 2026\

The content of this report is freely available, but publication (with reference) may only be pursued due to agreement with the author.\

**Abstract:** This project set out to develop a predictive and collaborative control scheme for the AAU BUILD smart building to address the issues and limitations of its current system. To this end, Neural Ordinary Differential Equations (NODEs) are investigated as a data-driven approach for modelling room temperature dynamics, and the resulting model is integrated with the UPPAAL tool suite for controller synthesis and analysis. A NODE model is trained to predict future room temperature trajectories and deployed as a stateless REST service. Integration with UPPAAL is achieved through a novel approach that leverages the existing C function calling capabilities to interface with external models through a REST API, opening up the possibility of easily integrating other popular ecosystems like Pytorch into UPPAAL. Several controllers are evaluated, including a random baseline, global and perroom bang-bang control, and an online policy learning controller synthesised with UPPAAL Stratego. In the absence of classic analysis methods, Statistical Model Checking is used to assess behaviour under continuous variables and black-box dynamics. Results show stable, equilibrium-seeking NODE dynamics, improved performance from per-room bang-bang compared to global control, and further gains in regulation and actuator efficiency from the learning-based controller. Overall, the approach is well-suited for developing and validating data-driven control strategies that combine learned dynamics with formal synthesis and analysis.

[CS Department](https://www.cs.aau.dk/) · [Project GitHub](https://github.com/AAU-CS-IT-07-02)

## Acronyms

AAU — Aalborg University 12\
AHU — Air Handling Unit 9\
API — Application Programming Interface vi, 9, 12, 13, 24\
BAS — Building Automation System 1\
BMS — Building Management System 1, 2, 9\
HVAC — Heating, Ventilation and Air Conditioning 1, 11\
MAE — Mean Absolute Error 15\
MPC — Model Predictive Control 5, 23\
NODE — Neural Ordinary Differential Equation vi, 5, 10, 11, 14, 15, 17–25, II\
ODE — Ordinary Differential Equation vi, 6, 10–12, 23, 24\
PID — Proportional-Integral-Derivative 3, 5\
PINN — Physics Informed Neural Network 23\
PIR — Passive Infrared Sensor 9\
ReLU — Rectified Linear Unit 11\
RK4 — Fourth-order Runge–Kutta method 11\
RMSE — Root Mean Squared Error 15\
SINDy — Sparse Identification of Nonlinear Dynamical systems 23\
SMC — Statistical Model Checking 3, 13, 19–21, 24\
Tanh — Hyperbolic Tangent 11\
TMV 23 — Thomas Manns Vej 23 1, 2\
VAV — Variable air volume 9

## List of Figures

2.1 Milestone graph — p. 3\
5.1 Dataset overview — p. 10\
5.2 Architecture graph of the encoder–latent ODE function–decoder — p. 12\
5.3 Order of action with the REST API and UPPAAL — p. 13\
6.1 NODE performance comparison in closed-loop simulation — p. 14\
6.2 NODE performance comparison in open-loop simulation — p. 15\
6.3 Performance comparison through prediction horizons on closed-loop simulation — p. 16\
6.4 Performance comparison through prediction horizons on open-loop simulation — p. 16\
6.5 Performance comparison comparing frozen observations between window predictions and the baseline — p. 17\
6.6 Random controller performance on NODE simulation through UPPAAL — p. 17\
6.7 Global and per-room bang-bang controllers performance on NODE simulation through UPPAAL — p. 18\
6.8 Policy learning controller performance on NODE simulation through UPPAAL — p. 19\
6.9 Stability analysis visualisation for upper and lower temperature bounds, on NODE simulation through UPPAAL — p. 19\
6.10 Equilibrium under no actuator control, on NODE simulation through UPPAAL — p. 20\
6.11 Stability analysis visualization on NODE simulation through UPPAAL — p. 21\
6.12 Controllability analysis showing temperature range reachability under Warm and Cold control strategies on NODE simulation through UPPAAL — p. 22

## Contents

- Acronyms v\
- List of Figures vi\
- Preface viii\
- 1 Introduction 1\
- 2 Problem Statement and Proposed Solution 2\
  - 2.1 Problem Statement 2\
  - 2.2 Proposed Solution 2\
- 3 Literature review 4\
- 4 Methodology and requirements 7\
  - 4.1 Project Management and Collaboration 7\
  - 4.2 Systematic Exploration of Solutions 7\
- 5 Design and Implementation 9\
  - 5.1 Data 9\
  - 5.2 Modelling 10\
  - 5.3 Control 12\
- 6 Results and Evaluation 14\
- 7 Discussion 23\
- 8 Conclusions and further work 24\
  - 8.1 Conclusions 24\
  - 8.2 Further work 25\
- Bibliography 27\
- Appendix A: Dataset Columns Used for Model Training II

## Preface

Group CS-IT7-02 produced the following report as part of the first semester of the Master’s programme in Computer Science at Aalborg University. The semester theme was “Secure, Scalable and Useful Systems”, and the project focused on the proposal “Smart Building Control.”. The report presents a data-driven approach to modelling and controlling the dynamics of smart building systems.

Acknowledgement is given to the project supervisor, Marco Antonio Muñiz Rodriguez, for guidance throughout the project, as well as to Simon Pommerencke Melgaard and Rasmus Lund Jensen from AAU BUILD, for providing initial information, access to the systems, and answers to technical and operational questions.

Artificial intelligence tools, such as large language models (e.g., Copilot, DeepSeek), were used in a limited and reflective manner during this project. Their use was limited to supporting the writing and revision process, including improving clarity, organising content, and assisting with formatting-related tasks such as reference ordering. In addition, these tools were used in a supporting role similar to a search engine, for compiling relevant literature, challenging early brainstorming ideas, and providing critical feedback on documentation and report drafts; being of little use when reasoning about physical systems and more conceptually challenging tasks. Both locally run language models and Microsoft Copilot were used, accessed through Aalborg University resources. The use of the university-provided Copilot service was preferred where applicable, as it is offered under institutional data protection agreements intended to ensure data privacy.

AI tools were not used to generate technical content, analyse data, or develop arguments. All conceptual, analytical, and methodological work was carried out independently. The use of AI was therefore supplementary and did not influence the intellectual substance of the project.

**Aalborg University, January 24, 2026**\
Beltrán José Aceves Gil <baceve25@student.aau.dk>\
Eduard Brahas <ebraha25@student.aau.dk>\
Matei-Mihai Dragomir <mdrago25@student.aau.dk>\
Nasik Ali Khan <nkhan25@student.aau.dk>\
Rares , Monda <rmonda25@student.aau.dk>\
Pedro Felizardo Pedroso Carreira Lima <pfeliz25@student.aau.dk>

## Chapter 1 — Introduction

The notion of smart buildings originated in the twentieth century, with a 1984 New York Times article describing office buildings that combined building management and telecommunication [19]. Since then, various attempts to define the concept have emerged, including those by Sinopoli (2010) and Zhou and Yang (2018) et. al [3], although no consensus has been reached. Smart buildings feature, among other key components, Building Management Systems (BMSs) and Building Automation Systems (BASs). Architecturally, they have five main layers, spanning from "physical" to "user interface". Their benefits include increased energy efficiency, improved comfort, predictive maintenance, reduced costs, and greater sustainability [11].

For this project, Thomas Manns Vej 23 (TMV 23), a 2016 office building used for teaching and research at Aalborg University, serves as the case study and is represented as a data-driven thermal system. The building spans around 9000 m^2 over five storeys above the ground level and a single basement floor, featuring a range of different rooms, such as offices, group rooms, common areas, and laboratories. It accommodates around 150 staff members and 600 students, with a primary energy use of 56.3 kWh/m^2·year. TMV 23’s Heating, Ventilation and Air Conditioning (HVAC) system includes 14 ventilation systems, radiators, radiative panels and three cooling units. Its BMS, Schneider EcoStruxure™, provides occupancy detection and monitors indoor temperature, humidity, CO2 levels, lighting and ventilation [14].

Despite such an advanced infrastructure, many limitations have been reported in the operation of TMV 23. More specifically, the controllers in each room operate independently and do not communicate with one another, leading to high energy consumption. In addition, the automation controllers are challenged by varying conditions, such as changes in weather or unexpected heat sources, and do not adapt their behaviour based on the actions of other controllers. As a consequence of these limitations, there is a need for more predictive and coordinated control strategies that can better take advantage of the available data and system-level information. In this context, several potential approaches aimed at modelling and predicting building behaviour have been discussed. These include the use of industry tools for whole-building thermodynamic modelling, black-box multi-room thermodynamic models, alarm systems for anomaly detection, predictive control strategies, and globally coordinated controllers.

## Chapter 2 — Problem Statement and Proposed Solution

### 2.1 Problem Statement
TMV 23 features extensive hardware infrastructure that incorporates a variety of sensors and actuators in each room, yet its control system exhibits several shortcomings. Out of all the potential areas for improvement, this project focuses specifically on the absence of predictive and coordinated control mechanisms across rooms. The current BMS employs a reactive strategy using independent Proportional (P) controllers. These units respond only after detecting a sensor deviation, artificially adding latency into the system. This response delay creates problems due to the building’s high thermal inertia and asymmetric controllability. For instance, once a room overheats, the system cannot cool it down quickly as it relies on passive ventilation, and leads to prolonged occupant discomfort.

Beyond its impact on thermal and air-quality conditions, the current control scheme lacks coordination between actuatos, leading to energy inefficiencies. For example, the heating subsystem may supply heat to a zone while ventilation simultaneously removes it, resulting in unnnecessary energy consumption. The existing controllers operate by defining minimum and maximum values for individual variables and triggering control actions only when these bounds are violated. This threshold-based decision logic limits the ability to implement more advanced control objectives. In particular, this makes it difficult to satisfy strict building regulations that impose conflicting requirements, such as lowering municipal return water temperatures while at the same time preventing Legionella growth.

Due to their reactive nature, the controllers frequently overshoot and undershoot the defined setpoints, forcing actuators to cycle frequently. This behavior results in system oscillations and instability, primarily caused by the lack of coordination between the control loops. As a consequence, mechanical wear is increased and energy consumtion is amplified compared to steady-state operation. Moreover, the rigid control logic also makes it challenging to prioritise energy savings during vacancies or restrict actuator use. The building currently operates as a collection of independent devices rather than a single ecosystem. This prevents TMV 23 from realising its potential as an energy-efficient facility.

### 2.2 Proposed Solution
Changes are proposed for the control of the TMV 23 building. The objective is to introduce a predictive control framework that integrates with the existing building infrastructure. This approach addresses the limitations of the current reactive systems, which respond only after temperature deviations occur and therefore lead to inefficiency in actuator operation.

The new system operates with room-level precision. To function effectively, the system requires two core components: a predictive model and a controller. The predictive model is developed using a historical dataset spanning 10 months. The controller then utilizes this model to simulate future scenarios and determine the optimal action. Every 5 minutes, the controller evaluates the situation to balance priorities, including saving energy and maintaining occupant comfort, as set by the user [16].

The first milestone acts as the prerequisite to fulfil these objectives. The dataset is utilised to capture seasonal changes and usage patterns [15, 22]. For the core modelling method, blackbox identification is selected [20]. This approach is chosen because quality data is available, allowing the model to identify complex thermodynamic patterns directly. This approach was selected over "grey-box" methods [2] as it provides accuracy and scalability without the need for prior knowledge of the underlying physical equations.

The second step involves designing the constrained predictive and collaborative controller. This approach differs from conventional methods, such as Proportional-Integral-Derivative (PID) controllers. PID controllers operate in a primarily reactive manner, as control actions are computed from the current and past tracking error rather than from explicit predictions. In contrast, the predictive controller is more proactive. It plans actions for the future while following operational constraints. This capability is vital for managing conflicting needs, such as prioritising comfort or energy savings, based on the operator’s needs [13].

The final goal is the analysis and validation of the model using UPPAAL [10]. Simulations are run to observe the system’s thermal response under various conditions. Additionally, Statistical Model Checking (SMC) is used to perform probabilistic analysis. Since evaluating every possible state of a complex system is not feasible, it computes the probabilities of particular events instead. Although this method does not provide mathematical proof, it identifies safety thresholds that might otherwise be missed.

**Figure 2.1: Milestone graph**

![Milestone graph](images/MILESTONES%20GRAPH.png)

## Chapter 3 — Literature review

Smart buildings are built environments equipped with integrated sensing, communication, and automation systems that enable coordinated operation of key services such as heating, ventilation, air conditioning, lighting, and safety [15, 22]. Their primary objective is to improve energy efficiency, maintain indoor environmental quality, and support reliable building operation, emphasised by larger initiatives on energy-flexible buildings[13]. This functionality is achieved through continuous data acquisition from sensors and the use of control algorithms that adjust system behaviour based on observed conditions [16].

A defining feature of smart buildings is the ability to utilise data for informed decision-making [16]. While some implementations incorporate predictive analytics and advanced optimisation, these are not strict requirements [22]. At a minimum, smart buildings employ automation that responds dynamically to changes in occupancy, environmental conditions, and operational demands [15]. This way, interoperability between subsystems, secure exchange, and coordination remain essential. Within the scope of this work, smart buildings are understood as systems that leverage available data and automation to maintain comfort and operational efficiency[13]. Predictive modelling and advanced control strategies represent desirable enhancements but are not mandatory requirements [16]. The emphasis is placed on enabling building systems to interact, share information, and adapt dynamically to varying conditions without reliance on manual intervention [15].

Smart buildings rely on models to describe how indoor conditions evolve under external influences and system actions [21]. The literature distinguishes three main modelling paradigms: white-box, grey-box, and black-box approaches [2]. These approaches differ fundamentally in the degree to which the internal mechanisms driving system behaviour are transparent and interpretable.

White-box models are characterised by full transparency: all internal processes and parameters have direct physical interpretations, enabling examination and understanding of every mechanism that drives system outputs. Examples include ordinary differential equations that explicitly encode thermal dynamics, and they are based on physical principles, such as heat transfer and mass balance, which are widely used in classical building simulation [2, 21]. Such models provide high transparency and are ideal for design and new construction, but require complete documentation and detailed knowledge of building properties [1, 13].

Grey-box models combine simplified physics (e.g., lumped thermal networks) with parameter estimation from data, such as resistor-capacitor networks or continuous state-space models, calibrated against sensor or meter data [2, 21]. Examples include RC networks where unknown thermal resistances and capacitances are estimated from historical measurements. Grey-box models maintain partial interpretability: the structure remains grounded in physics, but some parameters or relationships are learned from data rather than derived from first principles. This hybrid approach reduces complexity while preserving physical meaning, making it suitable for control-oriented applications [1].

Black-box models, in contrast, rely entirely on data-driven techniques. Examples include neural networks or gradient boosting models trained directly on historical data [20]. These models operate with minimal or no requirement for knowledge of underlying physical mechanisms, and instead they learn functional relationships directly from observations. However, the resulting model is limited by the inability to inspect and understand its internal decision-making process, a defining characteristic of this technique. They can capture non-linear dynamics and interactions without explicit physical assumptions but demand large, high-quality datasets and often lack transparency and interpretability [21].

These differences have practical implications. White-box models are ideal for simulation and design but less feasible for rapid deployment in existing buildings, especially when documentation is incomplete or outdated. Grey-box models offer a compromise between accuracy and effort, and therefore explain their popularity for existing buildings and retrofit studies [21]. Black-box models are attractive when data is abundant and computational resources are available, yet their limited interpretability and poor generalisation under changing conditions remain concerns [20, 21].

Traditional reactive controllers, such as PID loops, dominate current practice because they are simple and robust; but they operate without foresight, adjusting only after deviations occur. Model Predictive Control (MPC) uses a model to anticipate system behaviour and optimise control actions over a future horizon, enabling constraint handling, multi-objective optimisation (comfort, cost, energy) and inclusion of external forecasts (weather, occupancy). However, MPC adoption is limited by the effort required to develop accurate models and the computational demands involved in real-time optimisation, the tradeoffs of model complexity vs. tractability for control are discussed in grey-box identification literature [2]. Learning-based methods, including reinforcement learning or data-driven control, aim to reduce reliance on explicit models. Yet they raise concerns about safety, convergence, and interpretability, especially in real, safety-critical building environments[10].

Recent research explores methods to reduce the complexity of model development through data-driven identification. For instance, machine-learning algorithms estimating thermal dynamics of buildings using grey-box or similar methods [20]. Neural-network-based and physics-informed techniques are also gaining ground: works such as “Physics Informed Neural Networks for Control-Oriented Thermal Modelling of Buildings” illustrate how combining physical constraints with data-driven learning can yield models that are both flexible and more data-efficient than purely black-box approaches [**Drgona_2021_PINNBuildings**]. Each approach presents trade-offs: sparse or data-driven regression-style methods are sensitive to noise, pure data-driven neural models may overfit, and physics-informed strategies require careful tuning and validation.

NODEs and continuous-time neural modelling frameworks are emerging as promising techniques for representing dynamical systems while leveraging neural networks: their differentiable structure facilitates integration with control algorithms and optimisation routines, enabling representation of dynamics learned from irregular or incomplete data. Recent building-modelling literature discusses such approaches and highlights their compatibility with control frameworks, while also flagging computational cost and regularisation as active challenges [20]. At the same time, establishing formal guarantees for neural or data-driven models remains difficult because the learned dynamics typically do not provide explicit, physically derived equations. Classical analysis methods such as Lyapunov-based stability, reachability analysis, or invariant-set computation assume an interpretable mathematical structure, which neural ODEs or black-box models often lack[10].

Formal verification and synthesis tools give a complementary perspective: for instance, the tool UPPAAL is a widely used environment for modelling and verifying discrete-time or real-time systems represented as timed automata, supporting correctness and safety guarantees for control strategies (reviewed in modelling/verification surveys) [24]. In contexts such as smart building control, especially when building subsystems, scheduling, or fault detection are involved, such formal methods can play a role. For example, studies on formal verification of smart building security or IoT-based control use UPPAAL to simulate and verify system operations under timing or safety constraints [4]. Thus, while data-driven and learning-based modelling and control approaches offer flexibility and efficiency, the trade-off between expressiveness, interpretability, and safety/guarantees remains a central issue, which strongly shapes research and practice in smart building modelling and control.

## Chapter 4 — Methodology and requirements

### 4.1 Project Management and Collaboration
All the actions regarding the management of the project, such as coordination, organisation, version control, and documentation, are tracked on the GitHub platform. The project follows a milestone-driven development approach in which each milestone corresponds to an independent part of the project, as defined in the proposed solution section. Each Milestone is split into goals using GitHub Issues [9], then the goals are divided into sub-issues representing individual tasks. This partitioning ensures the appropriate level of granularity for tracking the progress of each milestone. To guarantee uniformity, all issues follow a pre-defined template composed of a title and a multi-part description, which includes task description, expected outcome, checklist, and documentation. The progression of the tasks is plotted with the GitHub Project board tool [8]. At the same time, the standardised procedures for contribution are documented on the wiki feature of the project.

Version control is managed using Git in tandem with a trunk-based development protocol [18]. For traceability, all the feature branches follow the same naming convention: "user/issue_n/milestone_goal-short_desc". The same applies to the commits to ensure a clear and readable history; they are structured as follows: "[TAG]: short description". In addition to the Wiki, project documentation is maintained using MkDocs [23]. This tool generates a static website from Markdown files stored within the \docs folder of the project. Moreover, the plug-in library available for MkDocs provides integrations to make the documentation searchable, and the code snippets are enriched by syntax highlighting and aligned with the current implementations.

### 4.2 Systematic Exploration of Solutions
The technical work follows a process that is intended to navigate the research and development practices in an organised and traceable way. This framework supports iterative experimentation and progressive refinement of both models and control strategies. The initial stage involves identifying technical approaches and frameworks through literature reviews, documentation analysis, and internet searches for state-of-the-art tools. This stage aims to generate a broad solution space rather than focusing on a single approach. Afterwards, each approach is characterised by its theoretical base and technical requirements, as well as scalability, potential benefits, and drawbacks in the context of the project’s objectives. Particular emphasis is put on evaluating the compatibility with the available data, project philosophy, and integration with control components.

Since data availability and quality directly influence modelling and control performance, having a clear and explicit data strategy is a foundational component of the overall methodology. The project requires complete time-series data from sensors (e.g., temperature, CO2), actuators (e.g., radiator valve position) and system configurations. Data must be available at different spatial resolutions, including individual rooms, clusters of rooms, and complete building sections. To ensure numerical stability during training and facilitate meaningful model interpretation, the data should adhere to certain criteria, including minimal gaps, consistent and high-frequency time steps, synchronised data streams, and clearly labelled features. Finally, a data processing pipeline is applied to the dataset according to the modelling implementation. This might also include interpolation to handle possible missing values, normalisation (e.g., z-score) to scale features for model training, and minimising the impact of outliers. Furthermore, augmentation with relevant external data sources is also applied (e.g., weather conditions and estimated occupancy schedules).

## Chapter 5 — Design and Implementation

### 5.1 Data
The foundation of any data-driven modelling project is the data itself. For model identification, validation, and controller design, the data must be consistent, well-structured, and of high quality. If these conditions are not met, the resulting models can become inaccurate and unstable. This section outlines the data pipeline, from acquisition to the final processed dataset used for modelling.

The physical sensing infrastructure consists of a network of distributed sensors that continuously measure indoor temperature, CO2 concentration, and airflow rates. The building’s BMS regulates heating and ventilation through actuator commands such as valve openings, heater power, and damper positions. These actuators respond either to control logic on temperature and CO2 threshold setpoints or to user inputs, managed through a Schneider Electric EcoStruxure web-based control panel, managed by Campus Service. This interface provides real-time access to the BMS, allowing authorised staff to monitor current sensor readings and adjust control thresholds and setpoints.

Due to slow extraction times and persistent data quality issues in the BMS API, the analysis relies on a curated dataset obtained through the university’s research portal. This dataset, titled “Detailed operational building data for six office rooms in Denmark: Occupancy, indoor environment, heating, ventilation, lighting and room control monitoring with sub-hourly temporal resolution”, comprises the following parts: room-level indoor environmental quality (air temperature, CO2 concentration, lux sensor), presence detection from Passive Infrared Sensor (PIR), and occupancy measurements, along with artificial lighting, radiator valve, and Variable air volume (VAV) damper operational data in six different office rooms, measurements of the central Air Handling Unit (AHU), measurements of the central heating system supplying the radiator to the six office rooms and measurements of the outdoor conditions [17]. In total, the dataset contains 88705 time-stamped measurements, spanning 10 months with a fixed 5-minute sampling interval. This uniform temporal resolution ensures reliable representation of room-level thermodynamics. Moreover, it contains a very low fraction of missing values, ranging between 0.1% and 0.3%, which stands in contrast to the irregular and incomplete data available through the BMS API. Each sensor or actuator is clearly labelled, documented (measuring unit, limits on operating usage, number of data points, number of missing points, and missing data points in percentage), and organised.

The dataset requires several processing steps to be suitable for training. The remaining missing values are handled using linear interpolation, followed by back-filling and forward-filling. All features are standardised using z-score normalisation. Occupancy information is derived from the CO2_level measurements of each room, as the original PIR-based occupancy contains a high proportion of missing values. The processed dataset is partitioned into training and testing subsets using a chronologically split by month. One month is reserved for testing, while the remaining data is used for training. This strategy ensures evaluation on unseen data and provides a realistic assessment of the model’s generalisation capabilities. The resulting `train_data.csv` and `test_data.csv` files serve as the inputs to the modelling chapter.

**Figure 5.1: Dataset overview**

![Dataset overview](images/dataset_overview.png)

### 5.2 Modelling
Given the availability of a reliable dataset, the modelling step focuses on identifying a suitable representation of the building’s thermodynamic behaviour. The implementation of NODEs addresses the absence of detailed physical knowledge of room-level thermodynamics by relying on a data-driven approach to capture continuous-time dynamics, while also respecting thermodynamic relationships.

NODEs provide a modelling framework in which the system dynamics are represented as a continuous-time differential equation that is implemented as a neural network. Instead of predicting the next state directly, the model learns the underlying time derivative `dz/dt = f_θ(z(t), u(t))` (where `z` is the latent state and `u` is the interpolated control), which is then integrated numerically to obtain state trajectories. This formulation is particularly suitable for thermal dynamics, where the system’s evolution is inherently continuous and governed by differential relationships. Following the framework introduced by Chen et al. [5], this work is realised in PyTorch using the `torchdiffeq` library, which provides differentiable ODE solvers.

The original NODE formulation focuses on modelling the direct evolution of system states. Expanding on the original paper proposal, this implementation uses an encoder–decoder architecture and includes explicit control inputs, while also maintaining the continuous-time dynamics that characterise NODEs. The proposed NODE formulation therefore enables data-driven learning while remaining aligned with the physical structure of the problem, making it an appropriate fit for modelling room-level thermodynamic behaviour.

The model architecture consists of an encoder–latent ODE function–decoder structure. The encoder maps the initial observed state as sensor data and control inputs to a latent space representation, while the decoder reconstructs temperature predictions from the latent space. This architecture serves two purposes: (1) it reduces the dimensionality of the problem for the learning of the underlying dynamics, (2) the use of a latent space allows the model to learn a representation space where the temporal evolution may be more regular and easier to capture with an ODE when working with raw data, noise, and intricate relationships between data.

The encoder takes as input the concatenation of initial room temperatures `y0 ∈ R^{dy}` and initial control inputs `u0 ∈ R^{du}`, producing a latent state `z0 ∈ R^{dz}`:

`z0 = Encoder([y0, u0]; θ_enc)`  (5.1)

Both encoder and decoder are implemented as feed-forward networks with a hidden layer of 512 units and Rectified Linear Unit (ReLU) activations. The core of the NODE is the dynamics function `f_θ` that defines the time evolution in latent space. This function is implemented as a neural network that takes both the current latent state `z(t)` and time-varying control inputs `u(t)` as inputs:

`dz/dt = f_θ(z(t), u(t))`  (5.2)

The inclusion of control inputs directly in the dynamics function reflects our understanding of the thermal modelling problem. Room temperatures are governed by heat exchange processes influenced by three key factors: (1) external temperature, which drives heat transfer through the building envelope; (2) HVAC control states (valve positions, damper settings), which we explicitly include as they represent the actuators available for temperature regulation, and (3) disturbances and observations such as occupancy and solar radiation. By incorporating these variables into `u(t)`, the model can learn how each type of input affects the thermal dynamics, enabling its use for predictive control applications.

The model is intentionally built using a minimal subset of dataset columns that is sufficient to represent the system’s state, disturbances, and the effect of controllable actuators while maintaining accuracy. This design choice reduces the complexity and ensures compatibility with control tasks. The complete list of dataset columns used for model inputs and outputs is provided in the appendix A.

The ODE formulation requires continuous-time data sampling, the dataset however is not continuous but discrete. To address the ODE solver that may request the control at any intermediate time, a linear interpolator provides smooth control signals `u(t)` between discrete time steps. The dynamics network uses a deeper architecture with two hidden layers of 512 units each and Hyperbolic Tangent (Tanh) activation functions. The choice of Tanh over ReLU was motivated by the need for improving numerical stability during integration. The latent trajectories are obtained by numerically solving the ODE using the `torchdiffeq` library’s adaptive solvers. Fourth-order Runge–Kutta method (RK4) is used as the default solver, consistent with the solver choices discussed by Chen et al. [5]. At each time step `ti` in the prediction horizon, the decoder maps the latent state `z(ti)` back to the output space:

`ŷ(ti) = Decoder(z(ti); θ_dec)`  (5.3)

**Figure 5.2: Architecture graph of the encoder–latent ODE function–decoder**

![Architecture graph](images/full_model.png)

Training was performed on the Aalborg University (AAU) MCC3 high-performance computing cluster via a SLURM script that defines a pipeline to exploit PyTorch’s multi-threaded support. The trained model, alongside normalisation scalers, is saved to disk.

### 5.3 Control
The control step focuses on developing an interface between the model and UPPAL to allow strategy synthesis and evaluation. The primary challenge was not the implementation of control logic itself, but instead ensuring compatibility between the control framework (UPPAAL) and the neural network model.

The implemented bridge between the model and UPPAAL relies on UPPAAL’s external functions feature (this requires UPPAAL > 5.0 and UPPAAL Stratego > 4.1.20-7 [6]). The connector is engineered to bridge UPPAAL’s discrete, synchronous analysis with our continuous-time Python model through three main components. First, a C wrapper (`uppaal_wrapper.c`) performs a synchronous call from UPPAAL into an external service. The wrapper uses libcurl to send an HTTP request to the model API and uses cJSON [12] to parse the JSON response. Compiled as a linkable function, this wrapper is invoked directly by UPPAAL as an external function.

The second component is a lightweight REST API (`server.py`) implemented with FastAPI [7], and Uvicorn [25] that runs the model. The API accepts state queries (current temperatures, candidate control actions, disturbances) and returns the model outputs (predicted next state or short trajectory) as JSON. Keeping the API interface minimal and stateless is necessary to preserve UPPAAL’s semantics and reproducibility, and it maintains a response time of 6ms on average.

Finally, UPPAAL integrates this call with a timed automaton that calls the C bridge to query the neural model. The automaton uses the model response synchronously during strategy evaluation or synthesis. Figure 5.3 illustrates the interaction between UPPAAL, the C bridge, and the external neural model API.

**Figure 5.3: Order of action with the REST API and UPPAAL**

![Order of action REST API and UPPAAL](images/HTTP_TO_UPPAAL.png)

Because UPPAAL treats external calls as synchronous and expects deterministic behaviour, the connector and model must be stateless and return consistent outputs for identical inputs. Therefore, the current setup uses a saved model checkpoint for inference and avoids stochastic layers or sampling at query time.

This approach has implications for the types of UPPAAL analyses that can be performed. The main restriction arises with the use of floating-point (double) variables in the template. Using this type of variable prevents the use of the UPPAAL’s symbolic simulator and TiGa features. As a result, classic verification is not available, and instead analyses are handled with Stratego and SMC). To work around this limitation and regain access to classic features, implementing an encoder for converting doubles to integers and vice versa will be effective. While this allows TiGa/classic queries, the resulting state-space is too large for exhaustive exploration in most realistic scenarios.

These constraints change the verification from model checking to stochastic reachability and SMC-based characterisation. A set of controllers and synthesis workflows was implemented and evaluated using the connector pattern previously described. The set of controllers consists of a random baseline to reactive bang-bang strategies with global and per-room threshold logic, and an online policy learning controller. All of these controllers request model predictions during simulation to guide the action selection.

Finally, practical considerations follow from this design. The modular nature of the connector, in which the model is wrapped in a stand-alone REST API, allows the model to be replaced or updated without the need for intervention on the UPPAAL templates. This separation is valuable in settings where the model might undergo retraining, ensuring that controllers remain compatible with updated dynamics. Maintenance-wise, the connector is bound to the platform for which it is built, meaning that changes in the execution environment will require corresponding updates to the bridge layer.

![UPPAAL template and bridge illustration](images/controlpanel_sensors.png)

## Chapter 6 — Results and Evaluation

The model’s predictive capability is evaluated by comparing predicted temperatures with ground-truth values from a test split of the dataset, by using the same control inputs and external observations, ensuring consistency across all test scenarios. Two operating modes are used to run the model: open and closed-loop simulation.

**Figure 6.1: NODE performance comparison in closed-loop simulation**

![Closed-loop performance](images/multi_run_comparison_closed.png)

In open-loop evaluation, it receives actual temperature values from the dataset at each time step, which serve as initial state for the following prediction horizon window. This setup more closely mimics how the model would operate in a real predictive controller scenario, where sensor feedback continuously corrects the model’s internal state before generating new predictions. Consequently, open-loop performance provides a more realistic assessment of the model’s utility for practical control applications. In contrast, closed-loop evaluation feeds the model’s own predictions back as inputs for subsequent steps. While this approach does not reflect real-world operation, it is necessary for evaluating model behaviour under control inputs that differ from those in the training dataset, such as when exploring different control strategies.

**Figure 6.2: NODE performance comparison in open-loop simulation**

![Open-loop performance](images/multi_run_comparison_open.png)

The following evaluation covers prediction horizons from 1 to 16 steps, with each step corresponding to a 5-minute interval. Performance is quantified using two standard metrics: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), computed across all six rooms in the dataset. Figure 6.2 presents temporal prediction charts comparing actual temperatures against predicted temperatures for each room across all four horizons (`H = 1, H = 4, H = 8, H = 16`).

Open-loop performance demonstrates good accuracy across all evaluated horizons, as shown in Figure 6.2. The shortest horizon, `H = 1`, achieves the lowest errors, as predictions require minimal extrapolation beyond the current observed state. Performance degrades slightly at longer horizons (`H = 4, H = 8, H = 16`), as expected from accumulated modelling uncertainty over extended prediction windows. However, the error metrics in Figure 6.4 reveal that this degradation remains moderate across all rooms, indicating the model captures meaningful temperature dynamics even at extended horizons.

Closed-loop performance exhibits significantly different behaviour, as seen in Figure 6.1 and quantified in Figure 6.3. At `H = 1` and `H = 2`, prediction accuracy is poor, likely stemming from the model’s training procedure, which used moving windows of length 16 to improve longer-horizon predictions rather than one-step-ahead accuracy. This training choice prioritises performance across extended horizons at the cost of single-step fidelity. Notably, performance improves significantly at `H = 4`, aligning better with the model’s training objective and supported by the error metrics. Performance then plateaus at `H = 8` and `H = 16`, suggesting the model has adapted to capture dynamics at these intermediate and longer scales.

**Figure 6.3: Performance comparison through prediction horizons on closed-loop simulation**

![Closed-loop horizons](images/metrics_comparison_mae_closed.png)

**Figure 6.4: Performance comparison through prediction horizons on open-loop simulation**

![Open-loop horizons](images/metrics_comparison_mae_open.png)

Additionally, the model relies on external observations not derived from system states: weather conditions, solar irradiance, and outside temperature. These observations represent data available from sensors but are treated as external inputs to the predictive model. To assess the importance of these external forecasts, an additional open-loop evaluation scenario freezes these observations at constant values, simulating a scenario without forecasting data. Figure 6.5 presents this comparison, which reveals a slight and clear degradation in model performance compared to the baseline open-loop case. This suggests that accurate external forecasts could meaningfully improve the model’s predictive accuracy, and underscores the importance of reliable weather and environmental data for effective model operation.

**Figure 6.5: Performance comparison comparing frozen observations between window predictions and the baseline**

![Frozen vs baseline](images/model_frozen_observations.png)

Four control strategies are implemented and evaluated through UPPAAL to assess their effectiveness in managing room temperatures. A random controller serves as a baseline, issuing actuator commands without any feedback mechanism or learned strategy. It provides minimal bias and acts primarily as a sanity check, confirming that informed control outperforms arbitrary actions.

**Figure 6.6: Random controller performance on NODE simulation through UPPAAL**

![Random controller](images/random_controller.png)

Bang-bang controllers represent a step toward reactive control. A global bang-bang controller operates on building-wide temperature thresholds, issuing commands to all actuators simultaneously when the average temperature deviates from a target range. This approach introduces a critical limitation: when individual rooms follow opposing temperature trends over a threshold, a global decision cannot address room-specific needs, creating deadlocks.

A per-room bang-bang controller addresses this by maintaining independent threshold logic for each room’s heater and ventilation actuators. Figure 6.7 showcases this improvement: the global strategy fails to recover from imbalanced states, while the per-room approach successfully maintains all rooms within acceptable bounds by allowing independent actuator control.

**Figure 6.7: Global and per-room bang-bang controllers performance on NODE simulation through UPPAAL**

![Bang-bang comparison](images/controller_BB_per_room.png)

Finally, an online policy learning controller is implemented to represent predictive and collaborative control schemes. Rather than following fixed rules, it learns an optimal control policy through interaction with the system model in UPPAAL. The controller’s state space comprises current room temperatures and actuator states, its action space consists of binary decisions for heater and ventilation actuators in each room, applied independently. The learning objective balances two competing goals: minimising total temperature error and reducing actuator use, with a configurable weight of the total cost function. This weighting reflects the capability of balancing or prioritising different goals. The policy learns to anticipate thermal dynamics and coordinate actuator use across rooms, avoiding the deadlock situations encountered by the global bang-bang controller while maintaining better energy efficiency than the per-room bang-bang baseline.

However, this learning-based approach carries inherent limitations. The policy is learned within a specific simulation environment and may not generalise perfectly to real-world conditions with unforeseen dynamics or to scenarios significantly different from the training distribution. To aid in reducing these limitations, rather than exploring the state space and its optimal solutions more eagerly, the learning process can be carried online over the current control and simulation window in a repeating fashion. In such a configuration, strategies are learned for specific scenarios and discarded immediately, rather than stored, retrained and reused to achieve better generalisation capabilities.

**Figure 6.8: Policy learning controller performance on NODE simulation through UPPAAL**

![Policy learning](images/learning_controller_acum.png)

**Figure 6.9: Stability analysis visualisation for upper and lower temperature bounds, on NODE simulation through UPPAAL**

(a) Lower bound analysis — (b) Upper bound analysis

![Stability bounds](images/stability_diff_temp_closed.png)

Traditional dynamical systems are modelled as ordinary differential equations, enabling classical control theory to prove system properties analytically. The NODE model, learned by a black-box neural network rather than derived from physical principles, prevents classical analytical approaches. SMC in UPPAAL provides an alternative by treating each property as a stochastic reachability question and simulating scenarios to compute the probability that classical control properties hold true.

**Figure 6.10: Equilibrium under no actuator control, on NODE simulation through UPPAAL**

![Equilibrium no control](images/equilibria_diff_temp_closed.png)

Out of the more common key characteristics examined in control systems, the current methodology is able to analyse the following: equilibrium behaviour, local stability, and controllability. Equilibrium points are states where system dynamics stabilise. It is commonly defined as `x˙ = f(x,u) = 0`. For the NODE, this cannot be verified analytically, so the property is formulated as the probabilistic query: `Pr[<=T](<> total_t_derivative < threshold)`. This query measures the probability that the sum of all room temperature rates of change falls below a threshold within time horizon `T`. Three distinct equilibria emerge depending on control strategy, achieving a query result `P(equilibrium) >= 0.95`. Under active heating, the system reaches thermal saturation as shown in Figure 6.9b. Under passive cooling, the system reaches the opposite equilibrium as displayed in Figure 6.9a. Under no active control, temperatures equilibrate with outside conditions as seen in Figure 6.10. In each case, temperature derivatives converge toward zero, confirming equilibrium behaviour.

Stability is a characteristic of systems that return to equilibrium after suffering perturbations. Lyapunov stability is formally defined as: if `||x(0) − x*|| < δ`, then `||x(t) − x*|| < ε` for all `t ≥ 0`. The SMC formulation uses the probabilistic query `Pr[<= T]((time > T/2)&&( |t_avg(T/2) − t_avg(T)| < margin))` to check convergence within a specified tolerance after sufficient time has elapsed. To assess this property, a UPPAAL template is used to initialise the system at one of the equilibrium points identified in the previous section. A bounded perturbation is then applied for a fixed number of time steps, after which the original actuator configuration is restored. Stability is then evaluated by verifying whether the state returns to the same equilibrium region, achieving a query result `P(stability) >= 0.95`. Figure 6.11 demonstrates this behaviour following perturbation under the Closed controller. The system deviates momentarily, then returns to its original temperature range, indicating local stability around the equilibrium point.

**Figure 6.11: Stability analysis visualization on NODE simulation through UPPAAL**

![Stability visualisation](images/stability_diff_temp_warm.png)

Reachability defines the set of states achievable from actuator inputs: a target state `xT` is reachable if there exists a control input sequence `u(t)` that drives the system to `xT` in finite time. Controllability characterises the extent to which actuators can influence system dynamics: it measures which regions of the state space are reachable through control action, `∀x0, xT : ∃u(t) : x0 → xT`. Assuming continuity of the system’s control dynamics, if the upper and lower bound states `xmin` and `xmax` are reachable, then all intermediate states are also reachable. Since system trajectories are continuous, the system must pass through intermediate states when moving between extremes. Therefore, testing the reachability of the bounds establishes controllability across the entire interval, defining a reachable envelope.

The SMC formulation uses the probabilistic queries `Pr[<= T](<> t_avg >= max_temp)` and `Pr[<= T](<> t_avg <= min_temp)` to determine the bounding reachable temperatures within time horizon `T`. To evaluate controllability limits, the system is initialised with all rooms at outside temperature using two control strategies: a Warm controller to determine maximum reachable temperature, and a Cold controller to determine minimum reachable temperature. Results in Figure 6.12 show that the reachable envelope spans from outside temperature minus 3 degrees Celsius to outside temperature plus 5 degrees Celsius, achieving a query result `P(reachability) >= 0.95`. These asymmetric limits reflect physical constraints of the building’s thermal system, which features active heating but not mechanical cooling.

The NODE model demonstrates suitability for smart building control applications. The model evaluation shows accurate prediction of future room temperatures across multiple prediction horizons. The SMC analysis confirms the model responds to control inputs and exhibits dynamical system properties expected from building thermal behaviour: the system reaches equilibrium under different control regimes, returns to equilibrium following perturbations, and operates within a reachable envelope defined by physical constraints.

**Figure 6.12: Controllability analysis showing temperature range reachability under Warm and Cold control strategies on NODE simulation through UPPAAL**

![Controllability envelope](images/controlability_warm_20.png)

## Chapter 7 — Discussion

The development of the final modelling and control framework followed an iterative process of exploration. Several candidate approaches were prototyped and evaluated with respect to applicability for the problem, integration effort, and practicality for data-driven building control.

An initial investigation considered symbolic system identification using Sparse Identification of Nonlinear Dynamical systems (SINDy). This approach was explored because of its ability to discover interpretable differential equations directly from data. While rapid identification was possible, the resulting models proved difficult to validate and integrate into a simulation and control workflow. In particular, simulation performance and robustness under realistic noise levels were insufficient for downstream control tasks. Given these practical limitations, further refinement of symbolic approaches was abandoned.

To investigate whether partial physical structure could improve learning, Physics Informed Neural Network (PINN) were attempted using DeepXDE. A simplified ordinary differential equation was embedded into the loss function to guide learning. Training for this approach was successful, but the predictive performance did not meet the accuracy requirements for control. In contrast, a purely data-driven neural network achieved better results with a relatively similar training effort. This indicated that, for the assumptions made, simplified physics constraints were not practical.

Subsequently, a neural latent-state modelling approach was considered using the Neuromancer framework, implementing an encoder–decoder latent ODE architecture. Closed-loop evaluation using real measurements showed promising short-horizon behaviour. However, limitations related to long-horizon stability and model portability were encountered. In particular, exporting the trained models for use in external control frameworks was not supported, limiting their use in the broader system architecture.

Model predictive control (MPC) was also taken into account for the controller part. However, integrating the learned model into existing MPC frameworks proved impractical given the lack of support for complex neural-network-based models.

To support evaluation and gain insight into model characterisation, simple rule-based controllers were implemented within a custom Python simulation environment. These controllers provided a baseline for both open and closed-loop testing and guided the latter control design decisions.

Overall, the exploration progressed from symbolic and hybrid methods to fully data-driven models, followed by formal controller synthesis using UPPAAL Stratego. This progression reflects a trade-off between interpretability, integration effort, and reliability. The final architecture combines a PyTorch-based NODE model with UPPAAL synthesis, which was selected as it best satisfied the project’s accuracy, portability, and implementation goals.

## Chapter 8 — Conclusions and further work

### 8.1 Conclusions
The NODE model demonstrates sufficient performance and accuracy for smart building control applications. The model evaluation shows accurate prediction of future room temperatures across multiple prediction horizons. The SMC analysis confirms the model responds to control inputs and exhibits dynamical system properties expected from building thermal behaviour: the system reaches equilibrium under different control regimes, returns to equilibrium following perturbations, and operates within a reachable envelope defined by physical constraints.

The integration of UPPAAL with external models through C function calls and REST API calls provides a flexible framework for controller synthesis and verification. This approach enables direct coupling with arbitrary machine learning models, including the entire PyTorch ecosystem, opening possibilities for hybrid verification of modern learning-based control systems.

In the presence of black-box or non-ODE models lacking analytical interpretability, SMC in UPPAAL provides a powerful alternative. Reframing system characterisation as stochastic reachability queries of properties under hypothesis scenarios enables stochastic verification and property checking without requiring model interpretability or mathematical insight.

Per-room bang-bang control outperforms global control strategies by avoiding deadlock situations when rooms have conflicting thermal requirements. The online policy learning controller demonstrates further improvements by balancing temperature accuracy and actuator efficiency, though generalisation to scenarios outside the training distribution remains limited.

External forecasts, including weather conditions and solar irradiance, contribute meaningfully to model prediction accuracy, suggesting that reliable environmental data and predictive capabilities around it are critical for effective building control.

Automatic system identification is a rapidly advancing field with strong technological support through diverse libraries and methods. However, selecting the appropriate modelling approach still requires making informed assumptions about the system’s characteristics, dynamics, and control structure. The choice of model architecture, training objectives, and evaluation methodology depends fundamentally on understanding the physical system being modelled.

### 8.2 Further work
This study unlocks several directions for future work. A first step is real-world deployment and validation, where the NODE and the proposed controllers are tested on the actual building rather than in a simulated scenario. The deployment in this setting would allow the evaluation under realistic conditions and provide a deeper insight into the modelled dynamics.

Furthermore, the enhancement of the expressivity of the controllers could be the next step: instead of limiting actuators to binary decisions, controllers could use a wider range of setpoints. Increasing the action space would allow a fine-grained temperature regulation. Additional work could include enhancing the existing objective optimisations (actuator wear and comfort) with additional goals such as energy cost and carbon footprint.

Another direction is transfer learning across buildings, where the ability of the learned models to generalise to different buildings is evaluated, instead of retraining models for each individual scenario. Robustness analysis would assess controller performance under sensor noise, actuator faults, and modelling errors. In addition, architectural modifications could be explored to improve short-horizon prediction accuracy for closed-loop operations, where the errors can build up over time.

Finally, long-horizon planning could incorporate known states, controls, observations, or desired trajectories that fall outside the prediction horizon (e.g., scheduled occupancy patterns like room bookings) to enable anticipatory control strategies. Related to this, occupancy-aware control could be introduced by using booking systems that schedule the occupancy to enable predictive pre-heating or cooling strategies.

## Bibliography

[1] Aalborg University Student. “Grey-box modelling of buildings”. MA thesis. Aalborg University, 2019. <https://projekter.aau.dk/projekter/files/306658337/MasterThesis.pdf>.

[2] Peder Bacher and Henrik Madsen. *Experiments and Data for Building Energy Performance Analysis: Procedure for identifying models for the heat dynamics of buildings*. Technical University of Denmark, 2010. <https://henrikmadsen.org/wp-content/uploads/2014/10/Report_-_2010_-_Experiments_and_Data_for_Building_Energy_Performance_Analysis.pdf>.

[3] David Blum et al. *Data-Driven Smart Buildings: State-of-the-Art Review*. CSIRO on behalf of the International Energy Agency (IEA EBC), 2023. <https://annex81.iea-ebc.org/Data/publications/Annex%2081%20State-of-the-Art%20Report%20(final).pdf>.

[4] CEUR Workshop Proceedings. “Security Implementation and Verification in Smart Buildings”. In: *CEUR Workshop Proceedings* 2589 (2020). <https://ceur-ws.org/Vol-2589/Paper8.pdf>.

[5] Ricky T. Q. Chen et al. “Neural Ordinary Differential Equations”. 2019. arXiv:1806.07366. <https://arxiv.org/abs/1806.07366>.

[6] UPPAAL documentation: External Functions. <https://docs.uppaal.org/language-reference/system-description/declarations/external-functions/>.

[7] FastAPI. <https://fastapi.tiangolo.com/>.

[8] GitHub. About Projects — GitHub Docs. <https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects>.

[9] GitHub Issues. <https://github.com/features/issues>.

[10] Rachid Hadjidj et al. “Towards a reliable smart city through formal verification”. *Computer Communications* 176 (2021). <https://www.sciencedirect.com/science/article/abs/pii/S0140366421003352>.

[11] IBM. “What Are Smart Buildings?” <https://www.ibm.com/think/topics/smart-buildings>.

[12] Dave Gamble. cJSON: Ultralightweight JSON parser in ANSI C. <https://github.com/DaveGamble/cJSON>.

[13] IEA EBC Annex 67. *Principles of Energy Flexible Buildings*. DTU / Aalborg University, 2020. <https://orbit.dtu.dk/files/212198399/principles_of_energy_flexible_buildings.pdf>.

[14] Hicham Johra. *General Study Case Description of TMV 23: A Multi-Storey Office Building and Living Lab in Denmark*. DCE Technical Reports 306. Aalborg University, Jan. 2023. <https://doi.org/10.54337/aau511019002>.

[15] Hicham Johra et al. “Treatment and analysis of smart energy meter data from a cluster of buildings connected to district heating: A Danish case”. *E3S Web of Conferences* 172 (2020), p.12004. <https://doi.org/10.1051/e3sconf/202017212004>.

[16] Daniel Leiria. “From Smart Heat Meters to Diagnostics: Data-Driven Methodologies for Building Efficiency Assessment within District Heating”. PhD thesis. Aalborg University, 2024. <https://doi.org/10.54337/aau749598601>.

[17] Simon Pommerencke Melgaard et al. “Detailed operational building data for six office rooms in Denmark: Occupancy, indoor environment, heating, ventilation, lighting and room control monitoring with sub-hourly temporal resolution”. *Data in Brief* 54 (Mar. 2024), p.110326. <https://vbn.aau.dk/en/publications/detailed-operational-building-data-for-six-office-rooms-indenmar/>.

[18] Paul-Hammant. Trunk based development. <https://trunkbaseddevelopment.com/>.

[19] Frank Prial. “Wiring Buildings for Intelligence”. *The New York Times*, May 13, 1984. <https://www.nytimes.com/1984/05/13/realestate/wiring-buildings-for-intelligence.html>.

[20] Juan Pablo Real, Peder Bacher, et al. “Characterisation of thermal energy dynamics of residential buildings with scarce data”. 2021. <https://vbn.aau.dk/ws/portalfiles/portal/465270377/Characterisation_of_thermal_energy_dynamics_of_residential_buildings_with_scarce_data.pdf>.

[21] — same as [20].

[22] Markus Schaffer et al. “Dataset of smart heat and water meter data with accompanying building characteristics”. *Data in Brief* 52 (2024), p.109964. <https://doi.org/10.1016/j.dib.2023.109964>.

[23] MkDocs Team. MkDocs. <https://www.mkdocs.org/>.

[24] UPPAAL Team. UPPAAL: Home. <https://uppaal.org/>.

[25] Uvicorn. <https://uvicorn.dev/>.

## Appendix A — Dataset Columns Used for Model Training

The following table lists all columns used as inputs for the NODE, grouped by category.

**Category** | **Variable**
---|---
Sensor | Outdoor:Solar__direct_radiation__east_façade
Sensor | Outdoor:Solar__direct_radiation__south_façade
Sensor | Outdoor:Solar__direct_radiation__west_façade
Sensor | Outdoor:Temperature_air
Sensor | RoomA:Sensor__room_temperature
Sensor | RoomB:Sensor__room_temperature
Sensor | RoomC:Sensor__room_temperature
Sensor | RoomD:Sensor__room_temperature
Sensor | RoomE:Sensor__room_temperature
Sensor | RoomF:Sensor__room_temperature
Sensor | Ventilation:Sensor__air_temperature__supply
Actuators | RoomA:Radiator__control_signal__motor_valve
Actuators | RoomA:Damper__position
Actuators | RoomA:AHU__active
Actuators | RoomB:Radiator__control_signal__motor_valve
Actuators | RoomB:Damper__position
Actuators | RoomB:AHU__active
Actuators | RoomC:Radiator__control_signal__motor_valve
Actuators | RoomC:Damper__position
Actuators | RoomC:AHU__active
Actuators | RoomD:Radiator__control_signal__motor_valve
Actuators | RoomD:Damper__position
Actuators | RoomD:AHU__active
Actuators | RoomE:Radiator__control_signal__motor_valve
Actuators | RoomE:Damper__position
Actuators | RoomE:AHU__active
Actuators | RoomF:Radiator__control_signal__motor_valve
Actuators | RoomF:Damper__position
Actuators | RoomF:AHU__active
Actuators | Heating:Control__setpoint_water_temperature__supply
Disturbances | RoomA_is_occupied
Disturbances | RoomA:Window__opened_closed
Disturbances | RoomB_is_occupied
Disturbances | RoomB:Window__opened_closed
Disturbances | RoomC_is_occupied
Disturbances | RoomC:Window__opened_closed
Disturbances | RoomD_is_occupied
Disturbances | RoomD:Window__opened_closed
Disturbances | RoomE_is_occupied
Disturbances | RoomE:Window__opened_closed
Disturbances | RoomF_is_occupied
Disturbances | RoomF:Window__opened_closed

