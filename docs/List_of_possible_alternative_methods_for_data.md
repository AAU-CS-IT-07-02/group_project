**List of possible alternative methods**

TO DO (in progress): write examples on how to use these alternatives

Here is a curated list of possible alternative methods (note that these complement MPC):

-   A lumped RC thermal model (simplified representation of thermal system) could be used for basic thermodynamics. Key libraries for this are numpy, scipy, matplotlib. The advantage of using RC-networks to represent buildings is that they can be mathematically modelled by a set of first order differential equations, also called state-space systems. The integration of these systems provides the variables of the model (temperatures of building elements and zones) at a relatively low computational cost. Constructing building models with RC-networks implies representing every element of the building with resistors and capacitors. Including all the layers in the construction for all the surfaces of the envelope leads to large RC-networks; to integrate the set of differential equations of a state-space system the time is discretised using a time step and the variables can be obtained using xn+1 = e^(AΔt) * xn + Ku where xn + 1 and xn are vectors representing the variables in time step n + 1 and n, u is the vector of inputs, and e^(AΔt) and K are calculated matrices to integrate the state-space system. The vector of variables (xn) has to be multiplied for the matrix e^(AΔt) in every time step and so has to be done with matrix K and u. The matrix e^(AΔt) could be calculated once for the whole year simulation. Please see example included in the respective file, where we model a simple room with: Thermal capacitance (C), which represents how much heat the room stores; thermal resistance (R), which determines how easily heat flows between inside and outside; heat input (Q_in) from HVAC or internal gains; and outdoor temperature (T_out) as a time-varying input. The differential equation describing the system is: C * dT_in/dt = (T_out - T_in)/R + Q_in. (https://www.sciencedirect.com/science/article/abs/pii/S0378778813000315)

## Example of how to use RC networks
::: alternative_methods.RC_network_example

-   Grey-box parameter estimation could be used to fit physical parameters to data and we could use scipy.optimize and pandas. This is a procedure where model parameters are estimated in order to make sure that selected model predicts output (e.g. internal temperature) that is comparable to measured values. Please see example included in the respective file, where we use the minimize function. (https://projekter.aau.dk/projekter/files/306658337/MasterThesis.pdf)

## Example of how to do parameter estimation
::: alternative_methods.Parameter_estimation_example


