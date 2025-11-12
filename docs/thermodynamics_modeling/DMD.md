# PyDMD
This module performs dynamic mode decomposition (DMD) analysis on building sensor, actuator, and configuration data. It includes preprocessing via interpolation, normalization, dimensionality reduction using PCA, and multi-resolution decomposition using DMD variants (DMD, HODMD, MrDMD).

The goal is to extract interpretable dynamic patterns from building data for modeling, control, and diagnostics. 

>It was just a test to see how it compares with pySINDy

::: thermodynamics_modeling.dynamic_mode_decomposition.pyDMD_test

## PyDMD vs pySINDy

### **PyDMD (Dynamic Mode Decomposition)**

*   **What it does**: PyDMD decomposes time-series data into spatial-temporal modes and their dynamics (frequencies and growth/decay rates).
*   **Output**: A set of modes and eigenvalues that describe how the system evolves over time.
*   **Modeling style**: **Data-driven spectral analysis**, not a direct equation-based model.
*   **Control theory relevance**:
    *   Useful for **modal analysis**, **reduced-order modeling**, and **system identification**.
    *   Can be used to build surrogate models for simulation or prediction.
    *   But it **does not produce explicit differential equations** or control-affine models.

***

### **PySINDy (Sparse Identification of Nonlinear Dynamics)**

*   **What it does**: Learns **explicit governing equations** (e.g., ODEs) from data using sparse regression.
*   **Output**: A symbolic model like:
    $$ \dot{x} = Ax + Bx^2 + C\sin(x) + \dots $$
*   **Modeling style**: **Equation discovery** — ideal for control design, simulation, and analysis.
*   **Control theory relevance**:
    *   Directly usable for **controller design**, **observer synthesis**, and **stability analysis**.
    *   Can incorporate control inputs (e.g., SINDy with control).

***

### PyDMD vs PySINDy

| Feature                   | PyDMD                         | PySINDy                 |
| ------------------------- | ----------------------------- | ----------------------- |
| Output                    | Modes + dynamics              | Explicit equations      |
| Model type                | Spectral / modal              | Symbolic / differential |
| Control design use        | Indirect (via reduced models) | Direct (equation-based) |
| Handles nonlinear systems | Limited                       | Yes                     |
| Time-scale decomposition  | Yes (via MrDMD)               | No                      |
