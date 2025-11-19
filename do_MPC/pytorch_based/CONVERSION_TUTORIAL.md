# Converting a PyTorch Surrogate into do-mpc (CasADi) — Tutorial

Last updated: 2025-11-19

## Overview

This document explains how the example files in `do_MPC/pytorch_based` work together: `main.py`, `template_converter.py`, `template_model.py`, `template_mpc.py`, and `template_simulator.py`.

Purpose: show how a trained PyTorch feed-forward neural network is converted into a CasADi/do-mpc surrogate model, used by an MPC controller, and compared to a physics-based (real) plant in closed-loop simulation.

Audience: engineers converting trained PyTorch models into symbolic surrogates for MPC, and LLMs used as developer assistants for the conversion and test automation.

---

## Domain

- Model Predictive Control (MPC) for dynamical systems.
- do-mpc framework (Python) built on CasADi for symbolic modeling and optimization.
- PyTorch for defining and training neural-network-based surrogates of system dynamics.

Use case: Use a trained neural network as the system dynamics model inside MPC so the controller optimizes against the surrogate; validate surrogate against a physics model in simulation.

---

## File responsibilities (concise)

- `main.py` — example runner: builds or loads a PyTorch model, converts it to a CasADi surrogate, creates MPC and simulators, runs closed-loop simulation, and plots results.
- `template_converter.py` — converter: translates a PyTorch `torch.nn.Sequential` feed-forward net (Linear + Tanh) into a CasADi `do_mpc.model.Model` (discrete surrogate).
- `template_model.py` — a small continuous-time physics model (mass-spring-damper) used as the “real” plant.
- `template_mpc.py` — MPC setup for a model: sets horizon, timestep, objective, input/state bounds, and sets up the controller.
- `template_simulator.py` — simple simulator creation: sets `t_step` and calls `setup()`.

---

## How they work together (execution flow)

1. `main.py` defines a PyTorch Sequential model and loads pretrained weights (`model_weights.pth`).
2. It calls `template_converter(nn_model)` to transform the PyTorch net into a CasADi `do_mpc` discrete surrogate (states vector + inputs vector → next states).
3. `template_mpc(surrogate_model)` constructs an MPC controller that references `model.x['states']` and `model.u['inputs']`, sets `t_step` and `n_horizon`, objective and constraints.
4. Two simulators are created: one for the surrogate (discrete), and one for the real continuous model (`template_model`). Both are given the same initial `x0` and `set_initial_guess()` is called.
5. Closed-loop loop (50 steps):
   - `u_0 = surrogate_mpc.make_step(x0=x0)` — MPC computes control based on surrogate.
   - `real_simulator.make_step(u0=u_0)` — apply to real plant.
   - `surrogate_simulator.make_step(u0=u_0)` — apply to surrogate.
6. Data collected and plotted to compare real vs surrogate behavior.

---

## Deep dive: `template_converter.py` (what it does and limitations)

- Creates `surrogate_model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')`.
- Declares `states` (shape `(2,1)`) and `inputs` (shape `(1,1)`).
- Stacks `states` and `inputs` into a single CasADi `input_layer` (`vertcat`).
- Iterates `nn_model` layers:
  - `torch.nn.Linear`: extracts `weight` and `bias` to numpy and builds `ca.mtimes(weight, prev) + bias`.
  - `torch.nn.Tanh`: applies `ca.tanh` on the expression.
  - Other layer types: raise `RuntimeError` (not supported).
- Sets the computed `output_layer` as RHS for `states` and calls `setup()`.

Limitations & assumptions:
- Only supports `Linear` and `Tanh` from PyTorch Sequential.
- Assumes first linear layer input size equals (#states + #inputs), here 3.
- Uses CasADi `SX` symbolic type (works well for small models; for large networks consider `MX`).
- No support for ReLU, Sigmoid, BatchNorm, Dropout, convolutions, skip connections, or custom modules.

Suggested extensions:
- Add support for ReLU, Sigmoid, LeakyReLU (using CasADi equivalents and `if_else` where needed).
- Add dimensionality checks with clear error messages.
- Add an option to generate `ca.Function` wrappers for fast evaluation and unit testing.
- Optionally support ONNX → CasADi conversion pipeline for complex models.

---

## Mapping between PyTorch and CasADi (practical notes)

- When converting:
  - Move PyTorch model to `eval()` and CPU before reading `.weight` / `.bias`.
  - Verify `weight.shape` and `bias.shape` and ensure they match the symbolic vector shapes.
  - Convert biases to CasADi column vectors if needed (use `ca.DM(bias).reshape((-1,1))`).
- Choose `SX` vs `MX` depending on network size and derivative needs.

---

## How an LLM should work with these files (roles & prompts)

LLMs are useful for:
- Explaining code and documenting intent (what this doc is doing).
- Generating improved converter code that supports more PyTorch layers.
- Creating unit tests that verify numerical equivalence between PyTorch forward and the CasADi surrogate.
- Producing automation scripts to infer (#states, #inputs) and auto-generate the converter scaffolding.

Example prompts to give an LLM:
- "Refactor `template_converter.py` to also support ReLU, Sigmoid and LeakyReLU; add shape checks and create a `casadi.Function` named `surrogate_rhs` to test equivalence with PyTorch for random inputs."
- "Generate `tests/test_converter.py` (pytest) that loads `model_weights.pth`, picks 10 random states/inputs, and asserts |torch_out - casadi_out| < 1e-6."

Limitations to note to an LLM:
- LLM cannot run the tests here — ask it to produce runnable tests and scripts that you (the developer) will execute locally.
- For non-sequential or custom modules, the converter must be extended using the network's forward implementation or by exporting to ONNX.

---

## Conversion checklist (practical steps to adapt into your own system)

1. Dependencies
   - Ensure `python`, `torch`, `casadi`, `do-mpc`, `numpy`, `matplotlib` installed.
   - Use a virtual environment. Example (Windows `cmd.exe`):
     ```
     python -m venv .venv
     .venv\Scripts\activate
     python -m pip install -r requirements.txt
     python -m pip install casadi do-mpc torch matplotlib
     ```

2. Confirm canonical ordering for states and inputs (vectorized vs named states). Update code to use a single consistent mapping.

3. Prepare PyTorch model
   - Put model in eval mode: `model.eval()`.
   - Save weights: `torch.save(model.state_dict(), 'model_weights.pth')` (or export to ONNX when necessary).

4. Convert model
   - Use or extend `template_converter.py`:
     - Add support for more activations.
     - Add shape checks and helpful errors.
     - Optionally expose `use_mx` parameter for `MX`.

5. Unit tests
   - Write tests that compare raw PyTorch forward outputs with CasADi function outputs for random inputs.

6. Build MPC
   - Create objective and bounds consistent with surrogate variable names. Tune `t_step` and `n_horizon`.

7. Simulation & validation
   - Run closed-loop with surrogate-driven MPC on the real plant, measure tracking error and stability.

8. Production considerations
   - For real-time use, profile the solve time and consider compiled controllers or `MX` for faster symbolic computation.

---

## Minimal run instructions (from repository root) — Windows `cmd.exe`

1. Install dependencies (example):
```
python -m pip install -r requirements.txt
python -m pip install casadi do-mpc torch matplotlib
```
2. Run the example (go to example folder and run):
```
cd do_MPC\pytorch_based
python main.py
```

If `model_weights.pth` is missing, either:
- create matching weights by training a compatible PyTorch sequential net and saving `state_dict`, or
- modify `main.py` to randomly initialize weights for a demo run.

---

## Suggested follow-up code changes (prioritized)

1. Improve `template_converter.py`:
   - Add `to_casadi_layer()` helper, support ReLU/Sigmoid/LeakyReLU, add shape assertions, return `ca.Function`.
2. Add unit tests `tests/test_converter.py` to validate numerical equivalence.
3. Make `template_mpc.py` accept flexible state variable names (configurable), so it works with both vectorized `states` and named states like `position`/`velocity`.

---

## Next steps I can implement for you (choose one)

- Expand `template_converter.py` to support additional activations and add shape checks + produce `casadi.Function`.
- Add a pytest test verifying equivalence between PyTorch and CasADi for random inputs.
- Refactor `template_mpc.py` to be state-name-agnostic.

Tell me which to implement and I will update the repo with the code and tests.

---

## Contact / notes

This file was automatically generated from an analysis of the five files in `do_MPC/pytorch_based`. If you want the document in a different location or format, tell me and I will move or extend it.
