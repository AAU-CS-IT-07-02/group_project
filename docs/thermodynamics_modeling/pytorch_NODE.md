## PyTorch Neural ODE (NODE) Implementation

This page documents a minimal, self-contained Neural Ordinary Differential Equation (Neural ODE) implementation using PyTorch and `torchdiffeq`. It explains the background theory, the major components implemented in `thermodynamics_modeling/pythorch_node/torchdiffeq_model.py`, data handling, configuration expectations, usage examples, and tips for training and troubleshooting.

Overview:

- Repository path: `thermodynamics_modeling/pythorch_node/torchdiffeq_model.py`
- Purpose: provide a pipeline to train a Neural ODE model to predict multi-step room temperatures from control/disturbance sequences.

Background (brief)

- Neural ODEs: Neural ODEs replace discrete dynamics (e.g., RNN or ResNet) with a continuous-time ODE parameterized by a neural network. The latent state z(t) evolves according to

$$\frac{dz}{dt} = f(z(t), u(t); \theta)$$

where $u(t)$ is a time-varying control input (or exogenous signal). The ODE is integrated with an ODE solver (explicit or implicit) to produce latent trajectories, which are decoded to observations.

- Why useful for building dynamics: building thermal dynamics evolve continuously and may be better modeled with continuous-time dynamics, particularly when sensor/control times are irregular or when one wants solver-based integration methods.

## Mathematical pieces used

- Latent dynamics: $\dot z = f(z, u(t))$ implemented by `ODEFunc` (a small MLP).
- Encoder: maps initial observed state and initial control to latent initial state $z_0$.
- Decoder: maps latent state $z(t)$ to predicted outputs (room temperatures).
- Loss: mean-squared error across predicted sequence vs. ground-truth sequence.

## High-level pipeline

- Load CSV data and convert to numeric tensors.
- Build sliding windows of length `H` (sequence horizon) using `WindowedDataset`.
- Normalize using statistics computed from training data.
- For each batch: encode initial observation to latent `z0`, set up control interpolator over the horizon, integrate latent ODE across time vector `t_span`, decode to predictions, compute MSE loss, and step optimizer.

Files & important symbols

`torchdiffeq_model.py` — main implementation. Key classes/functions:

## Robust CSV reader with forward-fill and basic coercion.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.read_csv_as_dicts

## Extract a `[T, D]` tensor for selected features from parsed CSV rows.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.build_matrix

## Normalization helpers: compute per-feature mean/std, and (de)normalize tensors.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.compute_norm_stats
::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.normalize
::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.denormalize

## Builds sliding windows of length `H` and yields `(controls_seq, y_seq, y0)`.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.WindowedDataset

## Piecewise-linear interpolation of control values for arbitrary times in `[0,H-1]` (supports batched and single-sample modes).

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.ControlInterpolator

## Defines the neural RHS `f(z,u)`; set the control interpolator via `set_control` before integration.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.ODEFunc

## Small MLPs used for mapping to/from latent space (encode initial state to `z0`, decode `z(t)` to outputs).

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.Encoder
::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.Decoder

## Glue model: encodes `y0` + `u0` -> `z0`, sets `odefunc` control, integrates via `torchdiffeq.odeint`, and decodes predictions.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.NeuralODEModel

## Data loader creation and training/validation/test orchestration functions.

::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.make_dataloader
::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.train_loop
::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.evaluate
::: thermodynamics_modeling.pythorch_node.torchdiffeq_model.main

## Configuration (expected fields in `config.yml`)
The code loads a `config.yml` from the same directory as `torchdiffeq_model.py`. The keys used in the script include (example structure):

```yaml
train_data: path/to/train.csv
test_data: path/to/test.csv
observations: [list, of, control, columns]
disturbances: [list, of, disturbance, columns]
outdoor: [list, of, outdoor, columns]
rooms_temp: [room1_temp, room2_temp]
dataset:
	H: 24          # horizon (window length)
	batch_size: 32
model:
	latent_space_dimensions: 64
	epochs: 200
	patience: 10
	warmup: 5
outdir: outputs
device: cpu
metrics:
	start_idx: None
	end_idx: None
num_threads: 4
```

Notes:

- `observations`, `disturbances`, and `outdoor` are concatenated to form control inputs. `rooms_temp` is treated as the target output features.
- `H` determines both the dataset window length and the length of the `t_span` vector used for ODE integration.

## Data format
- CSV rows are read as dictionaries; column names must match those listed in the config.
- Non-numeric tokens are coerced: booleans map to 1/0, others are hashed to a small numeric code. Missing values are forward-filled, then zero-filled.

## Key implementation details

- Control interpolation: `ControlInterpolator` supports both single sample (`[H, d_u]`) and batched (`[B, H, d_u]`) control arrays. It returns `u(t)` for any scalar or batched `t` by linear interpolation between reference time points `t_ref = [0, 1, ..., H-1]`.

- ODE function: `ODEFunc.forward(self, t, z)` expects `z` with shape `[B, latent_dim]` and uses the attached control interpolator to obtain `u(t)` for that `t`. It concatenates `[z, u]` and passes through a small MLP to compute $\dot z$.

- Integration: `torchdiffeq.odeint` is used with `t_span = torch.linspace(0, H-1, H)`. The solver returns a tensor shaped `[H, B, latent_dim]`, which is decoded per time-step.

## Loss and metrics

- The training loop uses `nn.MSELoss()` between predicted sequence and ground-truth sequence (both normalized).
- Evaluation reports normalized MAE and normalized RMSE. Denormalization helpers are provided so you can convert metrics back to physical units if needed.

## Usage (run locally)
1. Make sure dependencies are installed:

```
torch
torchdiffeq
pyyaml
numpy
```

2. Prepare `config.yml` next to `torchdiffeq_model.py` as shown above and create your `train.csv` and `test.csv`.

3. Run the script (from the repo root):

```bash
python3 thermodynamics_modeling/pythorch_node/torchdiffeq_model.py
```

Outputs written to `outdir` (as `config.yml` specifies):
- `scalers.pt` — saved normalization stats (`c_mean`, `c_std`, `y_mean`, `y_std`).
- `best_model.pt` — checkpoint saved at best validation performance.
- `metrics.pt` — saved evaluation metrics for test set.

## Example: interpreting the model forward pass

- Given a batch of controls `controls_seq` shaped `[B, H, d_u]`, a sequence of targets `y_seq` shaped `[B, H, d_y]`, and the initial observed `y0 = y_seq[:, 0, :]`:
	1. `u0 = controls_seq[:, 0, :]` and `enc_in = concat(y0, u0)`
	2. `z0 = encoder(enc_in)`
	3. `interp = ControlInterpolator(t_span, controls_seq)` and `odefunc.set_control(interp)`
	4. `z_t = odeint(odefunc, z0, t_span)` produces latent trajectory
	5. `y_hat = decoder(z_t)` produces predictions for each time-step.

## Practical tips & tuning

- Latent dimension (`latent_space_dimensions`) controls representational capacity — too small underfits, too large may overfit.
- Batch size and `H` influence memory and computation (long horizons increase ODE solve cost).
- The chosen solver (`rk4`, `dopri5`, etc.) affects speed and stability. `rk4` is explicit and fixed-step; `dopri5` is adaptive and may be slower but more accurate for stiff dynamics.
- Gradient clipping (`clip_grad_norm_`) is included to improve training stability.
- Normalize inputs & outputs with training statistics (the pipeline already does this).

## Debugging checklist

- If training is failing or loss is NaN:
	- Check data preprocessing: missing columns, non-numeric tokens, or unexpected shapes.
	- Verify `H` matches the intended sequence length for both dataset and `t_span`.
	- Try smaller learning rate, smaller batch size, or enable gradient clipping tighter.
- If model predictions are constant:
	- Check encoder outputs for variance; ensure input features are not constant after normalization.
	- Try increasing latent dimension or adding nonlinearity/skip connections.

## References & further reading

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. In NeurIPS.
- `torchdiffeq` library: https://github.com/rtqichen/torchdiffeq

