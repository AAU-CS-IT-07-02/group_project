#!/usr/bin/env python3
"""
PINN for Building Thermal Dynamics (Room-level) using DeepXDE

Features:

- End-to-end: loads your CSVs (actuators/sensors/config), merges & preprocesses
- Fuzzy column matching (handles underscores/diacritics)
- DeepXDE PINN with ODE-style thermal balance residual
- Works with DeepXDE backends: torch (default), tf, jax
- Backend-aware data conversions to avoid NumPy ↔ GPU tensor errors
- CLI flags for backend, device, epochs, outdir, room

Usage example:
  python pinn_deepxde_roomA.py \
    
    --act ../data_fragmentation/.../data_actuators.csv \
    --sns ../data_fragmentation/.../data_sensors.csv \
    --cfg ../data_fragmentation/.../data_configuration.csv \
    --room RoomA \
    --epochs 15000 \
    --backend torch \
    --device cuda \
    --outdir ./outputs

"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# -----------------------------
# 1) CLI & Backend selection
# -----------------------------

def parse_args():
    """
    Parse command-line arguments for the PINN training script.

    Returns:
        argparse.Namespace: Parsed arguments including paths, backend, device, epochs, and output directory.

    CLI Arguments:

        --act (str): Path to actuators CSV file.
        --sns (str): Path to sensors CSV file.
        --cfg (str): Path to configuration CSV file.
        --room (str): Room identifier (e.g., 'RoomA', 'RoomC').
        --epochs (int): Number of Adam iterations before L-BFGS.
        --backend (str): DeepXDE backend ('torch', 'tf', 'jax').
        --device (str): Device preference ('auto', 'cpu', 'cuda').
        --outdir (str): Output directory for results.
    """
    p = argparse.ArgumentParser(description='PINN for building thermal dynamics (DeepXDE)')
    p.add_argument('--act', type=str, required=True, help='Path to data_actuators.csv')
    p.add_argument('--sns', type=str, required=True, help='Path to data_sensors.csv')
    p.add_argument('--cfg', type=str, required=True, help='Path to data_configuration.csv')
    p.add_argument('--room', type=str, default='RoomA', choices=['RoomA','RoomC'], help='Which room to model')
    p.add_argument('--epochs', type=int, default=15000, help='Adam iterations before L-BFGS')
    p.add_argument('--backend', type=str, default='torch', choices=['torch','tf','jax'], help='DeepXDE backend')
    p.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'], help='Device preference (torch backend)')
    p.add_argument('--outdir', type=str, default='./outputs', help='Output directory')
    return p.parse_args()

args = parse_args()

# Set DeepXDE backend BEFORE importing deepxde
backend_map = {'torch':'pytorch','tf':'tensorflow','jax':'jax'}
os.environ['DDE_BACKEND'] = backend_map[args.backend]

# If torch backend + CPU forced, disable GPUs before import
if args.backend == 'torch' and args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # make torch/deepxde run on CPU

import deepxde as dde

# -----------------------------
# 2) Load & preprocess data
# -----------------------------

def fuzzy_find(df_cols, tokens):
    """
    Find a column name in a DataFrame that matches all tokens (ordered or unordered).

    Args:
        df_cols (list[str]): List of column names.
        tokens (list[str]): Tokens to match (case-insensitive).

    Returns:
        str or None: Matching column name or None if not found.

    Example:
        >>> fuzzy_find(['RoomA:temperature', 'RoomA:humidity'], ['rooma', 'temperature'])
        
        'RoomA:temperature'
    """
    toks = [t.lower() for t in tokens]
    # ordered match
    for c in df_cols:
        cl = c.lower()
        pos = 0
        ok = True
        for t in toks:
            i = cl.find(t, pos)
            if i == -1:
                ok = False; break
            pos = i + len(t)
        if ok:
            return c
    # any-order match
    for c in df_cols:
        cl = c.lower()
        if all(t in cl for t in toks):
            return c
    return None


def load_and_prepare(act_path, sns_path, cfg_path, room='RoomA'):
    """
    Load actuator, sensor, and configuration data; merge, clean, and prepare arrays for PINN.

    Args:
        act_path (str): Path to actuators CSV.
        sns_path (str): Path to sensors CSV.
        cfg_path (str): Path to configuration CSV.
        room (str): Room identifier (default 'RoomA').

    Returns:
        dict: Contains time array `t`, target `y`, exogenous features `X`, normalization stats,
              feature names, and metadata.

    Raises:
        ValueError: If required columns cannot be found.

    Example:
        >>> data_dict = load_and_prepare('act.csv', 'sns.csv', 'cfg.csv', room='RoomA')
        >>> print(data_dict.keys())
    """

    parse_dates = ['timestamp']
    act = pd.read_csv(act_path, parse_dates=parse_dates)
    sns = pd.read_csv(sns_path, parse_dates=parse_dates)
    cfg = pd.read_csv(cfg_path, parse_dates=parse_dates)

    df = act.merge(sns, on='timestamp', how='outer').merge(cfg, on='timestamp', how='outer')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Target & exogenous columns
    room_prefix = f"{room}:"

    col_T_room = fuzzy_find(df.columns, [room_prefix, 'sensor', 'room', 'temperature'])
    col_T_out  = fuzzy_find(df.columns, ['outdoor:', 'temperature', 'air'])
    col_occ    = fuzzy_find(df.columns, [room_prefix, 'people', 'amount'])
    col_win    = fuzzy_find(df.columns, [room_prefix, 'window', 'opened'])
    col_dmp    = fuzzy_find(df.columns, [room_prefix, 'damper', 'position'])
    col_rad    = fuzzy_find(df.columns, [room_prefix, 'radiator', 'control', 'signal', 'motor', 'valve'])
    col_sE     = fuzzy_find(df.columns, ['outdoor:', 'solar', 'direct', 'east'])
    col_sS     = fuzzy_find(df.columns, ['outdoor:', 'solar', 'direct', 'south'])
    col_sW     = fuzzy_find(df.columns, ['outdoor:', 'solar', 'direct', 'west'])
    col_Tsup   = fuzzy_find(df.columns, ['ventilation:', 'sensor', 'air', 'temperature', 'supply'])
    col_msup   = fuzzy_find(df.columns, ['ventilation:', 'fan', 'air', 'flow', 'supply'])

    mapping = {
        'T_room': col_T_room,
        'T_out' : col_T_out,
        'occ'   : col_occ,
        'win'   : col_win,
        'dmp'   : col_dmp,
        'rad'   : col_rad,
        'sE'    : col_sE,
        'sS'    : col_sS,
        'sW'    : col_sW,
        'Tsup'  : col_Tsup,
        'msup'  : col_msup,
    }

    missing = [k for k,v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Could not find required columns for: {missing}\nAvailable columns example: {df.columns[:10].tolist()} ...")

    keep = ['timestamp'] + [v for v in mapping.values()]
    mdl = df[keep].copy()
    for c in keep:
        if c != 'timestamp':
            mdl[c] = pd.to_numeric(mdl[c], errors='coerce')

    mdl = mdl.set_index('timestamp').sort_index().ffill(limit=3)
    mdl = mdl.dropna(subset=[mapping['T_room']])

    # infer median step
    if len(mdl.index) > 1:
        step_sec = int(mdl.index.to_series().diff().dropna().dt.total_seconds().median())
    else:
        step_sec = 300

    mdl = mdl.resample(f'{step_sec}s').nearest(limit=1).dropna()

    # Build arrays
    t0 = mdl.index[0]
    t = (mdl.index - t0).total_seconds().astype(np.float64).values.reshape(-1,1)

    y = mdl[mapping['T_room']].values.reshape(-1,1)
    X_cols = [mapping[k] for k in ['T_out','sE','sS','sW','occ','win','dmp','rad','Tsup','msup']]
    X = mdl[X_cols].values

    X_mu = X.mean(axis=0, keepdims=True)
    X_sd = X.std(axis=0, keepdims=True) + 1e-8
    Y_mu = float(y.mean()); Y_sd = float(y.std() + 1e-8)

    return {
        't': t, 'y': y, 'X': X, 'X_mu': X_mu, 'X_sd': X_sd, 'Y_mu': Y_mu, 'Y_sd': Y_sd,
        'X_cols': X_cols, 'step_sec': step_sec, 't0_iso': str(t0), 'mapping': mapping
    }


# -----------------------------
# 3) PINN model (DeepXDE)
# -----------------------------

def to_backend_array(arr, like_tensor):
    """Cast NumPy array arr to same backend array as like_tensor (torch or np)."""
    # Torch backend tensor?
    if hasattr(like_tensor, 'detach'):
        import torch
        return torch.as_tensor(arr, dtype=like_tensor.dtype, device=like_tensor.device)
    # Fallback NumPy
    return np.asarray(arr)


def make_exo_at(t_np, Xn_np):
    """
    Return a function exo_at(t_query) that aligns exogenous Xn to t_query.
    Works for torch/np inputs; returns same backend as input.
    """
    def _exo_at(t_query):
        # Convert t_query to NumPy (CPU) for searchsorted
        if hasattr(t_query, 'detach'):  # torch tensor
            tq_np = t_query.detach().cpu().numpy().reshape(-1,1)
            is_torch = True
            t_dev = t_query.device
            t_dtype = t_query.dtype
        else:
            tq_np = np.asarray(t_query).reshape(-1,1)
            is_torch = False
            t_dev = None
            t_dtype = None

        idx = np.clip(np.searchsorted(t_np.flatten(), tq_np.flatten()) - 1, 0, len(t_np) - 1)
        ex_np = Xn_np[idx, :]

        if is_torch:
            import torch
            return torch.from_numpy(ex_np).to(t_dev).to(t_dtype)
        else:
            return ex_np
    return _exo_at


def train_pinn(data_dict, outdir, epochs=15000):
    """
    Train a Physics-Informed Neural Network (PINN) for room thermal dynamics using DeepXDE.

    Args:
        data_dict (dict): Prepared dataset dictionary from `load_and_prepare()`.
        outdir (str or Path): Directory to save outputs.
        epochs (int): Number of Adam iterations before L-BFGS (default 15000).

    Outputs:
        
        - room_pinn_predictions.npz: Predicted vs true temperatures.
        - learned_coefficients.json: Learned physical parameters.
        - dataset_pack.npz: Normalized dataset for reproducibility.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Unpack
    t = data_dict['t']; y = data_dict['y']; X = data_dict['X']
    X_mu = data_dict['X_mu']; X_sd = data_dict['X_sd']
    Y_mu = data_dict['Y_mu']; Y_sd = data_dict['Y_sd']
    X_cols = data_dict['X_cols']

    # Normalize
    Xn = (X - X_mu) / X_sd
    yn = (y - Y_mu) / Y_sd

    # Build exo_at aligned to t
    exo_at = make_exo_at(t_np=t, Xn_np=Xn)

    # Trainable coefficients
    C      = dde.Variable(1.0)
    U      = dde.Variable(0.5)
    alpha  = dde.Variable(0.5)
    beta   = dde.Variable(0.5)
    gE     = dde.Variable(0.05)
    gS     = dde.Variable(0.05)
    gW     = dde.Variable(0.05)
    delta  = dde.Variable(0.05)
    epsW   = dde.Variable(0.05)

    geom = dde.geometry.TimeDomain(float(t[0]), float(t[-1]))
    net = dde.maps.FNN([1] + [64]*3 + [1], 'tanh', 'Glorot normal')

    # Residual
    def pde_residual(x, ypred):
        dy_dt = dde.grad.jacobian(ypred, x, i=0, j=0)
        T_hat = Y_mu + Y_sd * ypred
        ex = exo_at(x)  # backend-aware

        # Cast normalization tensors to backend
        X_mu_b = to_backend_array(X_mu, ex)
        X_sd_b = to_backend_array(X_sd, ex)

        i = {k: idx for idx, k in enumerate(X_cols)}

        Tout = ex[:, i[X_cols[0]]:i[X_cols[0]]+1] * X_sd_b[:, i[X_cols[0]]:i[X_cols[0]]+1] + X_mu_b[:, i[X_cols[0]]:i[X_cols[0]]+1]
        sE   = ex[:, i[X_cols[1]]:i[X_cols[1]]+1]
        sS   = ex[:, i[X_cols[2]]:i[X_cols[2]]+1]
        sW   = ex[:, i[X_cols[3]]:i[X_cols[3]]+1]
        occ  = ex[:, i[X_cols[4]]:i[X_cols[4]]+1]
        win  = ex[:, i[X_cols[5]]:i[X_cols[5]]+1]
        dmp  = ex[:, i[X_cols[6]]:i[X_cols[6]]+1]
        rad  = ex[:, i[X_cols[7]]:i[X_cols[7]]+1]
        Tsup = ex[:, i[X_cols[8]]:i[X_cols[8]]+1] * X_sd_b[:, i[X_cols[8]]:i[X_cols[8]]+1] + X_mu_b[:, i[X_cols[8]]:i[X_cols[8]]+1]
        msup = ex[:, i[X_cols[9]]:i[X_cols[9]]+1]

        rhs = (
            U * (Tout - T_hat)
            + alpha * msup * (Tsup - T_hat) * dmp
            + beta * rad
            + gE * sE + gS * sS + gW * sW
            + delta * occ
            + epsW * win * (Tout - T_hat)
        )
        return C * Y_sd * dy_dt - rhs

    # Initial condition & observation BC
    ic_val = float(yn[0])
    def ic_func(x):
        import numpy as _np
        return _np.ones((x.shape[0],1)) * ic_val

    ic = dde.icbc.IC(geom, ic_func, lambda x, on_initial: on_initial)
    obs = dde.icbc.PointSetBC(t, yn, component=0)

    data = dde.data.PDE(
        geom,
        pde_residual,
        [ic, obs],               # pass BOTH IC and observations here (no add_bc)
        num_domain=2000,
        num_boundary=0,
        anchors=t,
    )

    model = dde.Model(data, net)
    model.compile('adam', lr=1e-3, loss_weights=[1.0, 1.0, 5.0])
    model.train(iterations=epochs)

    # L-BFGS polish
    try:
        model.compile('L-BFGS')
        model.train()
    except Exception as e:
        print('[WARN] L-BFGS stage skipped due to:', e)

    yn_hat = model.predict(t)
    T_hat = Y_mu + Y_sd * yn_hat

    # Save outputs
    np.savez(outdir / 'room_pinn_predictions.npz', t=t, T_hat=T_hat, T_true=y)
    with open(outdir / 'learned_coefficients.json', 'w') as f:
        json.dump({
            'C': float(C), 'U': float(U), 'alpha': float(alpha), 'beta': float(beta),
            'gamma': [float(gE), float(gS), float(gW)],
            'delta': float(delta), 'epsW': float(epsW)
        }, f, indent=2)

    # Also save dataset pack for reproducibility
    np.savez(outdir / 'dataset_pack.npz', t=t, y=y, X=X, X_mu=X_mu, X_sd=X_sd, Y_mu=Y_mu, Y_sd=Y_sd, X_cols=np.array(X_cols, dtype=object))

    print('Feature order:', X_cols)
    print('Saved:', outdir / 'room_pinn_predictions.npz')
    print('Saved:', outdir / 'learned_coefficients.json')


# -----------------------------
# 4) Main
# -----------------------------

def main():
    print(f"[INFO] DeepXDE backend = {dde.backend.backend_name}")
    print(f"[INFO] Args: {vars(args)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_dict = load_and_prepare(args.act, args.sns, args.cfg, room=args.room)

    # Dump mapping and metadata
    meta = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'room': args.room,
        'step_sec': data_dict['step_sec'],
        't0_iso': data_dict['t0_iso'],
        'mapping': data_dict['mapping'],
        'X_cols': data_dict['X_cols'],
    }
    (outdir / 'PREP_METADATA.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    train_pinn(data_dict, outdir, epochs=args.epochs)


if __name__ == '__main__':
    main()
