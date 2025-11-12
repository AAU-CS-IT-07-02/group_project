#!/usr/bin/env python3
"""
deepxde_make_dataset.py

Create simple docs-style DataSet files (easy.train / easy.test) from your building CSVs
so you can train a plain supervised FNN with DeepXDE (no physics, no BCs).

Key features
------------
- Explicit **--target** flag to choose the exact target column (e.g., your room temperature).
- Feature matrix = [ t_seconds, Fourier_k*(sin,cos) of time, exogenous columns, (optional) y_lag ]
- Train/test split = last N days (configurable with --test_days)
- Writes EASY_META.json with the exact feature order and chosen target

Usage
-----
python deepxde_make_dataset.py \
  --act path/to/data_actuators.csv \
  --sns path/to/data_sensors.csv \
  --cfg path/to/data_configuration.csv \
  --room RoomA \
  --target "RoomA:sensor__room_temperature" \
  --fourier_k 6 --test_days 2 --outdir out
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- utils ----------

def fuzzy_find(df_cols, tokens):
    """Return first column whose lowercase name contains all tokens in order; else any-order."""
    toks = [t.lower() for t in tokens]
    # ordered match
    for c in df_cols:
        cl = c.lower(); pos = 0; ok = True
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


def build_fourier_time(t_sec, kmax):
    """Return (N, 2*kmax) array with [sin(w_k t), cos(w_k t)] for daily harmonics.
    If kmax==0 -> (N, 0).
    """
    if kmax <= 0:
        return np.zeros((t_sec.shape[0], 0), dtype=float)
    feats = []
    base = 2.0 * np.pi / (24.0 * 3600.0)
    for k in range(1, kmax + 1):
        w = k * base
        s = np.sin(w * t_sec)
        c = np.cos(w * t_sec)
        # ensure (N,1) each
        if s.ndim == 1:
            s = s.reshape(-1, 1)
            c = c.reshape(-1, 1)
        feats.append(s)
        feats.append(c)
    return np.concatenate(feats, axis=1) if feats else np.zeros((t_sec.shape[0], 0), dtype=float)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description='Make easy train/test files (DataSet) from building CSVs')
    ap.add_argument('--act', required=True)
    ap.add_argument('--sns', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--room', default='RoomA', choices=['RoomA','RoomC'])
    ap.add_argument('--target', type=str, default=None, help='Exact column name to use as target (room temperature).')
    ap.add_argument('--outdir', default='./out')
    ap.add_argument('--fourier_k', type=int, default=6)
    ap.add_argument('--add_lag', action='store_true', help='Append one-step lag of target as an extra feature')
    ap.add_argument('--test_days', type=int, default=2, help='Hold out last N days for test set')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    parse_dates=['timestamp']
    act = pd.read_csv(args.act, parse_dates=parse_dates)
    sns = pd.read_csv(args.sns, parse_dates=parse_dates)
    cfg = pd.read_csv(args.cfg, parse_dates=parse_dates)

    # Merge & sort
    df = act.merge(sns, on='timestamp', how='outer').merge(cfg, on='timestamp', how='outer')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Fuzzy discovery for exogenous columns
    room_prefix = f"{args.room}:"
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

    # Target selection
    if args.target is not None:
        if args.target not in df.columns:
            raise SystemExit(f"--target '{args.target}' not found. Example columns: {list(df.columns)[:12]}")
        target_col = args.target
    else:
        target_col = col_T_room
        if target_col is None:
            raise SystemExit("Could not find room temperature column by fuzzy matching. Please pass --target <exact column name>.")

    # Ensure we have required exogenous columns; missing are OK but warn
    missing_exo = {k:v for k,v in mapping.items() if k!='T_room' and v is None}
    if missing_exo:
        print('[WARN] Some exogenous columns not found:', missing_exo)

    # Keep only discovered columns
    keep = ['timestamp'] + [c for c in mapping.values() if c is not None] + [target_col]
    # Remove duplicates if target overlaps with discovered list
    keep = list(dict.fromkeys(keep))
    mdl = df[keep].copy()

    # Numeric conversion
    for c in keep:
        if c != 'timestamp':
            mdl[c] = pd.to_numeric(mdl[c], errors='coerce')

    # Time index and basic cleaning
    mdl = mdl.set_index('timestamp').sort_index().ffill(limit=3)
    mdl = mdl.dropna(subset=[target_col])

    # Regularize sampling to dominant step
    if len(mdl.index) > 1:
        step_sec = int(mdl.index.to_series().diff().dropna().dt.total_seconds().median())
    else:
        step_sec = 300
    mdl = mdl.resample(f'{step_sec}s').nearest(limit=1).dropna()

    # Build arrays
    t0 = mdl.index[0]
    tsec = (mdl.index - t0).total_seconds().astype(float).values.reshape(-1,1)
    ft = build_fourier_time(tsec, args.fourier_k)

    X_cols = [c for c in [mapping[k] for k in ['T_out','sE','sS','sW','occ','win','dmp','rad','Tsup','msup']] if c is not None]
    X = mdl[X_cols].values if X_cols else np.zeros((tsec.shape[0], 0), dtype=float)

    y = mdl[target_col].values.reshape(-1,1)

    # Optional lag feature of the target
    if args.add_lag:
        y_lag = np.roll(y, 1, axis=0)
        if y_lag.shape[0] > 1:
            y_lag[0] = y_lag[1]  # simple fill for first row
        features = np.hstack([tsec, ft, X, y_lag])
        feat_names = ['t_seconds'] + [f'sin{k}' for k in range(1,args.fourier_k+1)] + [f'cos{k}' for k in range(1,args.fourier_k+1)] + X_cols + ['y_lag1']
    else:
        features = np.hstack([tsec, ft, X])
        feat_names = ['t_seconds'] + [f'sin{k}' for k in range(1,args.fourier_k+1)] + [f'cos{k}' for k in range(1,args.fourier_k+1)] + X_cols

    # Train/test split: last N days
    test_span_sec = args.test_days * 24 * 3600
    # If dataset shorter than requested test span, use 80/20 split
    if tsec[-1] - tsec[0] < test_span_sec and features.shape[0] >= 5:
        split_idx = int(features.shape[0] * 0.8)
        print(f"[WARN] Dataset shorter than {args.test_days} days. Using 80/20 split at index {split_idx}.")
    else:
        split_idx = int(np.searchsorted(tsec.ravel(), tsec[-1] - test_span_sec))
        split_idx = max(1, min(features.shape[0]-1, split_idx))

    X_tr, y_tr = features[:split_idx], y[:split_idx]
    X_te, y_te = features[split_idx:], y[split_idx:]

    # Save .train / .test (concat X | y in one file, as docs do)
    train_path = outdir/"easy.train"
    test_path  = outdir/"easy.test"
    np.savetxt(train_path, np.hstack([X_tr, y_tr]))
    np.savetxt(test_path,  np.hstack([X_te, y_te]))

    meta = {
        'generated_from': {'act': args.act, 'sns': args.sns, 'cfg': args.cfg},
        'room': args.room,
        'target': args.target,
        'Y_col': target_col,
        'fourier_k': args.fourier_k,
        'add_lag': bool(args.add_lag),
        'step_sec': step_sec,
        't0_iso': str(t0),
        'X_cols': feat_names,
        'train_rows': int(X_tr.shape[0]),
        'test_rows': int(X_te.shape[0]),
        'feature_dim': int(features.shape[1])
    }
    (outdir/"EASY_META.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')

    # Helpful prints
    print('\n=== EASY DATASET SUMMARY ===')
    print('Target column (Y):', target_col)
    print('Feature dim:', features.shape[1])
    print('Feature order:', feat_names)
    print('Shapes -> X_tr', X_tr.shape, 'y_tr', y_tr.shape, '| X_te', X_te.shape, 'y_te', y_te.shape)
    print('\nWrote:', train_path)
    print('Wrote:', test_path)
    print('Meta :', outdir/'EASY_META.json')

if __name__ == '__main__':
    main()
