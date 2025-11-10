#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt


def load_dims(train_file):
    arr = np.loadtxt(train_file)
    ncols = arr.shape[1]
    nfeat = ncols - 1
    return nfeat


def main():
    ap = argparse.ArgumentParser(description='Train a simple FNN on easy.train/easy.test using DeepXDE DataSet')
    ap.add_argument('--train', default='easy_out/easy.train')
    ap.add_argument('--test',  default='easy_out/easy.test')
    ap.add_argument('--iters', type=int, default=15000)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--outdir', default='easy_out')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Load raw arrays ----------
    train_arr = np.loadtxt(args.train)
    test_arr = np.loadtxt(args.test)

    X_train_raw = train_arr[:, :-1]
    y_train_raw = train_arr[:, -1:]
    X_test_raw  = test_arr[:, :-1]
    y_test_raw  = test_arr[:, -1:]

    # ---------- Manual standardization ----------
    X_mu = X_train_raw.mean(axis=0, keepdims=True)
    X_sd = X_train_raw.std(axis=0, keepdims=True)
    y_mu = y_train_raw.mean()
    y_sd = y_train_raw.std()

    # Avoid division by zero
    X_sd[X_sd == 0] = 1.0
    if y_sd == 0:
        y_sd = 1.0

    X_train = (X_train_raw - X_mu) / X_sd
    y_train = (y_train_raw - y_mu) / y_sd
    X_test  = (X_test_raw  - X_mu) / X_sd
    y_test  = (y_test_raw  - y_mu) / y_sd

    # ---------- DeepXDE dataset (no internal standardization) ----------
    data = dde.data.DataSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        standardize=False,   # WE DO IT MANUALLY
    )

    # ---------- Build network ----------
    nfeat = X_train.shape[1]
    layer = [nfeat] + [args.width]*args.depth + [1]

    net = dde.nn.FNN(layer, 'tanh', 'Glorot normal')
    model = dde.Model(data, net)
    model.compile('adam', lr=args.lr, metrics=['l2 relative error'])
    losshistory, train_state = model.train(iterations=args.iters)

    # ---------- Predict (still standardized) ----------
    y_pred_std = model.predict(X_test)

    # ---------- Unstandardize ----------
    y_pred = y_pred_std * y_sd + y_mu
    y_true = y_test * y_sd + y_mu

    # ---------- Plot ----------
    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/'easy_pred.png', dpi=150)
    print("Saved:", outdir/'easy_pred.png')

if __name__ == '__main__':
    main()
