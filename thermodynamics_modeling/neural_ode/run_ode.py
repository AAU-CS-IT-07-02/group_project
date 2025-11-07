import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdyn.core import NeuralODE
import yaml

# ---------- Utils ----------

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_device(pref):
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref

def make_interp(times, values):
    """Piecewise-linear interpolation usable inside the ODE."""
    t = times.float()
    v = values.float()
    def _interp(tq):
        i = torch.clamp(torch.searchsorted(t, tq) - 1, 0, len(t)-2)
        t0, t1 = t[i], t[i+1]
        w = (tq - t0) / (t1 - t0 + 1e-12)
        return (1 - w) * v[i] + w[i+1] * w
    return _interp

# ---------- Dataset ----------

class RoomSequenceDataset(Dataset):
    def __init__(self, cfg, split="train", scalers=None):
        self.cfg = cfg
        root = Path(cfg["data_root"]) / cfg["room_folder"]
        files = cfg["files"]

        ts_col = cfg.get("timestamp_column", "timestamp")

        def read_csv(p):
            df = pd.read_csv(p)
            if ts_col not in df.columns:
                raise ValueError(f"Timestamp column '{ts_col}' not found in {p}")
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col).sort_index()
            return df

        df_sens = read_csv(root / files["sensors"])
        df_act  = read_csv(root / files["actuators"])
        df_conf = read_csv(root / files["configuration"])

        # Merge & resample to a uniform grid
        df = df_sens.join(df_act, how="outer").join(df_conf, how="outer")
        freq_s = cfg["sampling_seconds"]
        rule = f"{freq_s}s"
        df = (df
              .resample(rule)
              .mean()
              .interpolate(limit=2)        # short gaps
              .ffill().bfill())            # be robust

        y_cols = cfg["channels"]["y"]
        u_cols = cfg["channels"]["u"]
        w_cols = cfg["channels"]["w"]

        # Allow empty u or w if you really have none
        for col in y_cols + u_cols + w_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in merged dataframe.")

        self.Y = df[y_cols].to_numpy(np.float32)               # (N, Ny)
        self.U = df[u_cols].to_numpy(np.float32) if len(u_cols)>0 else np.zeros((len(df),0), np.float32)
        self.W = df[w_cols].to_numpy(np.float32) if len(w_cols)>0 else np.zeros((len(df),0), np.float32)

        # Standardization (fit on train split only)
        if scalers is None:
            self.scalers = {}
            def fit_scale(a):
                mu, sd = a.mean(0), a.std(0) + 1e-8
                return (mu, sd)
            self.scalers["Y"] = fit_scale(self.Y)
            self.scalers["U"] = fit_scale(self.U) if self.U.shape[1]>0 else (0.0, 1.0)
            self.scalers["W"] = fit_scale(self.W) if self.W.shape[1]>0 else (0.0, 1.0)
        else:
            self.scalers = scalers

        def scale(a, key):
            mu, sd = self.scalers[key]
            return (a - mu)/sd

        self.Ys = scale(self.Y, "Y").astype(np.float32)
        self.Us = scale(self.U, "U").astype(np.float32) if self.U.shape[1]>0 else self.U
        self.Ws = scale(self.W, "W").astype(np.float32) if self.W.shape[1]>0 else self.W

        # Build sliding windows
        T = cfg["horizon_steps"]; S = cfg["stride_steps"]
        self.indices = list(range(0, len(df) - T - 1, S))

        n = len(self.indices)
        ntr = int(n * cfg["train_valid_test_split"][0])
        nva = int(n * cfg["train_valid_test_split"][1])
        if split == "train":
            self.indices = self.indices[:ntr]
        elif split == "valid":
            self.indices = self.indices[ntr:ntr+nva]
        else:
            self.indices = self.indices[ntr+nva:]

        self.dt = float(freq_s)

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        i0 = self.indices[k]
        T  = self.cfg["horizon_steps"]

        # 2-state model: initialize T_mass with T_air at start (same value)
        x0 = np.array([self.Ys[i0, 0], self.Ys[i0, 0]], dtype=np.float32)  # [T_air0, T_mass0]

        # time from 0..T*dt (T+1 points)
        t  = np.linspace(0.0, self.dt*T, T+1, dtype=np.float32)
        u  = self.Us[i0:i0+T]   # (T, Du)
        w  = self.Ws[i0:i0+T]   # (T, Dw)
        y  = self.Ys[i0+1:i0+T+1, [0]]  # future room temp

        out = {
            "x0": torch.tensor(x0),
            "t": torch.tensor(t),
            "u_times": torch.tensor(t[:-1]),
            "u_vals": torch.tensor(u),
            "w_times": torch.tensor(t[:-1]),
            "w_vals": torch.tensor(w),
            "y": torch.tensor(y)
        }
        return out

# ---------- ODE model (RC + NN residual) ----------

class RCResidualODE(nn.Module):
    """
    x = [T_air, T_mass]
    u = [q_hvac_kw, ...]  # standardized
    w = [T_outdoor, GHI, ...]  # standardized; w[0] assumed outdoor temp
    """
    def __init__(self, rc_init, Du, Dw, hidden=32):
        super().__init__()
        self.C_air  = nn.Parameter(torch.tensor(rc_init["C_air"]))
        self.C_mass = nn.Parameter(torch.tensor(rc_init["C_mass"]))
        self.UA_out = nn.Parameter(torch.tensor(rc_init["UA_out"]))
        self.UA_am  = nn.Parameter(torch.tensor(rc_init["UA_am"]))
        self.sp = nn.Softplus()

        self.residual = nn.Sequential(
            nn.Linear(2 + Du + Dw, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t, x, args=None):
        u = args"u_fn" if args and "u_fn" in args else torch.zeros(0, device=x.device)
        w = args"w_fn" if args and "w_fn" in args else torch.zeros(0, device=x.device)

        T_air, T_mass = x[...,0], x[...,1]
        T_out = w[0] if w.numel()>0 else T_air  # if no w, avoid crash

        C_air  = self.sp(self.C_air)  + 1e-3
        C_mass = self.sp(self.C_mass) + 1e-3
        UA_out = self.sp(self.UA_out) + 1e-5
        UA_am  = self.sp(self.UA_am)  + 1e-5

        q_hvac = u[0] if u.numel()>0 else torch.tensor(0.0, device=x.device)

        dT_air  = ( UA_out*(T_out - T_air) + UA_am*(T_mass - T_air) + q_hvac ) / C_air
        dT_mass = ( UA_am*(T_air - T_mass) ) / C_mass
        phys = torch.stack([dT_air, dT_mass], dim=-1)

        res  = self.residual(torch.cat([x, u, w], dim=-1)) if (u.numel()+w.numel())>0 else self.residual(torch.cat([x, torch.zeros(1, device=x.device), torch.zeros(1, device=x.device)], dim=-1))
        return phys + res

class BuildingNODE(nn.Module):
    def __init__(self, ode_func, solver_cfg):
        super().__init__()
        self.node = NeuralODE(
            ode_func,
            solver=solver_cfg["method"],
            sensitivity=solver_cfg["sensitivity"],
            atol=solver_cfg.get("atol", 1e-3),
            rtol=solver_cfg.get("rtol", 1e-3),
            return_t_eval=True
        )

    def forward(self, x0, t_eval, u_times, u_vals, w_times, w_vals):
        u_fn = make_interp(u_times, u_vals) if u_vals.numel()>0 else (lambda t: torch.zeros(1, device=x0.device))
        w_fn = make_interp(w_times, w_vals) if w_vals.numel()>0 else (lambda t: torch.zeros(1, device=x0.device))
        return self.node(x0, t_eval, args={"u_fn": u_fn, "w_fn": w_fn})

# ---------- Training ----------

def main():
    cfg = load_cfg("neural_ode/config.yml")
    torch.manual_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"]["device"])

    # Datasets (fit scalers on train)
    train_ds = RoomSequenceDataset(cfg, split="train")
    valid_ds = RoomSequenceDataset(cfg, split="valid", scalers=train_ds.scalers)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    Du = len(cfg["channels"]["u"])
    Dw = len(cfg["channels"]["w"])

    ode_func = RCResidualODE(cfg["rc_init"], Du, Dw, cfg["residual"]["hidden"]).to(device)
    model = BuildingNODE(ode_func, cfg["solver"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    mse = nn.MSELoss()

    for ep in range(cfg["train"]["epochs"]):
        # ---- train ----
        model.train()
        tr_loss = 0.0; n_tr = 0
        for b in train_loader:
            x0, t, ut, uv, wt, wv, y = [b[k].to(device) for k in ["x0","t","u_times","u_vals","w_times","w_vals","y"]]
            t_eval, x_traj = model(x0, t, ut, uv, wt, wv)  # (T+1, B, 2)
            y_hat = x_traj[1:,...,0:1]                     # predict T_air
            loss = mse(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * y.shape[1]; n_tr += y.shape[1]
        tr_loss /= max(n_tr,1)

        # ---- valid ----
        model.eval()
        va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for b in valid_loader:
                x0, t, ut, uv, wt, wv, y = [b[k].to(device) for k in ["x0","t","u_times","u_vals","w_times","w_vals","y"]]
                _, x_traj = model(x0, t, ut, uv, wt, wv)
                y_hat = x_traj[1:,...,0:1]
                l = mse(y_hat, y)
                va_loss += l.item() * y.shape[1]; n_va += y.shape[1]
        va_loss /= max(n_va,1)

        print(f"[{ep+1:03d}] train MSE={tr_loss:.5f}  valid MSE={va_loss:.5f}")

    # Save weights
    torch.save(model.state_dict(), cfg["train"]["save_path"])
    print(f"Saved to {cfg['train']['save_path']}")

if __name__ == "__main__":
    main()
