# neural_ode_pipeline.py
# Minimal Neural ODE training pipeline using only torch, torch.nn, and torchdiffeq.odeint
# Everything else uses Python's standard library.

import os, csv, math, time, random, yaml
import torch
import torch.nn as nn
from torchdiffeq import odeint

# -----------------------------
# 1) Configuration (from your config.yml, embedded here as a Python dict)
# -----------------------------
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Derived feature groups
CONTROL_FEATURES = config["observations"] + config["disturbances"] + config["outdoor"]
TARGET_FEATURES = config["rooms_temp"]
H = int(config["dataset"]["H"])
BATCH_SIZE = int(config["dataset"]["batch_size"])
LATENT_DIM = int(config["model"]["latent_space_dimensions"])
EPOCHS = int(config["model"]["epochs"])
PATIENCE = int(config["model"]["patience"])
WARMUP = int(config["model"]["warmup"])
OUTDIR = config["outdir"]
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# 2) Utilities: CSV loading & normalization
# -----------------------------

def read_csv_as_dicts(path):
    """Read CSV rows into a list of dicts: {header: float_value}.
    Missing/empty values are forward-filled per column, then 0 if still missing.
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        # strip BOM or extra spaces
        headers = [h.strip().replace("\ufeff", "") for h in headers]
        rows = []
        prev_vals = [None] * len(headers)
        for row in reader:
            row_vals = {}
            for i, val in enumerate(row):
                v = val.strip()
                if v == "" or v.lower() == "nan":
                    # forward-fill or zero
                    if prev_vals[i] is not None:
                        v = prev_vals[i]
                    else:
                        v = "0.0"
                try:
                    fv = float(v)
                except ValueError:
                    # If nonnumeric, attempt to coerce (e.g., "True"/"False")
                    if v.lower() in ("true", "yes", "on"):
                        fv = 1.0
                    elif v.lower() in ("false", "no", "off"):
                        fv = 0.0
                    else:
                        # last resort: hash to a small numeric code
                        fv = float(abs(hash(v)) % 10)
                prev_vals[i] = str(fv)
                row_vals[headers[i]] = fv
            rows.append(row_vals)
    return headers, rows

def build_matrix(rows, selected_cols):
    """Extract matrix [T, D] for selected columns from list of row dicts."""
    T = len(rows)
    D = len(selected_cols)
    X = torch.empty(T, D, dtype=torch.float32)
    missing = []
    for j, col in enumerate(selected_cols):
        for t in range(T):
            if col in rows[t]:
                X[t, j] = float(rows[t][col])
            else:
                missing.append(col)
                X[t, j] = 0.0
    if missing:
        missing = sorted(set(missing))
        print(f"[WARN] Missing columns not found in CSV: {missing}")
    return X

def compute_norm_stats(X):
    """Compute mean/std per feature, with epsilon for numerical stability."""
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    eps = 1e-8
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean, std

def normalize(X, mean, std):
    return (X - mean) / std

def denormalize(Xn, mean, std):
    return Xn * std + mean

# -----------------------------
# 3) Dataset: sliding windows of length H
# -----------------------------
class WindowedDataset(torch.utils.data.Dataset):
    """
    Builds windows of length H:
      - controls_seq: [H, d_u] normalized
      - y_seq:        [H, d_y] normalized
      - y0:           [d_y]    first step of y_seq
    Loss is computed across y_seq vs predictions across all H steps.
    """
    def __init__(self, controls, targets, start_idx=None, end_idx=None):
        assert controls.shape[0] == targets.shape[0], "controls and targets must have same time length"
        self.controls = controls
        self.targets = targets
        self.T = controls.shape[0]
        # Restrict to indices if provided (metrics config)
        s = 0 if start_idx is None else (self.T + start_idx if start_idx < 0 else start_idx)
        e = self.T if end_idx is None else end_idx
        s = max(0, s)
        e = min(self.T, e)
        self.valid_start = s
        self.valid_end = e
        # Pre-compute window start indices
        self.starts = []
        for t0 in range(self.valid_start, self.valid_end - H + 1):
            self.starts.append(t0)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        c_seq = self.controls[t0:t0+H]            # [H, d_u]
        y_seq = self.targets[t0:t0+H]             # [H, d_y]
        y0 = y_seq[0]                             # [d_y]
        return c_seq, y_seq, y0

# -----------------------------
# 4) Control interpolator for ODE
# -----------------------------
class ControlInterpolator:
    """
    Piecewise-linear interpolation of controls U(t) given discrete times t_ref=[0..H-1].
    Supports batch controls: U has shape [B, H, d_u] or [H, d_u] (broadcast to batch).
    """
    def __init__(self, t_ref, U):
        # t_ref: [H] tensor ascending
        self.t_ref = t_ref.contiguous()                       # [H]
        self.U = U.contiguous()                                # either [H, d_u] or [B, H, d_u]
        self.H = t_ref.shape[0]
        self.du = U.shape[-1]
        self.has_batch = (U.dim() == 3)

    def __call__(self, t):
        """
        t: scalar tensor or shape [B]
        returns u(t): [B, d_u] if batch, else [d_u]
        """
        # Ensure t is tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=self.t_ref.dtype, device=self.t_ref.device)
        # Positions via searchsorted
        # pos = index of first t_ref >= t
        pos = torch.searchsorted(self.t_ref, t)
        left = torch.clamp(pos - 1, 0, self.H - 1)
        right = torch.clamp(pos, 0, self.H - 1)
        t_left = self.t_ref[left]
        t_right = self.t_ref[right]
        # Avoid division by zero if left==right
        denom = torch.where((t_right - t_left) == 0, torch.ones_like(t_right), (t_right - t_left))
        alpha = (t - t_left) / denom  # in [0,1]
        # Gather U_left and U_right
        if self.has_batch and t.dim() > 0:
            B = t.shape[0]
            idx_left = left.view(B, 1, 1).expand(B, 1, self.du)
            idx_right = right.view(B, 1, 1).expand(B, 1, self.du)
            # advanced indexing
            U_left = self.U[torch.arange(B), left]   # [B, d_u]
            U_right = self.U[torch.arange(B), right] # [B, d_u]
            u = (1 - alpha.view(B, 1)) * U_left + alpha.view(B, 1) * U_right
            return u
        else:
            # single sample (no batch)
            U_left = self.U[left] if not self.has_batch else self.U[0, left]
            U_right = self.U[right] if not self.has_batch else self.U[0, right]
            u = (1 - alpha) * U_left + alpha * U_right
            return u

# -----------------------------
# 5) Neural ODE components
# -----------------------------
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, control_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + control_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )
        # Will be set externally per batch
        self.control_interp = None

    def set_control(self, interp):
        self.control_interp = interp

    def forward(self, t, z):
        """
        z: [B, latent_dim] ; t: scalar or [B]
        """
        assert self.control_interp is not None, "Control interpolator not set"
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]
        # For a batch, build t per sample
        if torch.is_tensor(t) and t.dim() == 0:
            t_batch = t.expand(B)
        elif torch.is_tensor(t) and t.dim() == 1:
            t_batch = t
        else:
            t_batch = torch.tensor([t]*B, dtype=z.dtype, device=z.device)
        u_t = self.control_interp(t_batch)        # [B, d_u]
        dz = self.net(torch.cat([z, u_t], dim=-1))
        return dz

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x0):
        return self.net(x0)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z_t):
        return self.net(z_t)

class NeuralODEModel(nn.Module):
    def __init__(self, latent_dim, control_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim=output_dim + control_dim, latent_dim=latent_dim)
        self.odefunc = ODEFunc(latent_dim=latent_dim, control_dim=control_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=output_dim)

    def forward(self, y0, controls_seq, t_span):
        """
        y0: [B, d_y]              initial measured outputs (normalized)
        controls_seq: [B, H, d_u] normalized controls for the horizon
        t_span: [H] tensor, e.g., torch.linspace(0, H-1, H)
        Returns y_hat_seq: [H, B, d_y]
        """
        B, H_local, d_u = controls_seq.shape
        assert H_local == t_span.shape[0], "controls_seq length and t_span must match"
        # Encode initial condition using y0 + controls at t0
        u0 = controls_seq[:, 0, :]                 # [B, d_u]
        enc_in = torch.cat([y0, u0], dim=-1)       # [B, d_y + d_u]
        z0 = self.encoder(enc_in)                  # [B, latent_dim]

        # Set control interpolator for current batch
        interp = ControlInterpolator(t_ref=t_span, U=controls_seq)
        self.odefunc.set_control(interp)

        # Integrate
        z_t = odeint(self.odefunc, z0, t_span, method='rk4')   # [H, B, latent_dim]
        # Decode each step
        Hn, Bn, L = z_t.shape
        z_flat = z_t.reshape(Hn * Bn, L)
        y_hat_flat = self.decoder(z_flat)                      # [H*B, d_y]
        y_hat = y_hat_flat.view(Hn, Bn, -1)                    # [H, B, d_y]
        return y_hat

# -----------------------------
# 6) Training / Evaluation
# -----------------------------
def make_dataloader(controls, targets, mean_std_controls, mean_std_targets, batch_size, start_idx=None, end_idx=None, shuffle=True):
    c_mean, c_std = mean_std_controls
    y_mean, y_std = mean_std_targets
    controls_n = normalize(controls, c_mean, c_std)
    targets_n = normalize(targets, y_mean, y_std)
    ds = WindowedDataset(controls_n, targets_n, start_idx=start_idx, end_idx=end_idx)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def train_loop(model, train_loader, val_loader, epochs, warmup, patience, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_epoch = -1
    no_improve = 0
    ckpt_path = os.path.join(OUTDIR, "best_model.pt")
    t_span = torch.linspace(0, H-1, H, dtype=torch.float32, device=device)

    for epoch in range(1, epochs+1):
        model.train()
        t0 = time.time()

        # Warmup: linear scale LR for first 'warmup' epochs
        if warmup > 0 and epoch <= warmup:
            scale = epoch / float(max(1, warmup))
            for g in optimizer.param_groups:
                g['lr'] = lr * scale
        else:
            for g in optimizer.param_groups:
                g['lr'] = lr

        train_loss = 0.0
        n_train = 0
        for controls_seq, y_seq, y0 in train_loader:
            # Move to device
            controls_seq = controls_seq.to(device)        # [B, H, d_u]
            y_seq = y_seq.to(device)                      # [B, H, d_y]
            y0 = y0.to(device)                            # [B, d_y]

            optimizer.zero_grad()
            # Forward: predict sequence
            y_hat_seq = model(y0, controls_seq, t_span)   # [H, B, d_y]
            # Align dims for loss
            y_hat_seq = y_hat_seq.transpose(0,1)          # [B, H, d_y]
            loss = criterion(y_hat_seq, y_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * controls_seq.size(0)
            n_train += controls_seq.size(0)

        train_loss /= max(1, n_train)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for controls_seq, y_seq, y0 in val_loader:
                controls_seq = controls_seq.to(device)
                y_seq = y_seq.to(device)
                y0 = y0.to(device)
                y_hat_seq = model(y0, controls_seq, t_span).transpose(0,1)
                loss = criterion(y_hat_seq, y_seq)
                val_loss += loss.item() * controls_seq.size(0)
                n_val += controls_seq.size(0)
        val_loss /= max(1, n_val)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.4e} | "
              f"train={train_loss:.6f} | val={val_loss:.6f} | {dt:.2f}s")

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "best_val": best_val,
                        "epoch": epoch}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best val={best_val:.6f}")
                break

    print(f"Training finished. Best val={best_val:.6f} at epoch {best_epoch}")
    return ckpt_path

def evaluate(model, loader, mean_std_targets, device="cpu"):
    """Return MAE and RMSE in normalized space."""
    model.to(device)
    model.eval()
    mae_sum, rmse_sum, n = 0.0, 0.0, 0
    criterion_mse = nn.MSELoss(reduction='none')
    with torch.no_grad():
        t_span = torch.linspace(0, H-1, H, dtype=torch.float32, device=device)
        for controls_seq, y_seq, y0 in loader:
            controls_seq = controls_seq.to(device)
            y_seq = y_seq.to(device)
            y0 = y0.to(device)
            y_hat_seq = model(y0, controls_seq, t_span).transpose(0,1)  # [B, H, d_y]
            err = torch.abs(y_hat_seq - y_seq)                          # [B, H, d_y]
            mae_sum += err.mean().item() * controls_seq.size(0)
            mse = criterion_mse(y_hat_seq, y_seq).mean(dim=(1,2))       # [B]
            rmse_sum += torch.sqrt(mse).mean().item() * controls_seq.size(0)
            n += controls_seq.size(0)
    mae = mae_sum / max(1, n)
    rmse = rmse_sum / max(1, n)
    print(f"[Eval] Normalized MAE={mae:.6f} | Normalized RMSE={rmse:.6f}")
    # Optionally denormalize to get physical units:
    y_mean, y_std = mean_std_targets
    # If needed, user can denormalize externally with denormalize()
    return {"mae_norm": mae, "rmse_norm": rmse}

# -----------------------------
# 7) Main: load data, build loaders, train, evaluate
# -----------------------------
def main(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load train & test CSVs
    tr_headers, tr_rows = read_csv_as_dicts(config["train_data"])
    te_headers, te_rows = read_csv_as_dicts(config["test_data"])
    # Build matrices
    tr_controls = build_matrix(tr_rows, CONTROL_FEATURES)     # [T_tr, d_u]
    tr_targets = build_matrix(tr_rows, TARGET_FEATURES)       # [T_tr, d_y]
    te_controls = build_matrix(te_rows, CONTROL_FEATURES)     # [T_te, d_u]
    te_targets = build_matrix(te_rows, TARGET_FEATURES)       # [T_te, d_y]

    # Norm stats from training only
    c_mean, c_std = compute_norm_stats(tr_controls)
    y_mean, y_std = compute_norm_stats(tr_targets)
    torch.save({"c_mean": c_mean, "c_std": c_std, "y_mean": y_mean, "y_std": y_std},
               os.path.join(OUTDIR, "scalers.pt"))

    # Split a validation tail from train (e.g., last 10% windows)
    T_tr = tr_controls.shape[0]
    val_tail = max(H, int(0.1 * T_tr))
    train_loader = make_dataloader(tr_controls, tr_targets, (c_mean, c_std), (y_mean, y_std),
                                   batch_size=BATCH_SIZE, start_idx=0, end_idx=T_tr - val_tail, shuffle=True)
    val_loader = make_dataloader(tr_controls, tr_targets, (c_mean, c_std), (y_mean, y_std),
                                 batch_size=BATCH_SIZE, start_idx=T_tr - val_tail - H, end_idx=None, shuffle=False)

    
    # In make_dataloader or before passing to WindowedDataset:
    

    def safe_int_or_none(x):
        # Handles None, "None", and numeric strings
        if x is None:
            return None
        if isinstance(x, str):
            if x.strip().lower() == "none":
                return None
            try:
                return int(x)
            except ValueError:
                return None  # fallback: treat invalid as None
        return int(x)


    # Test loader respects metrics start/end in config (e.g., last 500 samples)
    
    start_idx = safe_int_or_none(config["metrics"]["start_idx"])
    end_idx = safe_int_or_none(config["metrics"]["end_idx"])

    
    test_loader = make_dataloader(te_controls, te_targets, (c_mean, c_std), (y_mean, y_std),
                                  batch_size=BATCH_SIZE, start_idx=start_idx, end_idx=end_idx, shuffle=False)

    # Build model
    d_u = tr_controls.shape[1]
    d_y = tr_targets.shape[1]
    model = NeuralODEModel(latent_dim=LATENT_DIM, control_dim=d_u, output_dim=d_y)

    # Train
    ckpt_path = train_loop(model, train_loader, val_loader,
                           epochs=EPOCHS, warmup=WARMUP, patience=PATIENCE,
                           lr=1e-3, device=device)

    # Load best checkpoint and evaluate on test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate(model, test_loader, (y_mean, y_std), device=device)
    torch.save({"metrics": metrics}, os.path.join(OUTDIR, "metrics.pt"))
    print("Done.")

if __name__ == "__main__":
    main()
