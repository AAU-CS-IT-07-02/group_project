"""
Simple FastAPI server that preloads a model and exposes `/predict`.

Behavior:

- If `model.pt` exists in this folder, attempt to load with `torch.jit.load` or `torch.load`.
- If PyTorch or model file is missing, fall back to a deterministic dummy function.

Endpoint:
`POST /predict`

body: 
```
{
  "y0": [v1, v2, ...],          # initial output vector
  "controls": [[u11, u12, ...],  # list of control vectors
               [u21, u22, ...],
  "method": "rk4",
  "is_normalized": 0
}  # batch of input vectors

resp: {"prediction": [y1, y2, ...]}  # one output per input row
```

Run locally:
```
$ pip install -r requirements.txt
$ uvicorn server:app --host 127.0.0.1 --port 8000
```
"""
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, HTTPException

# Optional deep model support
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import runpy
import os
import yaml

# Paths for locating the training module and default model output
BASE_DIR = Path(__file__).resolve().parent.parent
PYTORCH_NODE_DIR = BASE_DIR.parent / "thermodynamics_modeling" / "pythorch_node"
TRAINER_MODULE = PYTORCH_NODE_DIR / "torchdiffeq_model.py"
CONFIG_YML = PYTORCH_NODE_DIR / "config.yml"
# Default model output directory (the training scripts save scalers.pt / best_model.pt here)
DEFAULT_MODEL_OUT = PYTORCH_NODE_DIR / "out"


class SeqInput(BaseModel):
    y0: List[float]
    controls: List[List[float]]  # [H, d_u]
    method: Optional[str] = "rk4"
    is_normalized: Optional[int] = 1
    debug: Optional[int] = 0


class ModelWrapper:
    """
    High-level container for loading a trained NeuralODE model, its associated
    scalers, and running sequential predictions. The wrapper provides a safe
    fallback deterministic behaviour when PyTorch or expected artifacts are
    unavailable, and exposes a single convenience method predict_sequence for
    running multi-step inference.
    
    Behavior summary
    
    - On initialization, attempts to locate and import a trainer module (expected
        to provide: NeuralODEModel, normalize, denormalize), read configuration to
        infer a latent dimension, and load scalers and a checkpoint from a model
        output directory. The model output directory can be provided via the
        model_out_dir parameter or overridden by the environment variable
        MODEL_OUT_DIR; a module-level DEFAULT_MODEL_OUT is used as a last resort.
    - If any required dependency, module, or file is missing (e.g. TORCH not
        available, trainer module not found, scalers/checkpoint absent, or load
        errors), the wrapper falls back to a deterministic simple predictor and
        sets .model to None.
    - When a checkpoint lacks explicit metadata about latent dimension, a
        default latent_dim=16 is used.
    - Loaded NeuralODE model is constructed with the inferred control/output
        dimensions (from scalers) and the latent dimension, then put into eval mode.
    """

    def __init__(self, model_out_dir: Optional[Path] = None):
        self.model = None
        self.normalize = None
        self.denormalize = None
        self.NeuralODEModel = None
        self.c_mean = None
        self.c_std = None
        self.y_mean = None
        self.y_std = None
        self.device = "cpu"
        # Allow overriding model output directory via env var MODEL_OUT_DIR
        env_dir = os.environ.get("MODEL_OUT_DIR")
        if model_out_dir is not None:
            self.model_out_dir = Path(model_out_dir)
        elif env_dir:
            self.model_out_dir = Path(env_dir)
        else:
            self.model_out_dir = DEFAULT_MODEL_OUT
        self._load()

    def _load(self):
        # If Torch unavailable, keep fallback deterministic behavior
        if not TORCH_AVAILABLE:
            print("Torch not available: server will use deterministic fallback.")
            self.model = None
            return

        # Try to locate trainer module and load helper symbols
        if not TRAINER_MODULE.exists():
            print(f"Trainer module not found at {TRAINER_MODULE}; using simple fallback.")
            self.model = None
            return

        try:
            trainer_ns = runpy.run_path(str(TRAINER_MODULE))
            # Expected symbols in trainer module
            self.NeuralODEModel = trainer_ns.get("NeuralODEModel")
            self.normalize = trainer_ns.get("normalize")
            self.denormalize = trainer_ns.get("denormalize")
        except Exception as e:
            print("Failed to import trainer module:", e)
            self.model = None
            return

        # Load config to get latent_dim (if available)
        latent_dim = None
        try:
            if CONFIG_YML.exists():
                with open(CONFIG_YML, "r") as f:
                    cfg = yaml.safe_load(f)
                    latent_dim = int(cfg.get("model", {}).get("latent_space_dimensions", 0) or 0)
        except Exception:
            latent_dim = None

        # Load scalers and checkpoint from model_out_dir
        scalers_path = self.model_out_dir / "scalers.pt"
        ckpt_path = self.model_out_dir / "best_model.pt"
        if not scalers_path.exists() or not ckpt_path.exists():
            print(f"Scalers or checkpoint not found in {self.model_out_dir}; falling back.")
            self.model = None
            return

        try:
            scalers = torch.load(str(scalers_path), map_location=self.device)
            self.c_mean = scalers["c_mean"].to(self.device)
            self.c_std = scalers["c_std"].to(self.device)
            self.y_mean = scalers["y_mean"].to(self.device)
            self.y_std = scalers["y_std"].to(self.device)
        except Exception as e:
            print("Failed to load scalers:", e)
            self.model = None
            return

        # Infer control/output dims from scalers
        try:
            d_u = int(self.c_mean.numel())
            d_y = int(self.y_mean.numel())
        except Exception as e:
            print("Failed to infer dims from scalers:", e)
            self.model = None
            return

        # If latent_dim not found in config, try to infer from checkpoint metadata
        try:
            ckpt = torch.load(str(ckpt_path), map_location=self.device)
            # Some checkpoints include metadata; try to read latent dim
            latent_dim = latent_dim or int(ckpt.get("latent_dim", 0) or 0)
        except Exception as e:
            print("Failed to inspect checkpoint for metadata:", e)

        if not latent_dim or latent_dim <= 0:
            # As a last resort choose a reasonable default
            latent_dim = 32
            print(f"Using default latent_dim={latent_dim}")

        # Instantiate model and load weights
        try:
            model = self.NeuralODEModel(latent_dim=latent_dim, control_dim=d_u, output_dim=d_y)
            ckpt = torch.load(str(ckpt_path), map_location=self.device)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                # Try loading entire checkpoint if it is a state_dict
                model.load_state_dict(ckpt)
            model.eval()
            self.model = model
            print("Loaded NeuralODE model successfully.")
        except Exception as e:
            print("Failed to build/load NeuralODE model:", e)
            self.model = None

    def predict_sequence(self, y0: List[float], controls: List[List[float]], method: str = "rk4", debug: bool = False, is_normalized: int = 1):
        """Run sequence inference using the loaded NeuralODEModel.

        Args:
            y0: initial output vector [d_y]
            controls: list of H control vectors [[d_u], ...]
        Returns:
            prediction: list of H output vectors (denormalized)
        """
        # Fallback deterministic behavior if model not available
        if self.model is None or not TORCH_AVAILABLE:
            arr = np.asarray(controls, dtype=float)
            H = arr.shape[0]
            # Repeat simple scalar value for each step
            s = float(np.sum(y0)) * 0.0 + float(np.sum(arr.mean(axis=1))) * 0.1
            fallback = [[s for _ in range(len(y0))] for _ in range(H)]
            if debug == 1:
                # no normalized values available for fallback
                return {"denorm": fallback, "norm": None}
            return fallback

        # Convert inputs to tensors and normalize
        with torch.no_grad():
            device = self.device
            t_controls = torch.tensor(np.asarray(controls, dtype=np.float32), device=device)  # [H, d_u]
            t_y0 = torch.tensor(np.asarray(y0, dtype=np.float32), device=device)              # [d_y]

            # Normalize using scalers
            if is_normalized == 1:
                controls_n = t_controls
                y0_n = t_y0
            else:
                controls_n = self.normalize(t_controls, self.c_mean, self.c_std)  # [H, d_u]
                y0_n = self.normalize(t_y0, self.y_mean, self.y_std)              # [d_y]

            # Add batch dim [1, H, d_u] and [1, d_y]
            controls_n = controls_n.unsqueeze(0)
            y0_n = y0_n.unsqueeze(0)

            H = controls_n.shape[1]
            t_span = torch.linspace(0, H - 1, H, dtype=torch.float32, device=device)

            out = self.model(y0_n, controls_n, t_span, method=method)  # expected [H, B, d_y]
            # Convert to [H, d_y]
            out = out.squeeze(1)
            # Keep normalized output for debugging
            out_norm = out.cpu()
            # Denormalize
            out_denorm = self.denormalize(out, self.y_mean, self.y_std)
            out_np = out_denorm.cpu().numpy().tolist()
            if debug == 1:
                return {"denorm": out_np, "norm": out_norm.numpy().tolist()}
            return out_np

app = FastAPI(title="UPPAAL Neural ODE Inference API", version="0.1")
model = ModelWrapper()


@app.get("/health")
def health():
    """
    GET endpoint for health check.

    Response:

      {"ok": True, "model_loaded": bool}
    """
    return {"ok": True, "model_loaded": model.model is not None}


@app.post("/infer")
def infer(req: SeqInput):
    """
    POST endpoint for running model inference.

    Request JSON:

      `{"y0": [..], "controls": [[...], ...], "method": "rk4", "is_normalized": 1}`

    Response:
    
      `{"prediction": [[...], ...]}  # the predicted output sequence normalized or denormalized based on is_normalized flag`
    """
    try:
        preds = model.predict_sequence(req.y0, req.controls, method=req.method, debug=req.debug, is_normalized=req.is_normalized)
        print("[INFO]: Inference successful.", preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction": preds}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
