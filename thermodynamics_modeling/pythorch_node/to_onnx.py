import os
import torch
import yaml
from torchdiffeq_model import NeuralODEModel

# Load config
base_dir = os.path.dirname(__file__)
cfg_path = os.path.join(base_dir, "config.yml")
with open(cfg_path, "r") as f:
    config = yaml.safe_load(f)

# Model parameters from config
LATENT_DIM = int(config["model"]["latent_space_dimensions"])
H = int(config["dataset"]["H"])
BATCH_SIZE = 1  # For export, use batch size 1
CONTROL_FEATURES = config["observations"] + config["disturbances"] + config["outdoor"]
ROOMS_TEMP = config["rooms_temp"]

d_u = len(CONTROL_FEATURES)
d_y = len(ROOMS_TEMP)

# Build model and load weights
model = NeuralODEModel(latent_dim=LATENT_DIM, control_dim=d_u, output_dim=d_y)
ckpt_path = os.path.join(base_dir, "out", "best_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

# Dummy input for ONNX export
y0 = torch.randn(BATCH_SIZE, d_y)
controls_seq = torch.randn(BATCH_SIZE, H, d_u)
t_span = torch.linspace(0, H-1, H, dtype=torch.float32)

# Export to ONNX
onnx_path = os.path.join(base_dir, "out", "neural_ode_model.onnx")
torch.onnx.export(
    model,
    (y0, controls_seq, t_span),
    onnx_path,
    input_names=["y0", "controls_seq", "t_span"],
    output_names=["y_hat_seq"],
    dynamic_axes={
        "y0": {0: "batch"},
        "controls_seq": {0: "batch"},
        "y_hat_seq": {1: "batch"}
    },
    opset_version=17
)
print(f"Exported model to {onnx_path}")