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
if not os.path.exists(ckpt_path):
	raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train and place `best_model.pt` in out/ first.")
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)

# Dummy input for tracing/script
y0 = torch.randn(BATCH_SIZE, d_y)
controls_seq = torch.randn(BATCH_SIZE, H, d_u)
t_span = torch.linspace(0, H - 1, H, dtype=torch.float32)

jit_base = os.path.join(base_dir, "out", "neural_ode_model_jit")

print("Attempting torch.jit.trace(model, (y0, controls_seq, t_span)) — tracing may fail if model uses Python constructs (e.g., torchdiffeq.odeint)")
try:
	traced = torch.jit.trace(model, (y0, controls_seq, t_span), strict=False)
	traced_path = jit_base + ".pt"
	traced.save(traced_path)
	print(f"Saved traced model to {traced_path}")
except Exception as e:
	print("Tracing failed:", e)
	print("Attempting torch.jit.script(model) as a fallback — this may also fail if the model uses non-scriptable Python constructs.")
	try:
		scripted = torch.jit.script(model)
		scripted_path = jit_base + "_scripted.pt"
		scripted.save(scripted_path)
		print(f"Saved scripted model to {scripted_path}")
	except Exception as e2:
		print("Scripting failed:", e2)
		print("Model likely uses constructs (e.g., torchdiffeq.odeint or dynamic control interpolation) that are not TorchScriptable.")
		print("Fallback: tracing encoder and decoder separately so C++ can use them for parts of the pipeline.")
		try:
			# Trace encoder: input is concatenated [y0, u0]
			enc_in = torch.cat([y0, controls_seq[:, 0, :]], dim=-1)
			traced_enc = torch.jit.trace(model.encoder, enc_in)
			traced_enc_path = os.path.join(base_dir, "out", "encoder_jit.pt")
			traced_enc.save(traced_enc_path)
			print(f"Saved traced encoder to {traced_enc_path}")

			# Trace decoder: input is a latent vector
			dec_in = torch.randn(1, LATENT_DIM)
			traced_dec = torch.jit.trace(model.decoder, dec_in)
			traced_dec_path = os.path.join(base_dir, "out", "decoder_jit.pt")
			traced_dec.save(traced_dec_path)
			print(f"Saved traced decoder to {traced_dec_path}")

			print("Notes:")
			print("- The encoder/decoder traces allow C++ to encode initial conditions and decode latent states,")
			print("  but the latent integration (torchdiffeq.odeint) is Python-based and not TorchScriptable in general.")
			print("- Options: re-implement a fixed-step integrator (RK4) in pure TorchScript and wrap ODEFunc, or use the ONNX export script `to_onnx.py`.")
		except Exception as e3:
			print("Failed to trace encoder/decoder:", e3)
			print("Final suggestion: use `to_onnx.py` to export to ONNX for C++ consumption or implement a scriptable integrator.")

print("Done.")

