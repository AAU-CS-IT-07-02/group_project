**Summary:** Calling a Python Neural ODE Model from UPPAAL via C → HTTP API
**Summary**

This folder contains a small FastAPI inference server and a C wrapper that UPPAAL can call. The design keeps native functions deterministic and exposes simple scalar-returning functions that UPPAAL can safely use.

Key points about the current implementation

- `server.py`: FastAPI server exposing `POST /infer`. It will load a trained Neural ODE model if available (and PyTorch is installed), otherwise it returns a deterministic fallback prediction.
- `uppaal_wrapper.c`: C wrapper that uses `libcurl` + `cJSON` to POST JSON to the server and parse results.
  - `uppaal_nn_infer_scalar_fixed(...)`: compatibility function returning a single scalar prediction for a requested room index (keeps existing UPPAAL usage semantics).
  - `uppaal_nn_update(...)`: performs the HTTP call and stores the first predicted row in an internal static buffer (`latest_prediction`). Returns `1` on success, `0` on failure.
  - `uppaal_nn_get_pred(int room_id)`: returns the stored scalar prediction for `room_id` (or `NAN` if out-of-range).

Files of interest

- `server.py` — FastAPI server and model wrapper
- `uppaal_wrapper.c` — C wrapper with `uppaal_nn_update` and `uppaal_nn_get_pred`
- `cJSON.c` / `cJSON.h` — JSON helper (used by the C wrapper) made by [Dave Gamble](https://github.com/DaveGamble/cJSON)
- `UPPAAL_integration_example.txt` — example UPPAAL global declarations and templates

Quick start

1) Create and activate a Python venv and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Run the server (bind to localhost port 8000)

```bash
cd $(dirname "$0")
uvicorn server:app --host 127.0.0.1 --port 8000
```

3) Build the shared library for UPPAAL to load

```bash
gcc -shared -fPIC uppaal_wrapper.c cJSON.c -o libuppaal_nn.so -lcurl -lm
```

Usage from UPPAAL (recommended pattern)

UPPAAL cannot directly receive arrays from external C functions. Use this pattern:

1. From UPPAAL call `uppaal_nn_update(...)` once to perform the HTTP inference and cache the first row of predictions in the C library.
2. For each room index `i`, call `uppaal_nn_get_pred(i)` which returns a scalar `double` with the predicted temperature for that room.

This approach keeps calls scalar, deterministic, and compatible with UPPAAL's external function support.

Example UPPAAL flow (SMC / doubles supported)

1. Declare the externs in UPPAAL globals:
   - `extern int uppaal_nn_update(...);`
   - `extern double uppaal_nn_get_pred(int);`
2. In a transition action call:
   - `uppaal_nn_update(...args...);`
   - `t0 = uppaal_nn_get_pred(0); t1 = uppaal_nn_get_pred(1); ...`

Notes and tips

- Keep the server binding to `127.0.0.1` and a fixed port to avoid network nondeterminism.
- Keep timeouts short in the C wrapper (the code already sets a 3s timeout). For SMC, ensure any retry logic is deterministic or disabled.
- If you want a single-call-per-room interface, you can keep `uppaal_nn_infer_scalar_fixed(room_id, ...)` — it performs a POST and returns only the requested room value. The `update` + `get_pred` pattern is more efficient when you need all room values at once.

