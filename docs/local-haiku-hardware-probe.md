# Local Haiku Hardware Probe

Run the probe before optional neural or GPU experiments:

```bash
python3 scripts/probe_local_ml_env.py
python3 scripts/probe_local_ml_env.py --json
```

The command is read-only. It reports OS, Python, CPU/RAM, optional ML Python
libraries, GPU visibility, ROCm/HIP probes, Vulkan visibility, and a capability
classification:

- `cpu`: the required CPU n-gram baseline path should be used.
- `rocm-pytorch`: PyTorch reports a usable HIP/ROCm device, worth trying for the
  later GRU/LSTM follow-up.
- `onnx-rocm`: ONNX Runtime exposes ROCm, useful for ONNX experiments.
- `llama-vulkan`: Vulkan plus llama-cpp-python are visible, useful for later
  llama.cpp experiments.
- `unknown`: even the Python baseline requirements need attention.

CPU n-gram training is the required success path. Missing AMD GPU, missing
ROCm, or missing PyTorch must be treated as normal fallback conditions, not as a
training failure.

Downstream training commands should default to CPU. If a future command accepts
an acceleration flag, use `--device cpu` unless this probe reports a matching
optional path, and keep the n-gram model independent of PyTorch or ROCm.
