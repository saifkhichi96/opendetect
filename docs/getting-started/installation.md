# Installation

## Base Installation

```bash
pip install opendetect
```

The base install includes:

- `numpy`
- `opencv-python`
- `onnx`
- `onnxruntime` (CPU-capable runtime)

## Runtime Extras

Install a runtime extra when you want explicit backend control:

```bash
# Explicit CPU runtime package
pip install "opendetect[cpu]"

# CUDA-enabled ONNX Runtime package
pip install "opendetect[gpu]"

# TensorRT + GPU runtime Python dependencies
pip install "opendetect[tensorrt]"
```

Install ONNX simplify tooling when needed:

```bash
pip install "opendetect[simplify]"
```

## Important TensorRT Note

`opendetect[tensorrt]` installs Python-level dependencies only. You still need a working TensorRT system installation and a compatible NVIDIA stack (GPU, driver, CUDA, and cuDNN).

See [Runtime Backends](runtimes.md#tensorrt-setup) for setup and verification steps.

## Install From Source

```bash
git clone https://github.com/saifkhichi96/opendetect.git
cd opendetect
pip install -e .
```

With optional extras:

```bash
pip install -e ".[gpu,simplify]"
```

## Environment Variables

- `OPENDETECT_CACHE_DIR`: custom path for downloaded model artifacts.
- `OPENDETECT_MODEL_BASE_URL`: custom model registry base URL.
