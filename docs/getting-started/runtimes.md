# Runtime Backends

OpenDetect uses ONNX Runtime execution providers (EPs). Provider selection is automatic and ordered by capability.

## Selection Rules

Provider resolution currently follows this order:

1. CoreML (macOS)
2. NVIDIA stack (TensorRT RTX, TensorRT, CUDA)
3. AMD stack (ROCM, MIGraphX)
4. DirectML (Windows)
5. OpenVINO (Intel)
6. CPU fallback

You control behavior with:

- `hardware_acceleration`: default `True`, set `False` to force CPU
- `tensor_rt`: default `False`, set `True` to allow TensorRT providers
- `mixed_precision`: default `False` at CLI, can be enabled for FP16/TF32-friendly paths

## TensorRT Setup

`pip install "opendetect[tensorrt]"` is not sufficient by itself.

You must also install a compatible TensorRT system stack:

1. NVIDIA driver compatible with your GPU.
2. CUDA toolkit and cuDNN compatible with your ONNX Runtime GPU package.
3. TensorRT runtime/libraries from NVIDIA (matching CUDA major version).
4. Python environment with `opendetect[tensorrt]`.

Verification:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

You should see `TensorrtExecutionProvider` (or `NvTensorRtRtxExecutionProvider`) and `CUDAExecutionProvider`.

CLI usage:

```bash
opendetect-infer --image input.jpg --model-id rfdetr-m --tensor-rt --mixed-precision
```

Known limitation:

- YOLOX `m/l/x` are not currently supported on TensorRT due to export/runtime compatibility issues.

## Intel and AMD Support Status

OpenDetect already includes selection logic for:

- Intel: `OpenVINOExecutionProvider`
- AMD: `ROCMExecutionProvider`, `MIGraphXExecutionProvider`
- Windows GPU path: `DmlExecutionProvider` / `DirectMLExecutionProvider`

Current status:

- implementation-level support exists in code
- formal hardware validation is pending for Intel and AMD environments
- contributions with tested setup notes are welcome

## Troubleshooting

- If acceleration is not used, print available providers and verify expected EP names.
- Confirm your ONNX Runtime build matches the desired hardware backend.
- Fall back to `--no-hardware-acceleration` to confirm functional CPU inference first.
