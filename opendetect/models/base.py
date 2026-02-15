from __future__ import annotations

import hashlib
import os
import platform
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort

Detections = dict[str, np.ndarray]

ProviderEntry = Union[str, Tuple[str, Dict[str, Any]]]

# Some builds report DirectML as DmlExecutionProvider
_DIRECTML_ALIASES = {"DmlExecutionProvider", "DirectMLExecutionProvider"}


def _user_cache_root() -> str:
    sys = platform.system()
    home = os.path.expanduser("~")
    if sys == "Darwin":
        base = os.path.join(home, "Library", "Caches")
    elif sys == "Windows":
        base = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.join(home, ".cache"))
    path = os.path.join(base, "onnxrt_ep_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _ep_cache_dir(ep_name: str) -> str:
    path = os.path.join(_user_cache_root(), ep_name)
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_metadata_cache_key(onnx_path: str, key_name: str = "CACHE_KEY") -> None:
    """
    Ensure the ONNX model contains a stable cache key in metadata_props.
    Used by CoreML EP to reuse compiled models. Non-fatal if it fails.
    """
    try:
        m = onnx.load(onnx_path)
        if any(mp.key == key_name for mp in m.metadata_props):
            return

        h = hashlib.sha256()
        with open(onnx_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        digest = h.hexdigest()[:64]

        kv = m.metadata_props.add()
        kv.key = key_name
        kv.value = digest
        onnx.save(m, onnx_path)
    except Exception as e:
        warnings.warn(f"Unable to set CoreML cache metadata on '{onnx_path}': {e}")


def _available_providers() -> List[str]:
    return list(ort.get_available_providers())


def _has(ep: str, available: Iterable[str]) -> bool:
    if ep in _DIRECTML_ALIASES:
        return any(a in _DIRECTML_ALIASES for a in available)
    return ep in set(available)


def _canonical_directml_name(available: Iterable[str]) -> Optional[str]:
    for n in _DIRECTML_ALIASES:
        if n in available:
            return n
    return None


def resolve_execution_providers(
    hardware_acceleration: bool = True,
    tensor_rt: bool = False,
) -> List[str]:
    """
    Decide an ordered list of EPs to request, from most preferred to least.
    Always ends with CPUExecutionProvider as a safety net.
    """
    if not hardware_acceleration:
        return ["CPUExecutionProvider"]

    avail = _available_providers()

    order: List[str] = []

    # Apple first: CoreML is fastest when available on macOS
    if _has("CoreMLExecutionProvider", avail):
        order.append("CoreMLExecutionProvider")

    # NVIDIA stack
    cuda_ok = _has("CUDAExecutionProvider", avail)
    trt_ok = _has("TensorrtExecutionProvider", avail) and tensor_rt
    nvr_ok = _has("NvTensorRtRtxExecutionProvider", avail) and tensor_rt

    if nvr_ok and cuda_ok:
        order += [
            "NvTensorRtRtxExecutionProvider",
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
        ]
    elif trt_ok and cuda_ok:
        order += ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    elif cuda_ok:
        order += ["CUDAExecutionProvider"]

    # AMD stack
    if _has("ROCMExecutionProvider", avail):
        order.append("ROCMExecutionProvider")
    if _has("MIGraphXExecutionProvider", avail):
        order.append("MIGraphXExecutionProvider")

    # DirectML (Windows + DX12 GPUs)
    dml_name = _canonical_directml_name(avail)
    if dml_name:
        order.append(dml_name)

    # OpenVINO (Intel)
    if _has("OpenVINOExecutionProvider", avail):
        order.append("OpenVINOExecutionProvider")

    # Always include CPU last
    order.append("CPUExecutionProvider")

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for p in order:
        key = "DmlExecutionProvider" if p in _DIRECTML_ALIASES else p
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return deduped


def provider_options_for(
    ep: str,
    *,
    onnx_model: str,
    mixed_precision: bool,
) -> Optional[ProviderEntry]:
    """
    Return a (name, options) tuple for EPs that benefit from tuned options.
    Return just the string name (or None) to skip options.
    Options are conservative & known-good; if an EP rejects them, we retry without.
    """
    if ep == "CoreMLExecutionProvider":
        _ensure_metadata_cache_key(onnx_model)
        return (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "0",
                "EnableOnSubgraphs": "1",
                "SpecializationStrategy": "FastPrediction",
                "ProfileComputePlan": "0",
                "AllowLowPrecisionAccumulationOnGPU": "1" if mixed_precision else "0",
                "ModelCacheDirectory": _ep_cache_dir("CoreML"),
            },
        )

    if ep in ("TensorrtExecutionProvider", "NvTensorRtRtxExecutionProvider"):
        return (
            ep,
            {
                # Mixed precision through FP16; INT8 intentionally off unless model is calibrated.
                "trt_fp16_enable": True if mixed_precision else False,
                "trt_int8_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": _ep_cache_dir("TensorRT"),
                "trt_timing_cache_enable": True,
                "trt_max_workspace_size": str(
                    1 << 30
                ),  # 1GB workspace as a balanced default
            },
        )

    if ep == "CUDAExecutionProvider":
        return (
            "CUDAExecutionProvider",
            {
                # Enable TF32 for tensor cores when mixed_precision requested (widely beneficial on Ampere+)
                "use_tf32": 1 if mixed_precision else 0,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                # Leave gpu_mem_limit unset so ORT auto-manages
            },
        )

    if ep == "ROCMExecutionProvider":
        return (
            "ROCMExecutionProvider",
            {
                "miopen_conv_exhaustive_search": 0,  # faster startup; exhaustive search can be slow
                "tunable_op_enable": 1,
                "tunable_op_tuning_enable": 1,
                "enable_hip_graph": 1,
            },
        )

    if ep == "MIGraphXExecutionProvider":
        # MIGraphX is conservative; options surface varies by build. Keep minimal.
        return (
            "MIGraphXExecutionProvider",
            {
                # Prefer float16 when requested; MIGraphX respects model precision
                "fp16_enable": 1 if mixed_precision else 0,
            },
        )

    if ep in _DIRECTML_ALIASES:
        # Use the exact available name
        name = ep
        return (
            name,
            {
                # Let DML choose best operator implementations; TF32 toggle is effectively no-op for many ops
                "enable_metacommands": 1,
                "debug_output": 0,
            },
        )

    if ep == "OpenVINOExecutionProvider":
        return (
            "OpenVINOExecutionProvider",
            {
                "device_type": "AUTO",  # AUTO picks CPU/GPU/VPU
                "enable_dynamic_shapes": "YES",
                "cache_dir": _ep_cache_dir("OpenVINO"),
                "num_of_threads": max(os.cpu_count() // 2, 1),
            },
        )

    if ep == "CPUExecutionProvider":
        # Let ORT decide; we already tune SessionOptions threads
        return "CPUExecutionProvider"

    # Unknown EP: fall back to requesting by name
    return ep


def build_provider_entries(
    providers: List[str],
    *,
    onnx_model: str,
    mixed_precision: bool,
) -> List[ProviderEntry]:
    entries: List[ProviderEntry] = []
    for p in providers:
        entry = provider_options_for(
            p, onnx_model=onnx_model, mixed_precision=mixed_precision
        )
        if entry is not None:
            entries.append(entry)
    return entries


def create_ort_session(
    model_path: Union[str, Path],
    hardware_acceleration: bool = True,
    tensor_rt: bool = False,
    mixed_precision: bool = True,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ort.InferenceSession:
    providers = resolve_execution_providers(
        hardware_acceleration=hardware_acceleration, tensor_rt=tensor_rt
    )
    provider_entries = build_provider_entries(
        providers, onnx_model=str(model_path), mixed_precision=mixed_precision
    )

    if overrides is not None:
        for entry in provider_entries:
            if isinstance(entry, tuple):
                name, options = entry
                if name in overrides:
                    for k, v in options.items():
                        if k in overrides[name]:
                            options[k] = overrides[name][k]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            path_or_bytes=model_path,
            sess_options=so,
            providers=provider_entries,
        )
    except RuntimeError:
        # If session creation fails, try without providers
        session = ort.InferenceSession(
            path_or_bytes=model_path,
            sess_options=so,
            providers=resolve_execution_providers(
                hardware_acceleration=hardware_acceleration,
                tensor_rt=False,
            ),
        )
    return session


class DetectorModel(Protocol):
    model_name: str
    model_path: Path
    input_size: tuple[int, int]
    class_names: list[str] | None

    @classmethod
    def default_class_ids(cls) -> list[int] | None: ...

    @classmethod
    def default_input_size(cls) -> tuple[int, int]: ...

    @classmethod
    def default_class_names(cls) -> list[str] | None: ...

    def preprocess_rgb_frame(
        self, image_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def forward(self, input_tensor: np.ndarray) -> tuple[np.ndarray, ...]: ...

    def postprocess(
        self,
        outputs: tuple[np.ndarray, ...],
        target_sizes: np.ndarray,
    ) -> list[Detections]: ...

    def predict_rgb_frame(self, image_rgb: np.ndarray) -> Detections: ...

    def draw_detections_on_bgr_frame(
        self, frame_bgr: np.ndarray, detections: Detections
    ) -> np.ndarray: ...
