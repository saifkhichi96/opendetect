from pathlib import Path
import argparse
import time

import numpy as np
import onnxruntime as ort

from opendetect import Detector
from opendetect._cli_utils import parse_optional_input_size, parse_required_providers
from opendetect.models import available_models
from opendetect.models.base import DetectorModel
from opendetect.registry import list_model_ids


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def print_stage_stats(name: str, values: list[float]) -> None:
    stats = summarize(values)
    print(
        f"{name:24} mean={stats['mean']:.3f} ms  median={stats['median']:.3f} ms  "
        f"p95={stats['p95']:.3f} ms  min={stats['min']:.3f} ms  max={stats['max']:.3f} ms"
    )


def _format_count(value: int | None) -> str:
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.3f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}G"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def _shape_numel(shape: tuple[int | None, ...] | None) -> int | None:
    if shape is None or len(shape) == 0:
        return None
    total = 1
    for dim in shape:
        if dim is None:
            return None
        total *= int(dim)
    return total


def _shape_map_from_onnx(model) -> dict[str, tuple[int | None, ...]]:
    shape_map: dict[str, tuple[int | None, ...]] = {}

    def read_value_info(value_info) -> tuple[int | None, ...] | None:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None

        dims: list[int | None] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        return tuple(dims)

    for value_info in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        shape = read_value_info(value_info)
        if shape is not None:
            shape_map[value_info.name] = shape

    for initializer in model.graph.initializer:
        shape_map[initializer.name] = tuple(int(dim) for dim in initializer.dims)

    return shape_map


def _attr_int(node, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _estimate_onnx_flops(model) -> tuple[int | None, str]:
    try:
        import onnx
    except ImportError:
        return None, "onnx package not available for FLOPs estimation"

    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model

    shape_map = _shape_map_from_onnx(inferred)
    total_flops = 0
    covered_nodes = 0

    for node in inferred.graph.node:
        op_type = node.op_type

        if op_type == "Conv" and len(node.input) >= 2 and len(node.output) >= 1:
            weight_shape = shape_map.get(node.input[1])
            output_shape = shape_map.get(node.output[0])
            if weight_shape is None or output_shape is None:
                continue
            if len(weight_shape) < 4 or len(output_shape) < 4:
                continue

            out_numel = _shape_numel(output_shape)
            if out_numel is None:
                continue

            kernel_elems_per_output = _shape_numel(weight_shape[1:])
            if kernel_elems_per_output is None:
                continue

            total_flops += 2 * out_numel * kernel_elems_per_output
            covered_nodes += 1
            continue

        if op_type == "Gemm" and len(node.input) >= 2:
            a_shape = shape_map.get(node.input[0])
            b_shape = shape_map.get(node.input[1])
            if a_shape is None or b_shape is None:
                continue
            if len(a_shape) != 2 or len(b_shape) != 2:
                continue

            trans_a = _attr_int(node, "transA", 0) != 0
            trans_b = _attr_int(node, "transB", 0) != 0

            m = a_shape[1] if trans_a else a_shape[0]
            k_a = a_shape[0] if trans_a else a_shape[1]
            k_b = b_shape[1] if trans_b else b_shape[0]
            n = b_shape[0] if trans_b else b_shape[1]

            if None in (m, k_a, k_b, n) or k_a != k_b:
                continue

            total_flops += 2 * int(m) * int(n) * int(k_a)
            covered_nodes += 1
            continue

        if op_type == "MatMul" and len(node.input) >= 2 and len(node.output) >= 1:
            a_shape = shape_map.get(node.input[0])
            b_shape = shape_map.get(node.input[1])
            out_shape = shape_map.get(node.output[0])
            if a_shape is None or b_shape is None or out_shape is None:
                continue
            if len(a_shape) < 2 or len(b_shape) < 2 or len(out_shape) < 2:
                continue

            batch_shape = out_shape[:-2]
            batch = _shape_numel(batch_shape) if len(batch_shape) > 0 else 1
            m = out_shape[-2]
            n = out_shape[-1]
            k_a = a_shape[-1]
            k_b = b_shape[-2]

            if None in (batch, m, n, k_a, k_b) or k_a != k_b:
                continue

            total_flops += 2 * int(batch) * int(m) * int(n) * int(k_a)
            covered_nodes += 1
            continue

    if covered_nodes == 0:
        return None, "unable to infer FLOPs for supported ops (Conv/Gemm/MatMul)"

    note = f"estimated from {covered_nodes} Conv/Gemm/MatMul nodes"
    return int(total_flops), note


def inspect_model_complexity(model_path: Path) -> tuple[int | None, int | None, str]:
    if model_path.suffix.lower() != ".onnx":
        return None, None, "model stats are implemented for ONNX models only"

    try:
        import onnx
    except ImportError:
        return None, None, "onnx package not available for model stats"

    try:
        onnx_model = onnx.load(str(model_path))
    except Exception as exc:
        return None, None, f"failed to load ONNX for stats: {exc}"

    param_count = 0
    for initializer in onnx_model.graph.initializer:
        count = 1
        for dim in initializer.dims:
            count *= int(dim)
        param_count += count

    flops, flops_note = _estimate_onnx_flops(onnx_model)
    return int(param_count), flops, flops_note


def run_dummy_benchmark(
    model: DetectorModel,
    warmup: int,
    iterations: int,
    dummy_height: int,
    dummy_width: int,
    seed: int,
) -> tuple[int, list[float], list[float], list[float], list[float], list[float]]:
    rng = np.random.default_rng(seed)
    total_frames = warmup + iterations

    frame_load_ms = []
    preprocess_ms = []
    inference_ms = []
    postprocess_ms = []
    total_ms = []

    for frame_index in range(total_frames):
        frame_load_start = time.perf_counter()
        frame_rgb = rng.integers(
            0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8
        )
        frame_load_end = time.perf_counter()

        preprocess_start = frame_load_end
        input_tensor, target_sizes = model.preprocess_rgb_frame(frame_rgb)
        preprocess_end = time.perf_counter()

        infer_start = preprocess_end
        outputs = model.forward(input_tensor)
        infer_end = time.perf_counter()

        post_start = infer_end
        _ = model.postprocess(outputs=outputs, target_sizes=target_sizes)
        post_end = time.perf_counter()

        if frame_index >= warmup:
            frame_load_ms.append((frame_load_end - frame_load_start) * 1000.0)
            preprocess_ms.append((preprocess_end - preprocess_start) * 1000.0)
            inference_ms.append((infer_end - infer_start) * 1000.0)
            postprocess_ms.append((post_end - post_start) * 1000.0)
            total_ms.append((post_end - preprocess_start) * 1000.0)

    return (
        iterations,
        frame_load_ms,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
        total_ms,
    )


def run_video_benchmark(
    model: DetectorModel,
    video_path: Path,
    warmup: int,
    max_frames: int | None,
) -> tuple[int, int, list[float], list[float], list[float], list[float], list[float]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video mode requires opencv-python. Install it in your environment."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    seen_frames = 0
    measured_frames = 0
    frame_load_ms = []
    preprocess_ms = []
    inference_ms = []
    postprocess_ms = []
    total_ms = []

    while True:
        if max_frames is not None and measured_frames >= max_frames:
            break

        frame_load_start = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_load_end = time.perf_counter()

        preprocess_start = frame_load_end
        input_tensor, target_sizes = model.preprocess_rgb_frame(frame_rgb)
        preprocess_end = time.perf_counter()

        infer_start = preprocess_end
        outputs = model.forward(input_tensor)
        infer_end = time.perf_counter()

        post_start = infer_end
        _ = model.postprocess(outputs=outputs, target_sizes=target_sizes)
        post_end = time.perf_counter()

        if seen_frames >= warmup:
            measured_frames += 1
            frame_load_ms.append((frame_load_end - frame_load_start) * 1000.0)
            preprocess_ms.append((preprocess_end - preprocess_start) * 1000.0)
            inference_ms.append((infer_end - infer_start) * 1000.0)
            postprocess_ms.append((post_end - post_start) * 1000.0)
            total_ms.append((post_end - preprocess_start) * 1000.0)

        seen_frames += 1

    cap.release()
    return (
        seen_frames,
        measured_frames,
        frame_load_ms,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
        total_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark model inference pipeline speed."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="rfdetr",
        choices=available_models(),
        help="Backend implementation name (compatibility option).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        choices=list_model_ids(),
        help="Registry model ID (preferred), e.g. rfdetr-m or yolox-s.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to local ONNX model. Overrides registry download/default paths.",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default=None,
        help="Model input size as HxW (for example: 640x640).",
    )
    parser.add_argument("--mode", choices=["dummy", "video"], default="dummy")
    parser.add_argument(
        "--video", type=Path, help="Path to input video (required for --mode video)."
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup frames excluded from stats."
    )
    parser.add_argument(
        "--iterations", type=int, default=200, help="Measured frames for dummy mode."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max measured frames for video mode.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Postprocess score threshold."
    )
    parser.add_argument(
        "--num-select", type=int, default=300, help="Top-K candidates in postprocess."
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Class IDs to keep. Omit to use model-specific defaults.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers.",
    )
    parser.add_argument(
        "--dummy-height", type=int, default=720, help="Dummy frame height."
    )
    parser.add_argument(
        "--dummy-width", type=int, default=1280, help="Dummy frame width."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for dummy frames."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable model auto-download from registry.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Custom model cache directory for downloaded artifacts.",
    )
    args = parser.parse_args()

    if args.mode == "video" and args.video is None:
        raise ValueError("--video is required when --mode video is selected.")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Use --warmup >= 0 and --iterations > 0.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0 when provided.")

    providers = parse_required_providers(args.providers)
    available_providers = set(ort.get_available_providers())
    missing_providers = [
        provider for provider in providers if provider not in available_providers
    ]
    if missing_providers:
        raise ValueError(
            f"Requested providers not available: {missing_providers}. "
            f"Available providers: {sorted(available_providers)}"
        )

    model_load_start = time.perf_counter()
    detector = Detector(
        model=args.model_id or args.model_name,
        model_path=args.model,
        input_size=parse_optional_input_size(args.input_size),
        providers=providers,
        threshold=args.threshold,
        num_select=args.num_select,
        class_ids=args.classes,
        auto_download=not args.no_download,
        cache_dir=args.cache_dir,
        show_download_progress=True,
    )
    model = detector.backend
    model_load_end = time.perf_counter()
    model_load_ms = (model_load_end - model_load_start) * 1000.0
    param_count, flops, flops_note = inspect_model_complexity(model.model_path)
    session = getattr(model, "session", None)
    active_providers = session.get_providers() if session is not None else None

    if args.mode == "dummy":
        measured_frames, frame_load_ms, preprocess_ms, infer_ms, post_ms, total_ms = (
            run_dummy_benchmark(
                model=model,
                warmup=args.warmup,
                iterations=args.iterations,
                dummy_height=args.dummy_height,
                dummy_width=args.dummy_width,
                seed=args.seed,
            )
        )
        seen_frames = args.warmup + measured_frames
    else:
        (
            seen_frames,
            measured_frames,
            frame_load_ms,
            preprocess_ms,
            infer_ms,
            post_ms,
            total_ms,
        ) = run_video_benchmark(
            model=model,
            video_path=args.video,
            warmup=args.warmup,
            max_frames=args.max_frames,
        )

    if measured_frames == 0:
        raise RuntimeError(
            "No measured frames were processed. Reduce --warmup or provide longer input."
        )

    mean_total = float(np.mean(total_ms))
    mean_infer = float(np.mean(infer_ms))
    fps_total = 1000.0 / mean_total
    fps_infer = 1000.0 / mean_infer

    print("=== Benchmark Result ===")
    print(f"mode                     {args.mode}")
    print(f"model name               {args.model_id or args.model_name}")
    if detector.info.model_id is not None:
        print(f"resolved model id        {detector.info.model_id}")
    print(f"model path               {model.model_path}")
    print(f"model input size         {model.input_size[0]}x{model.input_size[1]}")
    print(f"providers (requested)    {providers}")
    if active_providers is not None:
        print(f"providers (active)       {active_providers}")
    print(
        f"parameters               {_format_count(param_count)}"
        + (f" ({param_count:,})" if param_count is not None else "")
    )
    print(
        f"estimated FLOPs          {_format_count(flops)}"
        + (f" ({flops:,})" if flops is not None else "")
    )
    if flops_note:
        print(f"flops note               {flops_note}")
    print(f"one-time model load      {model_load_ms:.3f} ms")
    print(f"frames seen              {seen_frames}")
    print(f"warmup frames            {args.warmup}")
    print(f"measured frames          {measured_frames}")
    print_stage_stats("frame load/gen", frame_load_ms)
    print_stage_stats("preprocess", preprocess_ms)
    print_stage_stats(
        "load+preprocess", [x + y for x, y in zip(frame_load_ms, preprocess_ms)]
    )
    print_stage_stats("inference", infer_ms)
    print_stage_stats("postprocess", post_ms)
    print_stage_stats("total no-load/frame", total_ms)
    print(f"pipeline FPS (no load)   {fps_total:.2f}")
    print(f"inference-only FPS       {fps_infer:.2f}")


if __name__ == "__main__":
    main()
