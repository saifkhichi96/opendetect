from __future__ import annotations

import argparse
from pathlib import Path

from opendetect import Detector
from opendetect._cli_utils import (
    parse_class_names,
    parse_optional_input_size,
)
from opendetect.models import available_models
from opendetect.registry import list_model_ids


def resolve_output_path(output: Path | None, is_video: bool) -> Path:
    if output is not None:
        return output
    return Path.cwd() / ("output.mp4" if is_video else "output.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run object detection on an image, video, or live camera."
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--image", type=Path, help="Path to input image.")
    input_group.add_argument("--video", type=Path, help="Path to input video.")
    input_group.add_argument("--camera", type=int, help="OpenCV camera index, e.g. 0.")
    parser.add_argument(
        "--tracker",
        choices=("none", "bytetrack"),
        default="none",
        help="Associate detections across video or camera frames.",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open a preview window (enabled by default for cameras).",
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
        help="Model input size as HxW (for example: 576x576).",
    )
    parser.add_argument(
        "--no-hardware-acceleration",
        action="store_true",
        help="Disable hardware acceleration and force CPUExecutionProvider.",
    )
    parser.add_argument(
        "--tensor-rt",
        action="store_true",
        help="Enable TensorRT providers when available.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision (FP16) when supported by the model and hardware. Can improve performance on compatible GPUs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write visualization (image or video based on input mode).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Detection confidence threshold (default: 0.3, or 0.1 with ByteTrack).",
    )
    parser.add_argument("--num-select", type=int, default=300)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process from a video or camera.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="0-based class IDs to keep (shared across models). Omit to keep all classes.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=None,
        help="Class names to keep (case-insensitive). Accepts space- or comma-separated names.",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List available classes for the selected model and exit.",
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

    if not args.list_classes and all(
        source is None for source in (args.image, args.video, args.camera)
    ):
        parser.error(
            "one of --image, --video, or --camera is required unless --list-classes is set"
        )
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be a nonnegative device index")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.image is not None and (args.tracker != "none" or args.preview):
        parser.error("tracking and preview require --video or --camera")

    tracker = None
    if args.tracker == "bytetrack" and not args.list_classes:
        try:
            from opendetect.trackers import ByteTracker
        except ModuleNotFoundError as exc:
            if exc.name != "scipy":
                raise
            parser.error('ByteTrack requires SciPy. Install "opendetect[tracking]".')
        tracker = ByteTracker()

    detector = Detector(
        model=args.model_id or args.model_name,
        model_path=args.model,
        input_size=parse_optional_input_size(args.input_size),
        hardware_acceleration=not args.no_hardware_acceleration,
        tensor_rt=args.tensor_rt,
        mixed_precision=args.mixed_precision,
        threshold=args.threshold
        if args.threshold is not None
        else (0.1 if tracker else 0.3),
        num_select=args.num_select,
        class_ids=None,
        auto_download=not args.no_download,
        cache_dir=args.cache_dir,
        show_download_progress=True,
    )

    if args.list_classes:
        classes = detector.list_classes()
        if not classes:
            print("No class names available for this model.")
            return
        for class_id, class_name in classes:
            print(f"{class_id:3d}: {class_name}")
        return

    selected_ids = list(args.classes or [])
    selected_names = parse_class_names(args.class_names)
    try:
        if selected_names:
            selected_ids.extend(detector.resolve_class_ids_from_names(selected_names))
        if tracker is not None:
            if not selected_ids:
                selected_ids = detector.resolve_class_ids_from_names(["person"])
            if len(set(selected_ids)) != 1:
                parser.error(
                    "ByteTrack requires a single class. Select one with --classes or --class-names."
                )
        detector.set_class_filter(selected_ids if selected_ids else None)
    except ValueError as exc:
        parser.error(str(exc))

    if args.camera is not None:
        frame_count = detector.infer_camera(
            args.camera,
            output_path=args.output,
            max_frames=args.max_frames,
            tracker=tracker,
            preview=args.preview is not False,
        )
        if args.output is None:
            print(f"Processed {frame_count} frames")
        else:
            print(f"Visualization saved to {args.output} ({frame_count} frames)")
        return

    output_path = resolve_output_path(args.output, is_video=args.video is not None)
    if args.video is not None:
        frame_count = detector.infer_video_file(
            args.video,
            output_path=output_path,
            max_frames=args.max_frames,
            tracker=tracker,
            preview=bool(args.preview),
        )
        print(f"Visualization saved to {output_path} ({frame_count} frames)")
        return

    detector.infer_image_file(args.image, output_path=output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
