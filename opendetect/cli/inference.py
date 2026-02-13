from __future__ import annotations

import argparse
from pathlib import Path

from opendetect import Detector
from opendetect._cli_utils import parse_optional_input_size, parse_providers
from opendetect.models import available_models
from opendetect.registry import list_model_ids


def resolve_output_path(output: Path | None, is_video: bool) -> Path:
    if output is not None:
        return output
    return Path.cwd() / ("output.mp4" if is_video else "output.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run object detection on an image or video."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Path to input image.")
    input_group.add_argument("--video", type=Path, help="Path to input video.")

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
        "--providers",
        type=str,
        default=None,
        help="Comma-separated ORT providers, e.g. CUDAExecutionProvider,CPUExecutionProvider.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write visualization (image or video based on input mode).",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--num-select", type=int, default=300)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process in video mode.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Class IDs to keep after thresholding. Omit for model defaults.",
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

    detector = Detector(
        model=args.model_id or args.model_name,
        model_path=args.model,
        input_size=parse_optional_input_size(args.input_size),
        providers=parse_providers(args.providers),
        threshold=args.threshold,
        num_select=args.num_select,
        class_ids=args.classes,
        auto_download=not args.no_download,
        cache_dir=args.cache_dir,
        show_download_progress=True,
    )

    output_path = resolve_output_path(args.output, is_video=args.video is not None)
    if args.video is not None:
        frame_count = detector.infer_video_file(
            args.video,
            output_path=output_path,
            max_frames=args.max_frames,
        )
        print(f"Visualization saved to {output_path} ({frame_count} frames)")
        return

    detector.infer_image_file(args.image, output_path=output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
