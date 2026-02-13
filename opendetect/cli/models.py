from __future__ import annotations

import argparse
from pathlib import Path

from opendetect import download_model
from opendetect.download import default_cache_dir
from opendetect.registry import get_model_spec, list_models, model_url


def _cmd_list() -> None:
    print("model_id       implementation  input_size  artifact")
    for spec in list_models():
        h, w = spec.input_size
        print(
            f"{spec.model_id:<14} {spec.implementation:<15} {h}x{w:<8} {spec.artifact_path}"
        )


def _cmd_info(model_id: str) -> None:
    spec = get_model_spec(model_id)
    print(f"model id        {spec.model_id}")
    print(f"implementation  {spec.implementation}")
    print(f"input size      {spec.input_size[0]}x{spec.input_size[1]}")
    print(f"artifact path   {spec.artifact_path}")
    print(f"download url    {model_url(spec)}")
    if spec.aliases:
        print(f"aliases         {', '.join(spec.aliases)}")
    if spec.description:
        print(f"description     {spec.description}")


def _cmd_download(model_id: str, cache_dir: Path | None, force: bool) -> None:
    model_path = download_model(
        model_id,
        cache_dir=cache_dir,
        force=force,
        show_progress=True,
    )
    print(f"Downloaded model: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and download model artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available model IDs.")

    info_parser = subparsers.add_parser("info", help="Show model metadata.")
    info_parser.add_argument("model_id", type=str)

    download_parser = subparsers.add_parser("download", help="Download a model.")
    download_parser.add_argument("model_id", type=str)
    download_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Destination cache directory. Default: {default_cache_dir()}",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached.",
    )

    args = parser.parse_args()

    if args.command == "list":
        _cmd_list()
        return

    if args.command == "info":
        _cmd_info(args.model_id)
        return

    if args.command == "download":
        _cmd_download(args.model_id, args.cache_dir, args.force)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
