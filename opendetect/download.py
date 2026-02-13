from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .registry import ModelSpec, get_model_spec, model_url


def default_cache_dir() -> Path:
    env_cache = os.getenv("OPENDETECT_CACHE_DIR")
    if env_cache:
        return Path(env_cache).expanduser().resolve()

    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "opendetect" / "checkpoints"

    return Path.home() / ".cache" / "opendetect" / "checkpoints"


def _print_progress(prefix: str, downloaded: int, total: int | None) -> None:
    if total is None or total <= 0:
        sys.stderr.write(f"\r{prefix}: {downloaded / (1024 * 1024):.1f} MiB")
    else:
        pct = min(100.0, 100.0 * downloaded / total)
        sys.stderr.write(
            f"\r{prefix}: {pct:6.2f}% ({downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MiB)"
        )
    sys.stderr.flush()


def download_url_to_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    show_progress: bool = True,
    timeout_sec: int = 60,
) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "opendetect"})
    response = urlopen(req, timeout=timeout_sec)
    total: int | None = None
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        total = int(content_length)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=destination.parent)
    hasher = hashlib.sha256() if expected_sha256 else None
    downloaded = 0

    try:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            tmp_file.write(chunk)
            downloaded += len(chunk)
            if hasher is not None:
                hasher.update(chunk)
            if show_progress:
                _print_progress(destination.name, downloaded, total)

        tmp_file.close()
        if show_progress:
            sys.stderr.write("\n")

        if hasher is not None:
            digest = hasher.hexdigest()
            if digest.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {destination.name}: expected {expected_sha256}, got {digest}"
                )

        Path(tmp_file.name).replace(destination)
    finally:
        tmp_file.close()
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

    return destination


def _extract_onnx_from_zip(zip_path: Path, output_dir: Path) -> Path:
    temp_extract_dir = output_dir / f".extract-{zip_path.stem}"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_extract_dir)

        candidates = sorted(temp_extract_dir.rglob("*.onnx"))
        if not candidates:
            raise RuntimeError(f"No .onnx file found in archive: {zip_path}")

        preferred = [path for path in candidates if path.name.endswith("end2end.onnx")]
        onnx_file = preferred[0] if preferred else candidates[0]

        destination = output_dir / (zip_path.stem + ".onnx")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(onnx_file), str(destination))
        return destination
    finally:
        shutil.rmtree(temp_extract_dir, ignore_errors=True)


def download_model(
    model: str | ModelSpec,
    *,
    cache_dir: Path | str | None = None,
    force: bool = False,
    show_progress: bool = True,
    timeout_sec: int = 60,
) -> Path:
    spec = model if isinstance(model, ModelSpec) else get_model_spec(model)
    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else default_cache_dir()
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    target_path = cache_root / spec.artifact_path
    if target_path.exists() and not force:
        return target_path

    url = model_url(spec)
    destination_name = Path(urlparse(url).path).name or spec.filename
    download_path = target_path.with_name(destination_name)

    download_url_to_file(
        url,
        download_path,
        show_progress=show_progress,
        timeout_sec=timeout_sec,
    )

    final_path = download_path
    if download_path.suffix.lower() == ".zip":
        final_path = _extract_onnx_from_zip(download_path, download_path.parent)
        download_path.unlink(missing_ok=True)

    if final_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(final_path), str(target_path))

    return target_path
