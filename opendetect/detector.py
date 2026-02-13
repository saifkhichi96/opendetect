from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .download import download_model
from .models import available_models, create_model, default_input_size
from .models.base import DetectorModel
from .registry import ModelSpec, get_model_spec
from .types import Detections, LoadedModelInfo


class Detector:
    def __init__(
        self,
        model: str = "rfdetr-m",
        *,
        model_path: str | Path | None = None,
        input_size: tuple[int, int] | None = None,
        providers: list[str] | None = None,
        threshold: float = 0.3,
        num_select: int = 300,
        class_ids: list[int] | None = None,
        auto_download: bool = True,
        cache_dir: str | Path | None = None,
        show_download_progress: bool = False,
    ) -> None:
        spec = self._try_get_spec(model)

        if spec is not None:
            implementation = spec.implementation
        else:
            implementation = model.strip().lower()
            if implementation not in available_models():
                raise ValueError(
                    f"Unknown model '{model}'. Use one of registry IDs or implementations: {available_models()}"
                )

        resolved_input_size = (
            input_size
            if input_size is not None
            else (
                spec.input_size
                if spec is not None
                else default_input_size(implementation)
            )
        )

        resolved_model_path: Path | None
        source: str
        if model_path is not None:
            resolved_model_path = Path(model_path)
            source = "local-path"
        elif spec is not None and auto_download:
            resolved_model_path = download_model(
                spec,
                cache_dir=Path(cache_dir) if cache_dir is not None else None,
                show_progress=show_download_progress,
            )
            source = "registry-download"
        else:
            resolved_model_path = None
            source = "backend-default"

        self._backend: DetectorModel = create_model(
            implementation,
            input_size=resolved_input_size,
            model_path=resolved_model_path,
            threshold=threshold,
            num_select=num_select,
            class_ids=class_ids,
            providers=providers,
        )

        resolved_id = spec.model_id if spec is not None else None
        self.info = LoadedModelInfo(
            model_id=resolved_id,
            implementation=implementation,
            input_size=tuple(self._backend.input_size),
            model_path=self._backend.model_path,
            source=source,
        )

    @property
    def backend(self) -> DetectorModel:
        return self._backend

    @staticmethod
    def _try_get_spec(model: str) -> ModelSpec | None:
        try:
            return get_model_spec(model)
        except ValueError:
            return None

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected image shape [H, W, 3].")

    def _to_rgb(self, image: np.ndarray, color: str) -> np.ndarray:
        self._validate_image(image)
        mode = color.lower()
        if mode == "rgb":
            return image
        if mode == "bgr":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raise ValueError("color must be 'rgb' or 'bgr'.")

    def predict(self, image: np.ndarray, *, color: str = "bgr") -> Detections:
        image_rgb = self._to_rgb(image, color)
        return self._backend.predict_rgb_frame(image_rgb)

    def annotate(
        self,
        image: np.ndarray,
        detections: Detections | None = None,
        *,
        color: str = "bgr",
    ) -> np.ndarray:
        mode = color.lower()
        if detections is None:
            detections = self.predict(image, color=color)

        if mode == "bgr":
            frame_bgr = image.copy()
            return self._backend.draw_detections_on_bgr_frame(frame_bgr, detections)

        if mode == "rgb":
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            annotated_bgr = self._backend.draw_detections_on_bgr_frame(
                frame_bgr, detections
            )
            return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        raise ValueError("color must be 'rgb' or 'bgr'.")

    def predict_and_annotate(
        self, image: np.ndarray, *, color: str = "bgr"
    ) -> tuple[Detections, np.ndarray]:
        detections = self.predict(image, color=color)
        annotated = self.annotate(image, detections=detections, color=color)
        return detections, annotated

    def infer_image_file(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> tuple[Detections, np.ndarray]:
        image_path = Path(image_path)
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Unable to open image: {image_path}")

        detections, annotated = self.predict_and_annotate(frame_bgr, color="bgr")

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), annotated):
                raise RuntimeError(f"Unable to write image: {out_path}")

        return detections, annotated

    def infer_video_file(
        self,
        video_path: str | Path,
        *,
        output_path: str | Path,
        max_frames: int | None = None,
    ) -> int:
        video_path = Path(video_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if suffix == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open video writer for: {output_path}")

        frames_processed = 0
        try:
            while True:
                if max_frames is not None and frames_processed >= max_frames:
                    break

                ok, frame_bgr = cap.read()
                if not ok:
                    break

                _, annotated = self.predict_and_annotate(frame_bgr, color="bgr")
                writer.write(annotated)
                frames_processed += 1
        finally:
            cap.release()
            writer.release()

        return frames_processed


def load_detector(
    model: str = "rfdetr-m",
    **kwargs,
) -> Detector:
    return Detector(model=model, **kwargs)
