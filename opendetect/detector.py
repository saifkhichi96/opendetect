from __future__ import annotations

import difflib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .download import download_model
from .models import available_models, create_model, default_input_size
from .models._viz import draw_box_with_label, draw_instance_mask
from .models.base import DetectorModel
from .registry import ModelSpec, get_model_spec
from .types import Detections, LoadedModelInfo

if TYPE_CHECKING:
    from .trackers import Tracker


class Detector:
    def __init__(
        self,
        model: str = "rfdetr-m",
        *,
        model_path: str | Path | None = None,
        input_size: tuple[int, int] | None = None,
        hardware_acceleration: bool = True,
        tensor_rt: bool = False,
        mixed_precision: bool = False,
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
            hardware_acceleration=hardware_acceleration,
            tensor_rt=tensor_rt,
            mixed_precision=mixed_precision,
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

    @property
    def class_names(self) -> list[str] | None:
        names = getattr(self._backend, "class_names", None)
        if names is not None:
            return [str(name) for name in names]

        default_names_fn = getattr(self._backend, "default_class_names", None)
        if callable(default_names_fn):
            fallback = default_names_fn()
            if fallback is not None:
                return [str(name) for name in fallback]

        return None

    @staticmethod
    def _try_get_spec(model: str) -> ModelSpec | None:
        try:
            return get_model_spec(model)
        except ValueError:
            return None

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        normalized = name.strip().lower().replace("-", " ").replace("_", " ")
        normalized = " ".join(normalized.split())
        return normalized.replace(" ", "")

    def resolve_class_ids_from_names(self, names: list[str]) -> list[int]:
        class_names = self.class_names
        if class_names is None:
            raise ValueError("Class names are not available for this model.")

        normalized_to_id: dict[str, int] = {}
        for class_id, class_name in enumerate(class_names):
            normalized_to_id[self._normalize_class_name(class_name)] = class_id

        resolved: list[int] = []
        unknown: list[str] = []
        for name in names:
            key = self._normalize_class_name(name)
            class_id = normalized_to_id.get(key)
            if class_id is None:
                unknown.append(name)
                continue
            if class_id not in resolved:
                resolved.append(class_id)

        if unknown:
            known_names = list(class_names)
            hints = []
            for value in unknown:
                match = difflib.get_close_matches(value, known_names, n=1, cutoff=0.6)
                if match:
                    hints.append(f"{value!r} -> {match[0]!r}")
            hint_text = f" Suggestions: {', '.join(hints)}." if hints else ""
            raise ValueError(
                f"Unknown class name(s): {', '.join(repr(item) for item in unknown)}.{hint_text}"
            )

        return resolved

    def set_class_filter(self, class_ids: list[int] | None) -> list[int] | None:
        resolved = None
        if class_ids:
            resolved = []
            for class_id in class_ids:
                class_id_int = int(class_id)
                if class_id_int not in resolved:
                    resolved.append(class_id_int)

            class_names = self.class_names
            if class_names is not None:
                max_id = len(class_names) - 1
                invalid = [idx for idx in resolved if idx < 0 or idx > max_id]
                if invalid:
                    raise ValueError(
                        f"Class IDs out of range: {invalid}. Valid range: 0..{max_id}."
                    )

        # Both current backends support dynamic class filter updates.
        self._backend.class_ids = resolved if resolved else None
        self._backend.class_ids_np = (
            np.asarray(self._backend.class_ids, dtype=np.int64)
            if self._backend.class_ids is not None
            else None
        )
        return self._backend.class_ids

    def list_classes(self) -> list[tuple[int, str]]:
        class_names = self.class_names
        if class_names is None:
            return []
        return list(enumerate(class_names))

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
        tracker: Tracker | None = None,
        preview: bool = False,
    ) -> int:
        return self._infer_capture(
            str(video_path), Path(output_path), max_frames, tracker, preview
        )

    def infer_camera(
        self,
        camera: int,
        *,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
        tracker: Tracker | None = None,
        preview: bool = True,
    ) -> int:
        """Run inference on an OpenCV camera device."""
        output = Path(output_path) if output_path is not None else None
        return self._infer_capture(camera, output, max_frames, tracker, preview)

    def _annotate_tracks(self, frame_bgr: np.ndarray, tracker: Tracker) -> np.ndarray:
        detections = self.predict(frame_bgr, color="bgr")
        boxes, scores = detections["xyxy"], detections["confidence"]
        masks = detections.get("masks")
        update_with_indices = getattr(tracker, "update_with_indices", None)
        if masks is not None and update_with_indices is not None:
            tracks = update_with_indices(boxes, scores)
        else:
            tracks = [(track, None) for track in tracker.update(boxes, scores)]

        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]
        bounds = [width - 1, height - 1] * 2
        labels = []
        for (track_id, box, score), detection_index in tracks:
            # An odd multiplier permutes the 21-bit palette by track ID.
            color_code = track_id * 0x17A4AB
            color = tuple(96 + ((color_code >> shift) & 127) for shift in (0, 7, 14))
            if masks is not None and detection_index is not None:
                draw_instance_mask(annotated, masks[detection_index], color)
            x1, y1, x2, y2 = map(int, np.clip(np.rint(box), 0, bounds))
            labels.append(((x1, y1), (x2, y2), color, f"ID {track_id} {score:.2f}"))
        for label in labels:
            draw_box_with_label(annotated, *label)
        return annotated

    def _infer_capture(self, source, output_path, max_frames, tracker, preview):
        cap = cv2.VideoCapture(source)
        writer = None
        frames_processed = 0
        window = "OpenDetect"
        window_open = False
        try:
            if not cap.isOpened():
                kind = "camera" if isinstance(source, int) else "video"
                raise RuntimeError(f"Unable to open {kind}: {source}")
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            while max_frames is None or frames_processed < max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    if frames_processed == 0 and isinstance(source, int):
                        raise RuntimeError(f"No frames received from camera: {source}")
                    break

                if tracker is None:
                    _, annotated = self.predict_and_annotate(frame_bgr, color="bgr")
                else:
                    annotated = self._annotate_tracks(frame_bgr, tracker)
                if output_path is not None:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if not math.isfinite(fps) or fps <= 0:
                            fps = 30.0
                        codec = (
                            "MJPG" if output_path.suffix.lower() == ".avi" else "mp4v"
                        )
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*codec),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Unable to open video writer for: {output_path}"
                            )
                    writer.write(annotated)
                frames_processed += 1
                if preview:
                    cv2.imshow(window, annotated)
                    window_open = True
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or (
                        cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0
                    ):
                        break
        except KeyboardInterrupt:
            if not isinstance(source, int) and not preview:
                raise
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if window_open:
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    pass

        return frames_processed


def load_detector(
    model: str = "rfdetr-m",
    **kwargs,
) -> Detector:
    return Detector(model=model, **kwargs)
