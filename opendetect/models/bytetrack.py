from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._nms import multiclass_nms
from ._viz import class_name_for_id, color_for_class_id, draw_box_with_label
from .base import Detections, create_ort_session


class ByteTrackModel:
    model_name = "bytetrack"
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        input_size: tuple[int, int],
        model_path: Path | None = None,
        threshold: float = 0.3,
        num_select: int = 300,
        class_ids: list[int] | None = None,
        nms_threshold: float = 0.7,
        with_p6: bool = False,
        hardware_acceleration: bool = True,
        tensor_rt: bool = False,
        mixed_precision: bool = True,
    ) -> None:
        self.input_size = (int(input_size[0]), int(input_size[1]))
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError(f"Invalid input_size: {self.input_size}")

        self.model_path = Path(model_path) if model_path is not None else None
        if self.model_path is None:
            raise ValueError("Model path must be provided")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        self.threshold = float(threshold)
        self.num_select = int(num_select)
        self.nms_threshold = float(nms_threshold)
        self.with_p6 = bool(with_p6)

        if class_ids is None:
            class_ids = self.default_class_ids()
        self.class_ids = class_ids if class_ids else None
        self.class_ids_np = np.asarray(self.class_ids, dtype=np.int64) if self.class_ids is not None else None

        self.session = create_ort_session(
            self.model_path,
            hardware_acceleration=hardware_acceleration,
            tensor_rt=tensor_rt,
            mixed_precision=mixed_precision,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = self.default_class_names()

    @classmethod
    def default_class_ids(cls) -> list[int] | None:
        return [0]

    @classmethod
    def default_class_names(cls) -> list[str] | None:
        return ["person"]

    @classmethod
    def default_input_size(cls) -> tuple[int, int]:
        return (608, 1088)

    @staticmethod
    def _decode_yolox(
        outputs: np.ndarray,
        input_size: tuple[int, int],
        *,
        with_p6: bool,
    ) -> np.ndarray:
        strides = [8, 16, 32, 64] if with_p6 else [8, 16, 32]
        grids = []
        expanded_strides = []

        for stride in strides:
            hsize = input_size[0] // stride
            wsize = input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride, dtype=np.float32))

        grids = np.concatenate(grids, axis=1)
        expanded_strides = np.concatenate(expanded_strides, axis=1)

        decoded = outputs.copy()
        decoded[..., :2] = (decoded[..., :2] + grids) * expanded_strides
        decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * expanded_strides
        return decoded

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return boxes_xyxy

    @staticmethod
    def _empty_detections() -> Detections:
        return {
            "xyxy": np.empty((0, 4), dtype=np.float32),
            "confidence": np.empty((0,), dtype=np.float32),
            "class_id": np.empty((0,), dtype=np.int64),
        }

    def _finalize(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        original_h: float,
        original_w: float,
    ) -> Detections:
        if boxes.size == 0 or scores.size == 0:
            return self._empty_detections()

        keep = scores > self.threshold
        if self.class_ids_np is not None:
            keep &= np.isin(class_ids, self.class_ids_np)

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if boxes.size == 0:
            return self._empty_detections()

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(0.0, original_w - 1))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(0.0, original_h - 1))

        if self.num_select > 0 and scores.shape[0] > self.num_select:
            topk_idx = np.argsort(-scores)[: self.num_select]
            boxes = boxes[topk_idx]
            scores = scores[topk_idx]
            class_ids = class_ids[topk_idx]

        return {
            "xyxy": boxes.astype(np.float32),
            "confidence": scores.astype(np.float32),
            "class_id": class_ids.astype(np.int64),
        }

    def preprocess_rgb_frame(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected image_rgb shape [H, W, 3].")

        original_h, original_w = image_rgb.shape[:2]
        input_h, input_w = self.input_size

        ratio = min(input_h / original_h, input_w / original_w)
        resized_h = max(1, int(round(original_h * ratio)))
        resized_w = max(1, int(round(original_w * ratio)))

        resized = cv2.resize(
            image_rgb,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        padded = np.full((input_h, input_w, 3), 114.0, dtype=np.float32)
        padded[:resized_h, :resized_w] = resized

        padded /= 255.0
        padded -= self.imagenet_mean
        padded /= self.imagenet_std

        input_tensor = np.transpose(padded, (2, 0, 1)).astype(np.float32)[None, ...]
        target_sizes = np.array([[original_h, original_w, ratio]], dtype=np.float32)
        return input_tensor, target_sizes

    def forward(self, input_tensor: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(self.session.run(None, {self.input_name: input_tensor}))

    def postprocess(
        self,
        outputs: tuple[np.ndarray, ...],
        target_sizes: np.ndarray,
    ) -> list[Detections]:
        if len(outputs) == 0:
            raise ValueError("Model returned no outputs.")

        raw = np.asarray(outputs[0])
        if raw.ndim == 2:
            raw = raw[None, ...]
        if raw.ndim != 3:
            raise ValueError(f"Unsupported output shape: {raw.shape}")

        if target_sizes.ndim != 2 or target_sizes.shape[1] < 2:
            raise ValueError("Expected target_sizes shape [batch, >=2].")
        if target_sizes.shape[0] != raw.shape[0]:
            raise ValueError("Batch size mismatch between outputs and target_sizes.")

        results: list[Detections] = []
        for batch_idx in range(raw.shape[0]):
            prediction = raw[batch_idx]
            original_h = float(target_sizes[batch_idx, 0])
            original_w = float(target_sizes[batch_idx, 1])
            ratio = (
                float(target_sizes[batch_idx, 2])
                if target_sizes.shape[1] >= 3
                else min(
                    self.input_size[0] / max(original_h, 1.0),
                    self.input_size[1] / max(original_w, 1.0),
                )
            )

            if prediction.shape[-1] == 5:
                boxes = prediction[:, :4] / max(ratio, 1e-6)
                scores = prediction[:, 4]
                class_ids = np.zeros((len(boxes),), dtype=np.int64)
                results.append(
                    self._finalize(
                        boxes=boxes,
                        scores=scores,
                        class_ids=class_ids,
                        original_h=original_h,
                        original_w=original_w,
                    )
                )
                continue

            decoded = self._decode_yolox(
                prediction[None, ...],
                self.input_size,
                with_p6=self.with_p6,
            )[0]
            boxes = decoded[:, :4]
            boxes_xyxy = self._cxcywh_to_xyxy(boxes) / max(ratio, 1e-6)
            if decoded.shape[1] <= 5:
                results.append(self._empty_detections())
                continue

            score_matrix = decoded[:, 4:5] * decoded[:, 5:]
            dets, _ = multiclass_nms(
                boxes_xyxy,
                score_matrix,
                nms_thr=self.nms_threshold,
                score_thr=self.threshold,
            )
            if dets is None or dets.size == 0:
                results.append(self._empty_detections())
                continue

            results.append(
                self._finalize(
                    boxes=dets[:, :4],
                    scores=dets[:, 4],
                    class_ids=dets[:, 5].astype(np.int64),
                    original_h=original_h,
                    original_w=original_w,
                )
            )

        return results

    def predict_rgb_frame(self, image_rgb: np.ndarray) -> Detections:
        input_tensor, target_sizes = self.preprocess_rgb_frame(image_rgb)
        outputs = self.forward(input_tensor)
        return self.postprocess(outputs, target_sizes)[0]

    def draw_detections_on_bgr_frame(
        self,
        frame_bgr: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected frame_bgr shape [H, W, 3].")

        for class_id, box, score in zip(
            detections["class_id"],
            detections["xyxy"],
            detections["confidence"],
        ):
            class_id_int = int(class_id)
            x1, y1, x2, y2 = box
            pt1 = (int(round(float(x1))), int(round(float(y1))))
            pt2 = (int(round(float(x2))), int(round(float(y2))))
            label_name = class_name_for_id(class_id_int, self.class_names)
            label = f"{label_name} {float(score):.2f}"
            draw_box_with_label(
                frame_bgr,
                pt1,
                pt2,
                color_for_class_id(class_id_int),
                label,
            )

        return frame_bgr
