from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .base import Detections
from ._viz import class_name_for_id, color_for_class_id, draw_box_with_label


class RFDETRModel:
    model_name = "rfdetr"
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixel_scale = np.float32(1.0 / 255.0)

    def __init__(
        self,
        input_size: tuple[int, int],
        model_path: Path | None = None,
        threshold: float = 0.3,
        num_select: int = 300,
        class_ids: list[int] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.input_size = (int(input_size[0]), int(input_size[1]))
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError(f"Invalid input_size: {self.input_size}")

        self.model_path = (
            Path(model_path) if model_path is not None else None
        )
        if self.model_path is None:
            raise ValueError("Model path must be provided")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        self.threshold = threshold
        self.num_select = num_select
        if class_ids is None:
            class_ids = self.default_class_ids()
        self.class_ids = class_ids if class_ids else None
        self.class_ids_np = (
            np.asarray(self.class_ids, dtype=np.int64)
            if self.class_ids is not None
            else None
        )

        session_kwargs: dict[str, object] = {}
        if providers is not None:
            session_kwargs["providers"] = providers
        self.session = ort.InferenceSession(str(self.model_path), **session_kwargs)
        self.input_name = self.session.get_inputs()[0].name
        self._imagenet_inv_std = (1.0 / self.imagenet_std).astype(np.float32)
        self.class_names = self._resolve_class_names()

    @classmethod
    def default_class_ids(cls) -> list[int] | None:
        return None

    @classmethod
    def default_class_names(cls) -> list[str] | None:
        from .yolox import YOLOXModel

        return list(YOLOXModel.coco_classes)

    @classmethod
    def default_input_size(cls) -> tuple[int, int]:
        return (576, 576)

    def _resolve_class_names(self) -> list[str] | None:
        base_names = self.default_class_names()
        if base_names is None:
            return None

        output_infos = self.session.get_outputs()
        if len(output_infos) < 2:
            return list(base_names)

        shape = output_infos[1].shape
        if len(shape) == 3 and isinstance(shape[2], int) and int(shape[2]) > 1:
            num_foreground = int(shape[2]) - 1
            if num_foreground <= len(base_names):
                return list(base_names[:num_foreground])
            extra = [
                f"class_{idx}" for idx in range(len(base_names), num_foreground)
            ]
            return list(base_names) + extra

        return list(base_names)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = np.clip(boxes[..., 2], a_min=0.0, a_max=None)
        h = np.clip(boxes[..., 3], a_min=0.0, a_max=None)
        return np.stack(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1
        )

    def preprocess_rgb_frame(
        self, image_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected image_rgb shape [H, W, 3].")

        original_h, original_w = image_rgb.shape[:2]
        resized = cv2.resize(
            image_rgb,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        image_array = resized.astype(np.float32)
        np.multiply(image_array, self.pixel_scale, out=image_array)
        np.subtract(image_array, self.imagenet_mean, out=image_array)
        np.multiply(image_array, self._imagenet_inv_std, out=image_array)
        input_tensor = np.transpose(image_array, (2, 0, 1))[None, ...]
        target_sizes = np.array([[original_h, original_w]], dtype=np.float32)
        return input_tensor, target_sizes

    def forward(self, input_tensor: np.ndarray) -> tuple[np.ndarray, ...]:
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return tuple(outputs)

    def postprocess(
        self, outputs: tuple[np.ndarray, ...], target_sizes: np.ndarray
    ) -> list[Detections]:
        out_bbox, out_logits = outputs

        if out_bbox.ndim != 3 or out_logits.ndim != 3:
            raise ValueError("Expected outputs with shape [batch, queries, ...].")
        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            raise ValueError("Expected target_sizes shape [batch, 2].")
        if (
            out_bbox.shape[0] != out_logits.shape[0]
            or out_logits.shape[0] != target_sizes.shape[0]
        ):
            raise ValueError("Batch size mismatch between outputs and target_sizes.")

        probs = self._sigmoid(out_logits)
        # RF-DETR exports include a background class at index 0.
        # Exclude it from ranking so class IDs match YOLOX-style 0-based foreground IDs.
        probs[..., 0] = -1.0
        batch_size, _, num_classes = probs.shape

        flat_probs = probs.reshape(batch_size, -1)
        num_topk = min(self.num_select, flat_probs.shape[1])
        if num_topk <= 0:
            return [
                {
                    "xyxy": np.empty((0, 4), dtype=np.float32),
                    "confidence": np.empty((0,), dtype=np.float32),
                    "class_id": np.empty((0,), dtype=np.int64),
                }
                for _ in range(batch_size)
            ]

        topk_unsorted_indices = np.argpartition(flat_probs, -num_topk, axis=1)[
            :, -num_topk:
        ]
        topk_unsorted_scores = np.take_along_axis(
            flat_probs, topk_unsorted_indices, axis=1
        )
        sort_order = np.argsort(-topk_unsorted_scores, axis=1)
        topk_indices = np.take_along_axis(topk_unsorted_indices, sort_order, axis=1)
        scores = np.take_along_axis(topk_unsorted_scores, sort_order, axis=1)

        topk_boxes = topk_indices // num_classes
        labels = topk_indices % num_classes
        mapped_labels = labels - 1

        boxes = self._box_cxcywh_to_xyxy(out_bbox)
        batch_indices = np.arange(batch_size)[:, None]
        boxes = boxes[batch_indices, topk_boxes]

        img_h = target_sizes[:, 0]
        img_w = target_sizes[:, 1]
        scale_factors = np.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_factors[:, None, :]

        detections = []
        for i in range(batch_size):
            keep = scores[i] > self.threshold
            keep &= labels[i] > 0
            if self.class_ids_np is not None:
                keep &= np.isin(mapped_labels[i], self.class_ids_np)

            detections.append(
                {
                    "xyxy": boxes[i][keep].astype(np.float32),
                    "confidence": scores[i][keep].astype(np.float32),
                    "class_id": mapped_labels[i][keep].astype(np.int64),
                }
            )

        return detections

    def predict_rgb_frame(self, image_rgb: np.ndarray) -> Detections:
        input_tensor, target_sizes = self.preprocess_rgb_frame(image_rgb)
        outputs = self.forward(input_tensor)
        return self.postprocess(outputs, target_sizes)[0]

    def draw_detections_on_bgr_frame(
        self, frame_bgr: np.ndarray, detections: Detections
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
