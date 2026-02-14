from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._nms import multiclass_nms
from .base import Detections, create_ort_session
from ._viz import class_name_for_id, color_for_class_id, draw_box_with_label


class YOLOXModel:
    model_name = "yolox"
    coco_classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        input_size: tuple[int, int],
        model_path: Path | None = None,
        threshold: float = 0.3,
        num_select: int = 300,
        class_ids: list[int] | None = None,
        nms_threshold: float = 0.45,
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
        if class_ids is None:
            class_ids = self.default_class_ids()
        self.class_ids = class_ids if class_ids else None
        self.class_ids_np = (
            np.asarray(self.class_ids, dtype=np.int64)
            if self.class_ids is not None
            else None
        )

        self.session = create_ort_session(
            self.model_path,
            hardware_acceleration=hardware_acceleration,
            tensor_rt=tensor_rt,
            mixed_precision=mixed_precision,
            # YOLOX works better with NeuralNetwork instead of MLProgram on CoreML EP
            overrides=dict(
                CoreMLExecutionProvider=dict(
                    ModelFormat="NeuralNetwork",
                )
            ),
        )
        self.input_name = self.session.get_inputs()[0].name
        self._decode_grids, self._decode_strides = self._build_decode_grids()
        self.class_names = self._resolve_class_names()

    @classmethod
    def default_class_ids(cls) -> list[int] | None:
        return None

    @classmethod
    def default_class_names(cls) -> list[str] | None:
        return list(cls.coco_classes)

    @classmethod
    def default_input_size(cls) -> tuple[int, int]:
        return (640, 640)

    def _resolve_class_names(self) -> list[str] | None:
        base_names = self.default_class_names()
        if base_names is None:
            return None

        # Raw YOLOX exports commonly produce [N, num_preds, 5 + num_classes].
        for output in self.session.get_outputs():
            shape = output.shape
            if len(shape) == 3 and isinstance(shape[2], int) and int(shape[2]) > 5:
                num_classes = int(shape[2]) - 5
                if num_classes <= len(base_names):
                    return list(base_names[:num_classes])
                extra = [f"class_{idx}" for idx in range(len(base_names), num_classes)]
                return list(base_names) + extra

        return list(base_names)

    def _build_decode_grids(self) -> tuple[np.ndarray, np.ndarray]:
        strides = (8, 16, 32)
        grids = []
        expanded_strides = []
        for stride in strides:
            hsize = self.input_size[0] // stride
            wsize = self.input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32)
            grids.append(grid)
            expanded_strides.append(
                np.full((1, grid.shape[1], 1), stride, dtype=np.float32)
            )
        return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return boxes_xyxy

    def preprocess_rgb_frame(
        self, image_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected image_rgb shape [H, W, 3].")

        original_h, original_w = image_rgb.shape[:2]
        input_h, input_w = self.input_size
        ratio = min(input_h / original_h, input_w / original_w)
        resized_h = max(1, int(round(original_h * ratio)))
        resized_w = max(1, int(round(original_w * ratio)))

        resized = cv2.resize(
            image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[:resized_h, :resized_w] = resized

        input_tensor = np.transpose(padded, (2, 0, 1)).astype(np.float32)[None, ...]
        target_sizes = np.array([[original_h, original_w, ratio]], dtype=np.float32)
        return input_tensor, target_sizes

    def forward(self, input_tensor: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(self.session.run(None, {self.input_name: input_tensor}))

    def _decode_if_needed(self, predictions: np.ndarray) -> np.ndarray:
        decoded = predictions.copy()
        if decoded.shape[1] == self._decode_grids.shape[1]:
            decoded[..., :2] = (
                decoded[..., :2] + self._decode_grids
            ) * self._decode_strides
            decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * self._decode_strides
        return decoded

    def _finalize(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        original_h: float,
        original_w: float,
    ) -> Detections:
        if boxes.size == 0 or scores.size == 0:
            return {
                "xyxy": np.empty((0, 4), dtype=np.float32),
                "confidence": np.empty((0,), dtype=np.float32),
                "class_id": np.empty((0,), dtype=np.int64),
            }

        keep = scores > self.threshold
        if self.class_ids_np is not None:
            keep &= np.isin(class_ids, self.class_ids_np)

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if boxes.size == 0:
            return {
                "xyxy": np.empty((0, 4), dtype=np.float32),
                "confidence": np.empty((0,), dtype=np.float32),
                "class_id": np.empty((0,), dtype=np.int64),
            }

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

    def _postprocess_raw_predictions(
        self,
        predictions: np.ndarray,
        ratio: float,
        original_h: float,
        original_w: float,
    ) -> Detections:
        decoded = self._decode_if_needed(predictions[None, ...])[0]
        boxes = decoded[:, :4]
        boxes_xyxy = self._cxcywh_to_xyxy(boxes) / max(ratio, 1e-6)
        if decoded.shape[1] <= 5:
            raise ValueError(f"Unsupported raw YOLOX output shape: {decoded.shape}")

        scores = decoded[:, 4:5] * decoded[:, 5:]
        dets, _ = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.nms_threshold,
            score_thr=self.threshold,
        )
        if dets is None or dets.size == 0:
            return {
                "xyxy": np.empty((0, 4), dtype=np.float32),
                "confidence": np.empty((0,), dtype=np.float32),
                "class_id": np.empty((0,), dtype=np.int64),
            }

        return self._finalize(
            boxes=dets[:, :4],
            scores=dets[:, 4],
            class_ids=dets[:, 5].astype(np.int64),
            original_h=original_h,
            original_w=original_w,
        )

    def postprocess(
        self, outputs: tuple[np.ndarray, ...], target_sizes: np.ndarray
    ) -> list[Detections]:
        if len(outputs) == 0:
            raise ValueError("Model returned no outputs.")

        if target_sizes.ndim != 2 or target_sizes.shape[1] < 2:
            raise ValueError("Expected target_sizes shape [batch, >=2].")

        if len(outputs) >= 3:
            boxes_out = np.asarray(outputs[0])
            scores_out = np.asarray(outputs[1])
            classes_out = np.asarray(outputs[2])
            if boxes_out.ndim == 2:
                boxes_out = boxes_out[None, ...]
            if scores_out.ndim == 1:
                scores_out = scores_out[None, ...]
            if classes_out.ndim == 1:
                classes_out = classes_out[None, ...]

            if (
                boxes_out.ndim == 3
                and boxes_out.shape[-1] == 4
                and scores_out.ndim == 2
                and classes_out.ndim == 2
                and boxes_out.shape[0] == scores_out.shape[0] == classes_out.shape[0]
            ):
                if target_sizes.shape[0] != boxes_out.shape[0]:
                    raise ValueError(
                        "Batch size mismatch between outputs and target_sizes."
                    )

                results: list[Detections] = []
                for batch_idx in range(boxes_out.shape[0]):
                    original_h = float(target_sizes[batch_idx, 0])
                    original_w = float(target_sizes[batch_idx, 1])
                    if target_sizes.shape[1] >= 3:
                        ratio = float(target_sizes[batch_idx, 2])
                    else:
                        ratio = min(
                            self.input_size[0] / max(original_h, 1.0),
                            self.input_size[1] / max(original_w, 1.0),
                        )
                    results.append(
                        self._finalize(
                            boxes=boxes_out[batch_idx] / max(ratio, 1e-6),
                            scores=scores_out[batch_idx],
                            class_ids=classes_out[batch_idx].astype(np.int64),
                            original_h=original_h,
                            original_w=original_w,
                        )
                    )
                return results

        first_output = np.asarray(outputs[0])
        if first_output.ndim == 2:
            first_output = first_output[None, ...]
        if first_output.ndim != 3:
            raise ValueError(f"Unsupported output shape: {first_output.shape}")

        batch_size = first_output.shape[0]
        if target_sizes.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between outputs and target_sizes.")

        results: list[Detections] = []
        for batch_idx in range(batch_size):
            original_h = float(target_sizes[batch_idx, 0])
            original_w = float(target_sizes[batch_idx, 1])
            if target_sizes.shape[1] >= 3:
                ratio = float(target_sizes[batch_idx, 2])
            else:
                ratio = min(
                    self.input_size[0] / max(original_h, 1.0),
                    self.input_size[1] / max(original_w, 1.0),
                )

            preds = first_output[batch_idx]
            if preds.shape[-1] >= 7:
                detections = self._postprocess_raw_predictions(
                    preds, ratio, original_h, original_w
                )
            elif preds.shape[-1] == 6:
                boxes = preds[:, :4] / max(ratio, 1e-6)
                scores = preds[:, 4]
                class_ids = preds[:, 5].astype(np.int64)
                detections = self._finalize(
                    boxes, scores, class_ids, original_h, original_w
                )
            elif preds.shape[-1] == 5:
                boxes = preds[:, :4] / max(ratio, 1e-6)
                scores = preds[:, 4]
                class_ids = np.zeros_like(scores, dtype=np.int64)
                detections = self._finalize(
                    boxes, scores, class_ids, original_h, original_w
                )
            else:
                raise ValueError(f"Unsupported YOLOX output shape: {preds.shape}")

            results.append(detections)

        return results

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
