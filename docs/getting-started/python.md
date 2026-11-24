# Python Usage

## Quickstart

```python
import cv2
from opendetect import Detector

detector = Detector(model="rfdetr-m")

image = cv2.imread("input.jpg")
detections = detector.predict(image, color="bgr")
annotated = detector.annotate(image, detections, color="bgr")

cv2.imwrite("output.jpg", annotated)
```

## File Helpers

```python
from opendetect import Detector

detector = Detector(model="yolox-s")
detector.infer_image_file("input.jpg", output_path="output.jpg")
detector.infer_video_file("input.mp4", output_path="output.mp4", max_frames=300)
```

## Choosing Models

You can load by registry model ID (recommended) or by implementation name.

- registry IDs: `rfdetr-n`, `rfdetr-s`, `rfdetr-m`, `rfdetr-l`, `yolox-tiny`, `yolox-s`, `yolox-m`, `yolox-l`, `yolox-x`, `rtmdet-n-hand`, `rtmdet-s`, `rtmdet-m`, `rtmdet-l`
- implementation names: `rfdetr`, `yolox`, `rtmdet` (maps to project defaults)

## Instance segmentation

Select an RF-DETR segmentation variant by model ID. The same `predict()`
method returns boxes, confidence scores, class IDs, and one boolean mask per
detection at the original image resolution.

```python
detector = Detector(model="rfdetr-seg-n", class_ids=[0])
detections = detector.predict(image, color="bgr")
masks = detections["masks"]  # shape (N, image_height, image_width)
annotated = detector.annotate(image, detections, color="bgr")
```

The mask rows match the box, score, and class ID rows. Box-only models
continue to return `xyxy`, `confidence`, and `class_id`.

## Track identities across frames

Install `opendetect[tracking]` to use identity tracking. Create one tracker per video or camera and call it once per frame, including frames with no detections.

```python
from opendetect import Detector
from opendetect.trackers import ByteTracker

detector = Detector(model="yolox-s", class_ids=[0], threshold=0.1)
tracker = ByteTracker(high_thr=0.5, low_thr=0.1)

for frame in frames:
    detections = detector.predict(frame, color="bgr")
    for track_id, xyxy, score in tracker.update(
        detections["xyxy"], detections["confidence"]
    ):
        print(track_id, xyxy, score)
```

Use a low detector threshold so ByteTrack can use weak detections to continue existing tracks. Only high-confidence detections can start a track. Call `reset()` when reusing the tracker for a different stream. Custom trackers can implement the `opendetect.trackers.Tracker` interface.

## Runtime Controls

```python
from opendetect import Detector

detector = Detector(
    model="rfdetr-l",
    hardware_acceleration=True,
    tensor_rt=True,
    mixed_precision=True,
)
```

- `hardware_acceleration=False` forces CPU execution provider.
- `tensor_rt=True` enables TensorRT provider selection when available.
- `mixed_precision=True` enables FP16/TF32-capable settings where supported.

## Class Filtering

By class IDs:

```python
detector.set_class_filter([0, 1, 2])
```

By class names:

```python
ids = detector.resolve_class_ids_from_names(["person", "car"])
detector.set_class_filter(ids)
```

Class IDs are normalized to 0-based foreground IDs across model families.

## Model Metadata

```python
from opendetect import list_models

for spec in list_models():
    print(spec.model_id, spec.implementation, spec.input_size)
```
