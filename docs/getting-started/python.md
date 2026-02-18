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

- registry IDs: `rfdetr-n`, `rfdetr-s`, `rfdetr-m`, `rfdetr-l`, `yolox-tiny`, `yolox-s`, `yolox-m`, `yolox-l`, `yolox-x`, `bytetrack-s`, `bytetrack-m`, `bytetrack-l`
- implementation names: `rfdetr`, `yolox`, `bytetrack` (maps to project defaults)

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
