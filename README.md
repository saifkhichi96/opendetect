# OpenDetect

Open-source object detection for Python developers. Frictionless installation. Free for commercial use.

`opendetect` packages high-quality ONNX object detection models behind one consistent API, with both Python and CLI workflows.

## Why OpenDetect

- Consistent API across multiple model families (`rfdetr`, `yolox`)
- ONNX Runtime inference with automatic provider selection
- OpenCV/NumPy-first I/O and preprocessing
- Built-in model registry with known input sizes and download URLs
- Optional auto-download from Hugging Face model artifacts
- Ready-to-use CLI for inference, benchmarking, and ONNX simplification

## Installation

```bash
pip install opendetect
```

For explicit runtime selection:

```bash
# CPU runtime (explicit)
pip install "opendetect[cpu]"

# CUDA runtime
pip install "opendetect[gpu]"
```

Core runtime dependencies are:

- `onnxruntime`
- `opencv-python`
- `numpy`

Optional extras:

```bash
pip install "opendetect[benchmark]"   # ONNX stats/FLOPs tooling
```

## Quickstart (Python)

```python
import cv2
from opendetect import Detector

# model can be a registry model-id (recommended) or backend name
# examples: rfdetr-m, rfdetr-l, yolox-s, yolox-m

detector = Detector(model="rfdetr-m")

image = cv2.imread("input.jpg")
detections = detector.predict(image, color="bgr")
annotated = detector.annotate(image, detections, color="bgr")

cv2.imwrite("output.jpg", annotated)
```

Run file-based inference directly:

```python
from opendetect import Detector

detector = Detector(model="yolox-s")
detector.infer_image_file("input.jpg", output_path="output.jpg")
detector.infer_video_file("input.mp4", output_path="output.mp4", max_frames=300)
```

Class IDs are normalized across models:

- `0`-based foreground class IDs by default
- RF-DETR background index `0` is ignored internally
- If `--classes` (or `class_ids`) is omitted, all foreground classes are kept

## Model Registry

OpenDetect includes an internal model registry with metadata per model:

- implementation family (`rfdetr` / `yolox`)
- default input size
- artifact path and remote URL

List models:

```python
from opendetect import list_models

for spec in list_models():
    print(spec.model_id, spec.input_size)
```

Inspect/download from CLI:

```bash
opendetect-models list
opendetect-models info rfdetr-m
opendetect-models download rfdetr-m
```

By default models are cached under `~/.cache/opendetect/checkpoints`.
Set `OPENDETECT_CACHE_DIR` to override.

## CLI

### Inference

```bash
opendetect --version
opendetect-infer --image data/images/crowd.png --model-id rfdetr-m --output output.png
opendetect-infer --video input.mp4 --model-id yolox-s --tensor-rt --output output.mp4
# list supported classes for a model
opendetect-infer --model-id yolox-s --list-classes
# filter by class names instead of IDs
opendetect-infer --image data/images/crowd.png --model-id yolox-s --class-names person,bicycle
```

### Benchmark

```bash
opendetect-benchmark --model-id yolox-m --mode dummy --warmup 20 --iterations 200 --tensor-rt
opendetect-benchmark --model-id rfdetr-l --mode video --video input.mp4 --max-frames 500
```

## License

Apache License 2.0 (`LICENSE`).
