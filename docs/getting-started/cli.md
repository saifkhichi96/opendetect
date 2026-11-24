# CLI Usage

OpenDetect installs these commands:

- `opendetect`
- `opendetect-infer`
- `opendetect-benchmark`
- `opendetect-models`
- `opendetect-simplify`

## Version

```bash
opendetect --version
```

## Inference

You can select any registered model ID from these families:

- `rfdetr`
- `yolox`
- `rtmdet`

Image inference:

```bash
opendetect-infer \
  --image data/images/crowd.png \
  --model-id rfdetr-m \
  --output output.png
```

Video inference:

```bash
opendetect-infer \
  --video input.mp4 \
  --model-id yolox-s \
  --tensor-rt \
  --mixed-precision \
  --output output.mp4
```

Live instance segmentation:

```bash
opendetect-infer --camera 0 --model-id rfdetr-seg-n --class-names person
```

The preview overlays masks and boxes. For a local ONNX export, add
`--model /path/to/rfdetr-seg-nano.onnx` with the matching model ID.

Live camera tracking:

```bash
pip install -e '.[tracking]'
opendetect-infer --camera 0 --model-id yolox-s --tracker bytetrack
```

The preview opens by default for cameras. Press **Q** or **Esc** to stop. To
save a clip, add `--output tracked.mp4`. For a headless capture, add
`--no-preview` and use `--max-frames` to limit the run.

To track a video file, use `--video input.mp4 --tracker bytetrack`. Add
`--preview` to watch as it processes. Tracking defaults to the person class
and a detection threshold of 0.1 so weak detections can maintain IDs. Use
`--classes` or `--class-names` to select a different single class.

Class discovery and filtering:

```bash
# Print supported classes for selected model
opendetect-infer --model-id yolox-s --list-classes

# Filter by class IDs
opendetect-infer --image input.jpg --model-id yolox-s --classes 0 2 5

# Filter by class names
opendetect-infer --image input.jpg --model-id yolox-s --class-names person,bicycle
```

## Benchmarking

Dummy benchmark (synthetic input):

```bash
opendetect-benchmark \
  --model-id rfdetr-l \
  --mode dummy \
  --warmup 20 \
  --iterations 200 \
  --tensor-rt \
  --mixed-precision
```

Video benchmark:

```bash
opendetect-benchmark \
  --model-id yolox-m \
  --mode video \
  --video input.mp4 \
  --max-frames 500
```

## Model Registry Tooling

```bash
opendetect-models list
opendetect-models info rfdetr-m
opendetect-models download yolox-s
```

## ONNX Simplify

```bash
opendetect-simplify model.onnx
```

Optional flags:

```bash
opendetect-simplify model.onnx --output model-sim.onnx --no-check
```
