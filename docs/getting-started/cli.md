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
- `bytetrack`

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
