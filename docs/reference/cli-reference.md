# CLI Reference

## `opendetect`

Top-level command currently exposes version output.

```bash
opendetect --version
```

## `opendetect-infer`

Run object detection on an image or video.

Required input:

- `--image PATH` or `--video PATH` (unless `--list-classes`)

Model selection:

- `--model-id` (preferred)
- `--model-name` (`rfdetr` or `yolox`)
- `--model PATH` (local ONNX model)
- `--input-size HxW`

Runtime:

- `--no-hardware-acceleration`
- `--tensor-rt`
- `--mixed-precision`

Filtering:

- `--classes ID [ID ...]`
- `--class-names NAME [NAME ...]`
- `--list-classes`

Output:

- `--output PATH`
- `--max-frames N` (video mode)

Model download controls:

- `--no-download`
- `--cache-dir PATH`

## `opendetect-benchmark`

Benchmark model pipeline speed.

Modes:

- `--mode dummy` with `--iterations`, `--dummy-height`, `--dummy-width`, `--seed`
- `--mode video` with `--video` and optional `--max-frames`

Shared controls:

- model and runtime flags from `opendetect-infer`
- `--warmup`
- `--threshold`
- `--num-select`
- class filters (`--classes`, `--class-names`, `--list-classes`)

Reported metrics include stage latency stats, pipeline FPS, and inference-only FPS.

## `opendetect-models`

Inspect model metadata and download artifacts.

Subcommands:

- `list`
- `info MODEL_ID`
- `download MODEL_ID [--cache-dir PATH] [--force]`

## `opendetect-simplify`

Simplify ONNX graphs using ONNX GraphSurgeon.

Usage:

```bash
opendetect-simplify INPUT.onnx [--output OUTPUT.onnx] [--no-fold-constants] [--no-infer-shapes] [--no-check]
```
