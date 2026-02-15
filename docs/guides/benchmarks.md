# Benchmarks

This page summarizes benchmark runs generated with `opendetect-benchmark`.

Definitions:

- `Pipeline FPS`: end-to-end runtime excluding model load (preprocess + inference + postprocess)
- `Infer FPS`: model forward pass only
- `mAP`: reported COCO mAP for each model variant

Plots map:

- `x`: inference FPS (log scale)
- `y`: COCO mAP
- marker shape: runtime (CPU/CoreML/CUDA/TRT)
- stroke color: model family
- fill color: model variant
- bubble size: parameter count

## RF-DETR

```{image} ../benchmarks/accuracy_vs_infer_fps_bubble_params_rfdetr.svg
:alt: RF-DETR accuracy vs inference FPS
```

| Model | Params (M) | FLOPs (G) | COCO mAP |
| --- | ---: | ---: | ---: |
| RF-DETR-n | 26.877 | 31.787 | 48.4 |
| RF-DETR-s | 28.524 | 59.560 | 53.0 |
| RF-DETR-m | 30.101 | 78.492 | 54.7 |
| RF-DETR-l | 30.350 | 125.234 | 56.5 |

| Model | CPU (Pipeline/Infer) | CoreML (Pipeline/Infer) | CUDA (Pipeline/Infer) | TRT FP32 (Pipeline/Infer) | TRT FP16 (Pipeline/Infer) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF-DETR-n | 20.95 / 22.15 | 39.95 / 44.04 | 128.48 / 155.59 | 213.20 / 301.27 | 348.89 / 665.33 |
| RF-DETR-s | 11.48 / 12.01 | 28.80 / 32.09 | 80.82 / 97.25 | 126.40 / 170.44 | 224.73 / 425.00 |
| RF-DETR-m | 8.15 / 8.49 | 22.19 / 24.53 | 62.53 / 74.13 | 93.42 / 121.64 | 189.81 / 365.26 |
| RF-DETR-l | 4.76 / 4.92 | 14.74 / 16.20 | 36.66 / 43.24 | 55.10 / 69.67 | 126.76 / 246.85 |

## YOLOX

```{image} ../benchmarks/accuracy_vs_infer_fps_bubble_params_yolox.svg
:alt: YOLOX accuracy vs inference FPS
```

| Model | Params (M) | FLOPs (G) | COCO mAP |
| --- | ---: | ---: | ---: |
| YOLOX-tiny | 5.061 | 6.413 | 32.8 |
| YOLOX-s | 8.990 | 26.686 | 40.5 |
| YOLOX-m | 25.338 | 73.530 | 46.9 |
| YOLOX-l | 54.208 | 155.293 | 49.7 |
| YOLOX-x | 99.055 | 281.410 | 51.1 |

| Model | CPU (Pipeline/Infer) | CoreML (Pipeline/Infer) | CUDA (Pipeline/Infer) | TRT FP32 (Pipeline/Infer) | TRT FP16 (Pipeline/Infer) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOX-tiny | 54.97 / 57.05 | 130.92 / 142.59 | 197.38 / 206.49 | 522.87 / 588.58 | 776.84 / 935.37 |
| YOLOX-s | 18.00 / 18.00 | 72.30 / 78.24 | 105.81 / 108.35 | n/a | n/a |
| YOLOX-m | 8.11 / 8.19 | 49.12 / 51.73 | 69.66 / 70.77 | n/a | n/a |
| YOLOX-l | 4.94 / 4.97 | 35.07 / 36.31 | 46.28 / 46.78 | n/a | n/a |
| YOLOX-x | 3.04 / 3.05 | 22.23 / 22.73 | 28.69 / 28.89 | n/a | n/a |

## Notes

- YOLOX `m/l/x` are currently not supported on TensorRT in this project.
- Benchmark numbers depend on export graph, runtime versions, and hardware/software stack.
