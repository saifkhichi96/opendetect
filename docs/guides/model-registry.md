# Model Registry

OpenDetect ships a built-in model registry. Each model entry defines:

- `model_id`
- implementation family (`rfdetr` or `yolox`)
- default input size
- remote artifact path
- aliases and description

## Included Families

| Family | Year | License | Why included |
| --- | ---: | --- | --- |
| RF-DETR | 2026 | Apache-2.0 | Strong modern accuracy/speed tradeoff |
| YOLOX | 2021 | Apache-2.0 | Stable last Apache-2.0 YOLO-family baseline |

## Model IDs

| Model ID | Family | Default Input |
| --- | --- | --- |
| `rfdetr-n` | RF-DETR | `384x384` |
| `rfdetr-s` | RF-DETR | `512x512` |
| `rfdetr-m` | RF-DETR | `576x576` |
| `rfdetr-l` | RF-DETR | `704x704` |
| `yolox-tiny` | YOLOX | `416x416` |
| `yolox-s` | YOLOX | `640x640` |
| `yolox-m` | YOLOX | `640x640` |
| `yolox-l` | YOLOX | `640x640` |
| `yolox-x` | YOLOX | `640x640` |

## Inspect and Download

CLI:

```bash
opendetect-models list
opendetect-models info rfdetr-m
opendetect-models download rfdetr-m
```

Python:

```python
from opendetect import list_models, get_model_spec

print([spec.model_id for spec in list_models()])
print(get_model_spec("rfdetr-m"))
```

## Caching

Default cache path:

- `~/.cache/opendetect/checkpoints`

Override with:

- `OPENDETECT_CACHE_DIR`

You can also override the registry base URL with:

- `OPENDETECT_MODEL_BASE_URL`
