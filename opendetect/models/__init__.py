from __future__ import annotations

from pathlib import Path

from .base import DetectorModel
from .rfdetr import RFDETRModel
from .yolox import YOLOXModel

MODEL_REGISTRY = {
    RFDETRModel.model_name: RFDETRModel,
    YOLOXModel.model_name: YOLOXModel,
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def create_model(
    model_name: str,
    *,
    input_size: tuple[int, int] | None = None,
    model_path: Path | None = None,
    threshold: float = 0.3,
    num_select: int = 300,
    class_ids: list[int] | None = None,
    providers: list[str] | None = None,
) -> DetectorModel:
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        )

    model_cls = MODEL_REGISTRY[key]
    resolved_input_size = (
        input_size if input_size is not None else model_cls.default_input_size()
    )
    return model_cls(
        input_size=resolved_input_size,
        model_path=model_path,
        threshold=threshold,
        num_select=num_select,
        class_ids=class_ids,
        providers=providers,
    )


def default_input_size(model_name: str) -> tuple[int, int]:
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        )
    return MODEL_REGISTRY[key].default_input_size()
