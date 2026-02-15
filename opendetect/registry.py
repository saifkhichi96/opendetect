from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_BASE_URL = "https://huggingface.co/saifkhichi96/opendetect/resolve/main"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    implementation: str
    input_size: tuple[int, int]
    artifact_path: str
    aliases: tuple[str, ...] = ()
    description: str = ""

    @property
    def filename(self) -> str:
        return Path(self.artifact_path).name


_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="rfdetr-n",
        implementation="rfdetr",
        input_size=(384, 384),
        artifact_path="rfdetr/rfdetr_n_v142_384x384.onnx",
        aliases=("rfdetr-nano",),
        description="RF-DETR Nano",
    ),
    ModelSpec(
        model_id="rfdetr-s",
        implementation="rfdetr",
        input_size=(512, 512),
        artifact_path="rfdetr/rfdetr_s_v142_512x512.onnx",
        aliases=("rfdetr-small",),
        description="RF-DETR Small",
    ),
    ModelSpec(
        model_id="rfdetr-m",
        implementation="rfdetr",
        input_size=(576, 576),
        artifact_path="rfdetr/rfdetr_m_v142_576x576.onnx",
        aliases=("rfdetr", "rfdetr-medium"),
        description="RF-DETR Medium",
    ),
    ModelSpec(
        model_id="rfdetr-l",
        implementation="rfdetr",
        input_size=(704, 704),
        artifact_path="rfdetr/rfdetr_l_v142_704x704.onnx",
        aliases=("rfdetr-large",),
        description="RF-DETR Large",
    ),
    ModelSpec(
        model_id="yolox-tiny",
        implementation="yolox",
        input_size=(416, 416),
        artifact_path="yolox/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
        aliases=("yolox-t", "yolox_tiny"),
        description="YOLOX Tiny",
    ),
    ModelSpec(
        model_id="yolox-s",
        implementation="yolox",
        input_size=(640, 640),
        artifact_path="yolox/yolox_s_8xb8-300e_humanart-3ef259a7.onnx",
        aliases=("yolox", "yolox-small"),
        description="YOLOX Small",
    ),
    ModelSpec(
        model_id="yolox-m",
        implementation="yolox",
        input_size=(640, 640),
        artifact_path="yolox/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
        aliases=("yolox-medium",),
        description="YOLOX Medium",
    ),
    ModelSpec(
        model_id="yolox-l",
        implementation="yolox",
        input_size=(640, 640),
        artifact_path="yolox/yolox_l_8xb8-300e_humanart-ce1d7a62.onnx",
        aliases=("yolox-large",),
        description="YOLOX Large",
    ),
    ModelSpec(
        model_id="yolox-x",
        implementation="yolox",
        input_size=(640, 640),
        artifact_path="yolox/yolox_x_8xb8-300e_humanart-a39d44ed.onnx",
        aliases=("yolox-xlarge",),
        description="YOLOX X",
    ),
)

_MODEL_BY_ID: dict[str, ModelSpec] = {spec.model_id: spec for spec in _MODEL_SPECS}
_MODEL_ALIASES: dict[str, str] = {}
for _spec in _MODEL_SPECS:
    _MODEL_ALIASES[_spec.model_id] = _spec.model_id
    for _alias in _spec.aliases:
        _MODEL_ALIASES[_alias] = _spec.model_id

_DEFAULT_ID_BY_IMPL = {
    "rfdetr": "rfdetr-m",
    "yolox": "yolox-s",
}


def model_base_url() -> str:
    return os.getenv("OPENDETECT_MODEL_BASE_URL", DEFAULT_MODEL_BASE_URL).rstrip("/")


def model_url(spec: ModelSpec, base_url: str | None = None) -> str:
    root = (base_url or model_base_url()).rstrip("/")
    return f"{root}/{spec.artifact_path}"


def list_models() -> list[ModelSpec]:
    return list(_MODEL_SPECS)


def list_model_ids() -> list[str]:
    return [spec.model_id for spec in _MODEL_SPECS]


def default_model_id(implementation: str) -> str:
    key = implementation.strip().lower()
    if key not in _DEFAULT_ID_BY_IMPL:
        raise ValueError(
            f"Unknown implementation '{implementation}'. Expected one of: {sorted(_DEFAULT_ID_BY_IMPL)}"
        )
    return _DEFAULT_ID_BY_IMPL[key]


def resolve_model_id(model: str) -> str:
    key = model.strip().lower()
    if key in _DEFAULT_ID_BY_IMPL:
        return _DEFAULT_ID_BY_IMPL[key]
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]

    raise ValueError(
        f"Unknown model '{model}'. Available model IDs: {', '.join(list_model_ids())}"
    )


def get_model_spec(model: str) -> ModelSpec:
    model_id = resolve_model_id(model)
    return _MODEL_BY_ID[model_id]


def get_model_spec_by_id(model_id: str) -> ModelSpec:
    key = model_id.strip().lower()
    if key not in _MODEL_BY_ID:
        raise ValueError(
            f"Unknown model ID '{model_id}'. Available model IDs: {', '.join(list_model_ids())}"
        )
    return _MODEL_BY_ID[key]
