from ._version import __version__
from .detector import Detector, load_detector
from .download import download_model
from .registry import (
    ModelSpec,
    get_model_spec,
    list_model_ids,
    list_models,
    model_url,
)
from .types import Detections, LoadedModelInfo

__all__ = [
    "Detections",
    "Detector",
    "LoadedModelInfo",
    "ModelSpec",
    "__version__",
    "download_model",
    "get_model_spec",
    "list_model_ids",
    "list_models",
    "load_detector",
    "model_url",
]
