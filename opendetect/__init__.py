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
    "__version__",
    "Detector",
    "Detections",
    "LoadedModelInfo",
    "ModelSpec",
    "download_model",
    "get_model_spec",
    "list_model_ids",
    "list_models",
    "load_detector",
    "model_url",
]
