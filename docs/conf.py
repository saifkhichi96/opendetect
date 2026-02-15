from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "OpenDetect"
author = "OpenDetect Contributors"
copyright = f"{datetime.now().year}, {author}"

try:
    from opendetect._version import __version__ as release
except Exception:
    release = "0.0.0"

version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_mock_imports = [
    "cv2",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnx_graphsurgeon",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"
html_title = "OpenDetect Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
