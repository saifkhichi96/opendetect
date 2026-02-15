# Development Setup

## Clone and Install

```bash
git clone https://github.com/saifkhichi96/opendetect.git
cd opendetect
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs,simplify]"
```

## Run Linters

```bash
ruff check .
```

## Build Documentation

```bash
make -C docs html
```

Output:

- `docs/_build/html/index.html`

For stricter local checks:

```bash
sphinx-build -W -b html docs docs/_build/html
```

## Benchmark Plot Assets

Current benchmark SVGs live under:

- `docs/benchmarks/`

Plot generation script:

- `src/plot_benchmarks_svg.py`

## Suggested Contributor Additions

- include hardware and runtime versions when reporting benchmark numbers
- share provider output from `onnxruntime.get_available_providers()`
- include minimal reproducer commands for runtime-specific issues
