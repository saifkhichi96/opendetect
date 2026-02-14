from __future__ import annotations

from .models import default_input_size


def parse_input_size(raw: str | None, model_name: str) -> tuple[int, int]:
    if raw is None:
        return default_input_size(model_name)

    normalized = raw.lower().replace("x", " ").replace(",", " ")
    parts = [part for part in normalized.split() if part]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --input-size '{raw}'. Expected format like '576x576'."
        )

    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid --input-size '{raw}'. Values must be > 0.")
    return height, width


def parse_optional_input_size(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None

    normalized = raw.lower().replace("x", " ").replace(",", " ")
    parts = [part for part in normalized.split() if part]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --input-size '{raw}'. Expected format like '576x576'."
        )

    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid --input-size '{raw}'. Values must be > 0.")
    return height, width


def parse_class_names(raw: list[str] | None) -> list[str]:
    if raw is None:
        return []

    names: list[str] = []
    for value in raw:
        for part in value.split(","):
            name = part.strip()
            if name:
                names.append(name)
    return names
