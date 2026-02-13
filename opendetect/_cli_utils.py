from __future__ import annotations

from .models import default_input_size


def parse_providers(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    providers = [value.strip() for value in raw.split(",") if value.strip()]
    return providers or None


def parse_required_providers(raw: str) -> list[str]:
    providers = [value.strip() for value in raw.split(",") if value.strip()]
    if not providers:
        raise ValueError("At least one execution provider is required.")
    return providers


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
