from __future__ import annotations

import cv2
import numpy as np


def color_for_class_id(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic bright BGR color for a class ID."""
    idx = int(class_id) + 1
    b = (37 * idx + 23) % 256
    g = (17 * idx + 71) % 256
    r = (29 * idx + 131) % 256
    return ((b + 255) // 2, (g + 255) // 2, (r + 255) // 2)


def class_name_for_id(class_id: int, class_names: list[str] | None) -> str:
    idx = int(class_id)
    if class_names is not None and 0 <= idx < len(class_names):
        return str(class_names[idx])
    return f"class_{idx}"


def draw_box_with_label(
    frame_bgr: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color_bgr: tuple[int, int, int],
    label: str,
) -> None:
    cv2.rectangle(frame_bgr, pt1, pt2, color_bgr, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    text_x = max(0, pt1[0])
    text_y = max(text_h + baseline + 2, pt1[1] - 4)

    bg_tl = (text_x, text_y - text_h - baseline - 2)
    bg_br = (text_x + text_w + 4, text_y + 2)
    cv2.rectangle(frame_bgr, bg_tl, bg_br, color_bgr, -1)

    cv2.putText(
        frame_bgr,
        label,
        (text_x + 2, text_y - baseline),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
