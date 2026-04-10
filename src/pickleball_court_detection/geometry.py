"""Court coordinate transforms and service-box rules."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def pixel_to_court(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    pt = np.array([[[px, py]]], dtype=np.float32)
    court_pt = cv2.perspectiveTransform(pt, H)[0][0]
    return float(court_pt[0]), float(court_pt[1])


def pixel_to_court_with_axes(
    px: float,
    py: float,
    H: np.ndarray,
    *,
    flip_court_x: bool = False,
) -> tuple[float, float]:
    """``pixel_to_court`` plus optional negation of court *x* (mirrored labels / camera)."""
    cx, cy = pixel_to_court(px, py, H)
    if flip_court_x:
        cx = -cx
    return cx, cy


def get_target_service_box(
    server_side: str,
    server_baseline: str,
    *,
    tolerance: float,
) -> dict[str, Any]:
    server_side = server_side.upper()
    if server_baseline == "near":
        target_y_min = 29 - tolerance
        target_y_max = 44 + tolerance
        if server_side == "R":
            target_x_min = -10 - tolerance
            target_x_max = 0 + tolerance
            target_name = "far-left"
        else:
            target_x_min = 0 - tolerance
            target_x_max = 10 + tolerance
            target_name = "far-right"
    else:
        target_y_min = 0 - tolerance
        target_y_max = 15 + tolerance
        if server_side == "R":
            target_x_min = -10 - tolerance
            target_x_max = 0 + tolerance
            target_name = "near-left"
        else:
            target_x_min = 0 - tolerance
            target_x_max = 10 + tolerance
            target_name = "near-right"

    return {
        "x_min": target_x_min,
        "x_max": target_x_max,
        "y_min": target_y_min,
        "y_max": target_y_max,
        "name": target_name,
    }


def is_in_target_box(cx: float, cy: float, target_box: dict[str, Any]) -> bool:
    return bool(
        target_box["x_min"] <= cx <= target_box["x_max"]
        and target_box["y_min"] <= cy <= target_box["y_max"]
    )


def court_feet_plausible(cx: float, cy: float, margin: float = 2.5) -> bool:
    """
    True if (cx, cy) could lie on/near the playing area.

    Homography from noisy keypoints can map pixels to huge values; reject those
    instead of reporting a misleading IN/OUT.
    """
    return bool(
        (-10.0 - margin) <= cx <= (10.0 + margin) and (-margin) <= cy <= (44.0 + margin)
    )
