"""Convenience wrappers around Ultralytics YOLO court inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from pickleball_court_detection.court import extract_pose_keypoints_from_yolo_result


def infer_court_keypoints_from_image(
    model: Any,
    image_bgr: np.ndarray,
    *,
    reorder_to_canonical: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """
    Run ``model`` (Ultralytics YOLO) on a BGR image and parse the first pose as court keypoints.

    Returns:
        ``(keypoints_xy, confidences, ok)`` — same contract as
        :func:`~pickleball_court_detection.court.extract_pose_keypoints_from_yolo_result`.
    """
    results = model(image_bgr, verbose=False)
    return extract_pose_keypoints_from_yolo_result(
        results, reorder_to_canonical=reorder_to_canonical
    )
