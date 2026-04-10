"""Court keypoint ordering, smoothing, and homography."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pickleball_court_detection.config import CourtReliabilityConfig

# Model output order (CVAT / dataset.yaml skeleton)
MODEL_TO_CANONICAL: list[int] = [0, 4, 5, 1, 6, 7, 2, 3]

NEAR_INDICES: tuple[int, ...] = (4, 5, 6, 7)
FAR_INDICES: tuple[int, ...] = (0, 1, 2, 3)


def keypoints_model_to_canonical(
    keypoints: np.ndarray,
    confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx = MODEL_TO_CANONICAL
    return keypoints[idx].copy(), confidences[idx].copy()


def extract_pose_keypoints_from_yolo_result(
    results: list[Any],
    *,
    reorder_to_canonical: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Parse first YOLO-pose result into (keypoints, confidences) or Nones."""
    if not results or results[0].keypoints is None:
        return None, None, False
    kpts = results[0].keypoints
    if kpts.xy is None or len(kpts.xy) == 0:
        return None, None, False

    xy_all = kpts.xy.cpu().numpy()
    if xy_all.ndim == 2:
        xy_all = xy_all[np.newaxis, :, :]

    cf_all = None
    if kpts.conf is not None:
        cf_all = kpts.conf.cpu().numpy()
        if cf_all.ndim == 1:
            cf_all = cf_all[np.newaxis, :]

    if xy_all.shape[0] > 1 and cf_all is not None:
        best_i = int(np.argmax(cf_all.mean(axis=1)))
    else:
        best_i = 0

    raw_keypoints = np.asarray(xy_all[best_i], dtype=np.float32)
    if cf_all is not None:
        raw_confidences = np.asarray(cf_all[best_i], dtype=np.float32)
    else:
        raw_confidences = np.ones(raw_keypoints.shape[0], dtype=np.float32)

    if raw_keypoints.shape[0] != 8:
        return None, None, False

    if reorder_to_canonical:
        kp, cf = keypoints_model_to_canonical(raw_keypoints, raw_confidences)
    else:
        kp, cf = raw_keypoints.copy(), raw_confidences.copy()
    return kp, cf, True


class CourtStateManager:
    """Temporal smoothing and reliability gating for court keypoints."""

    def __init__(self, config: CourtReliabilityConfig, num_keypoints: int = 8):
        self.cfg = config
        self.num_keypoints = num_keypoints
        self.persistent_keypoints: np.ndarray | None = None
        self.frames_since_reliable = 999
        self.last_reliable_frame_idx = -1

    def is_frame_reliable(
        self, keypoints: np.ndarray | None, confidences: np.ndarray | None
    ) -> tuple[bool, str]:
        if keypoints is None or confidences is None:
            return False, "no_detection"

        thr = self.cfg.keypoint_confidence_threshold
        valid_mask = confidences >= thr
        valid_count = int(valid_mask.sum())
        if valid_count < self.cfg.min_keypoints_required:
            return False, f"too_few_keypoints ({valid_count})"

        if self.cfg.require_near_and_far:
            has_near = any(valid_mask[i] for i in NEAR_INDICES)
            has_far = any(valid_mask[i] for i in FAR_INDICES)
            if not (has_near and has_far):
                return False, "missing_near_or_far"

        return True, "ok"

    def update(
        self,
        keypoints: np.ndarray | None,
        confidences: np.ndarray | None,
        frame_idx: int,
    ) -> tuple[bool, str]:
        is_reliable, reason = self.is_frame_reliable(keypoints, confidences)

        if is_reliable and keypoints is not None and confidences is not None:
            valid_mask = confidences >= self.cfg.keypoint_confidence_threshold
            alpha = self.cfg.smoothing_alpha
            max_jump = self.cfg.max_keypoint_jump_px

            view_jump = False
            if self.persistent_keypoints is not None:
                dists = [
                    float(np.linalg.norm(keypoints[i] - self.persistent_keypoints[i]))
                    for i in range(self.num_keypoints)
                    if valid_mask[i]
                ]
                need = self.cfg.court_jump_min_keypoints_for_test
                if len(dists) >= need:
                    med = float(np.median(np.asarray(dists, dtype=np.float64)))
                    if med > float(self.cfg.court_view_jump_median_px):
                        view_jump = True

            if self.persistent_keypoints is None:
                self.persistent_keypoints = keypoints.copy()
            elif view_jump:
                if self.cfg.court_jump_hard_reset:
                    self.persistent_keypoints = keypoints.copy()
                else:
                    for i in range(self.num_keypoints):
                        if valid_mask[i]:
                            self.persistent_keypoints[i] = keypoints[i]
            else:
                for i in range(self.num_keypoints):
                    if valid_mask[i]:
                        old_pt = self.persistent_keypoints[i]
                        new_pt = keypoints[i]
                        dist = float(np.linalg.norm(new_pt - old_pt))
                        if dist < max_jump:
                            self.persistent_keypoints[i] = alpha * new_pt + (1 - alpha) * old_pt

            self.frames_since_reliable = 0
            self.last_reliable_frame_idx = frame_idx
            if view_jump:
                return True, "view_jump_reset"
        else:
            self.frames_since_reliable += 1

        return is_reliable, reason

    def get_court_status(self) -> tuple[np.ndarray | None, bool, str]:
        if self.persistent_keypoints is None:
            return None, False, "no_court_ever_detected"
        if self.frames_since_reliable <= self.cfg.max_unreliable_frames:
            return self.persistent_keypoints, True, "ok"
        return self.persistent_keypoints, False, "court_too_old"


def build_homography(keypoints: np.ndarray) -> np.ndarray | None:
    court_coords = np.array(
        [
            [-10, 44],
            [-10, 29],
            [10, 29],
            [10, 44],
            [-10, 15],
            [10, 15],
            [-10, 0],
            [10, 0],
        ],
        dtype=np.float32,
    )
    pts_image = keypoints.astype(np.float32)
    H, _status = cv2.findHomography(pts_image, court_coords)
    return H
