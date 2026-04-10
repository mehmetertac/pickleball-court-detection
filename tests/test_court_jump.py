"""Court state resets when camera view jumps (median keypoint motion)."""

from __future__ import annotations

import numpy as np

from pickleball_court_detection import CourtReliabilityConfig, CourtStateManager


def _kp_base() -> tuple[np.ndarray, np.ndarray]:
    kp = np.array(
        [
            [100.0, 50.0],
            [200.0, 50.0],
            [200.0, 150.0],
            [100.0, 150.0],
            [100.0, 250.0],
            [200.0, 250.0],
            [100.0, 350.0],
            [200.0, 350.0],
        ],
        dtype=np.float32,
    )
    cf = np.ones(8, dtype=np.float32) * 0.9
    return kp, cf


def test_view_jump_resets_smoothed_state() -> None:
    cfg = CourtReliabilityConfig(
        court_view_jump_median_px=40.0,
        court_jump_min_keypoints_for_test=6,
        court_jump_hard_reset=True,
    )
    m = CourtStateManager(cfg)
    kp0, cf0 = _kp_base()
    ok, r0 = m.update(kp0, cf0, 0)
    assert ok and r0 == "ok"
    assert m.persistent_keypoints is not None

    kp1 = kp0 + 120.0  # large shift all corners → median motion >> 40
    ok, r1 = m.update(kp1, cf0, 1)
    assert ok and r1 == "view_jump_reset"
    assert np.allclose(m.persistent_keypoints, kp1)


def test_small_motion_still_smooths() -> None:
    cfg = CourtReliabilityConfig(
        court_view_jump_median_px=80.0,
        court_jump_min_keypoints_for_test=6,
        smoothing_alpha=0.5,
        max_keypoint_jump_px=200.0,
    )
    m = CourtStateManager(cfg)
    kp0, cf0 = _kp_base()
    m.update(kp0, cf0, 0)
    kp1 = kp0 + 2.0
    ok, r1 = m.update(kp1, cf0, 1)
    assert ok and r1 == "ok"
    assert not np.allclose(m.persistent_keypoints, kp1)
    assert not np.allclose(m.persistent_keypoints, kp0)
