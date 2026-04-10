"""Configuration for court reliability / smoothing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CourtReliabilityConfig:
    keypoint_confidence_threshold: float = 0.5
    min_keypoints_required: int = 4
    require_near_and_far: bool = True
    max_unreliable_frames: int = 30
    smoothing_alpha: float = 0.3
    max_keypoint_jump_px: float = 50.0
    # Camera cut / angle change: median keypoint motion vs smoothed state (px) → reset
    court_view_jump_median_px: float = 55.0
    court_jump_min_keypoints_for_test: int = 6
    court_jump_hard_reset: bool = True
