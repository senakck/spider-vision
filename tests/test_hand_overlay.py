"""Tests for the hand overlay, drawing onto a blank image."""

import numpy as np

from spider_vision.effects import draw_hand
from spider_vision.effects.hand_overlay import JOINT_COLOR
from spider_vision.vision.landmarks import Hand, HandLandmark, Handedness, Landmark


def make_hand() -> Hand:
    """All landmarks at the image center, except the index fingertip."""
    landmarks = [Landmark(0.5, 0.5, 0.0)] * 21
    landmarks[HandLandmark.INDEX_FINGER_TIP] = Landmark(0.25, 0.25, 0.0)
    return Hand(tuple(landmarks), Handedness.RIGHT, score=0.9)


def test_joint_is_drawn_at_landmark_pixel_position() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hand(frame, make_hand())
    # The index fingertip (0.25, 0.25) is pixel (160, 120); numpy indexes [y, x].
    assert tuple(frame[120, 160]) == JOINT_COLOR


def test_far_corner_is_left_untouched() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hand(frame, make_hand())
    assert frame[0, 639].sum() == 0
