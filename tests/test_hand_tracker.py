"""Tests for the MediaPipe wrapper.

Most tests use fake MediaPipe results. One test loads the real model and is
skipped if it has not been downloaded yet.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spider_vision.config import HandTrackingConfig
from spider_vision.vision.hand_tracker import (
    HandTracker,
    MonotonicTimestamp,
    VisionError,
    hands_from_result,
)
from spider_vision.vision.landmarks import HandLandmark, Handedness

MODEL_PATH = HandTrackingConfig().model_path


def fake_result(handedness: str) -> SimpleNamespace:
    """Looks like a MediaPipe result with one hand."""
    points = [SimpleNamespace(x=i / 100, y=0.5, z=-0.01) for i in range(21)]
    category = SimpleNamespace(category_name=handedness, score=0.97)
    return SimpleNamespace(hand_landmarks=[points], handedness=[[category]])


def test_result_is_converted_to_domain_hand() -> None:
    [hand] = hands_from_result(fake_result("Right"))
    assert hand.handedness is Handedness.RIGHT
    assert hand.score == pytest.approx(0.97)
    assert hand[HandLandmark.INDEX_FINGER_TIP].x == pytest.approx(0.08)


def test_empty_result_gives_no_hands() -> None:
    empty = SimpleNamespace(hand_landmarks=[], handedness=[])
    assert hands_from_result(empty) == []


def test_timestamps_always_increase_even_if_clock_does_not() -> None:
    timestamps = MonotonicTimestamp(clock_ms=lambda: 1000)
    assert [timestamps.next() for _ in range(3)] == [1000, 1001, 1002]


def test_missing_model_gives_helpful_error() -> None:
    config = HandTrackingConfig(model_path=Path("does/not/exist.task"))
    with pytest.raises(VisionError, match="download_models"):
        HandTracker(config).open()


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="model not downloaded")
def test_real_model_finds_no_hand_in_black_frame() -> None:
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    with HandTracker(HandTrackingConfig()) as tracker:
        assert tracker.detect(black) == []
