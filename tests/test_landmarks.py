"""Tests for the hand domain model. No MediaPipe or camera required."""

import pytest

from spider_vision.vision.landmarks import (
    HAND_CONNECTIONS,
    Hand,
    HandLandmark,
    Handedness,
    Landmark,
    WorldPoint,
)


def make_hand(count: int = 21, world_count: int = 21) -> Hand:
    """A hand whose landmark i sits at x = i / 100, so each is distinguishable."""
    landmarks = tuple(Landmark(x=i / 100, y=0.5, z=0.0) for i in range(count))
    world = tuple(WorldPoint(x=i / 1000, y=0.0, z=0.0) for i in range(world_count))
    return Hand(
        landmarks=landmarks,
        world_landmarks=world,
        handedness=Handedness.RIGHT,
        score=0.9,
    )


def test_there_are_21_landmarks_numbered_like_mediapipe() -> None:
    assert len(HandLandmark) == 21
    assert HandLandmark.WRIST == 0
    assert HandLandmark.INDEX_FINGER_TIP == 8
    assert HandLandmark.PINKY_TIP == 20


def test_normalized_point_converts_to_pixels() -> None:
    point = Landmark(x=0.5, y=0.25, z=0.0)
    assert point.to_pixel(width=640, height=480) == (320, 120)


def test_hand_landmarks_can_be_read_by_name() -> None:
    hand = make_hand()
    assert hand[HandLandmark.INDEX_FINGER_TIP].x == pytest.approx(0.08)
    assert hand.world(HandLandmark.INDEX_FINGER_TIP).x == pytest.approx(0.008)


@pytest.mark.parametrize("count", [0, 20, 22])
def test_hand_with_wrong_landmark_count_is_rejected(count: int) -> None:
    with pytest.raises(ValueError):
        make_hand(count=count)


def test_hand_with_wrong_world_landmark_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="world_landmarks"):
        make_hand(world_count=20)


def test_skeleton_connects_every_landmark() -> None:
    connected = {point for pair in HAND_CONNECTIONS for point in pair}
    assert connected == set(HandLandmark)
