"""Tests for hand geometry, using hand-made world coordinates."""

import pytest

from spider_vision.gestures.geometry import (
    Finger,
    distance,
    palm_size,
    pinch_ratio,
    straightness,
    thumb_to_pinky_ratio,
)
from spider_vision.vision.landmarks import (
    Hand,
    HandLandmark,
    Handedness,
    Landmark,
    WorldPoint,
)

L = HandLandmark
ORIGIN = WorldPoint(0.0, 0.0, 0.0)


def make_hand(
    points: dict[HandLandmark, WorldPoint] | None = None,
    screen: dict[HandLandmark, Landmark] | None = None,
) -> Hand:
    """A hand where unspecified points sit at the origin / screen center."""
    points = points or {}
    screen = screen or {}
    world = tuple(points.get(landmark, ORIGIN) for landmark in HandLandmark)
    center = Landmark(0.5, 0.5, 0.0)
    landmarks = tuple(screen.get(landmark, center) for landmark in HandLandmark)
    return Hand(landmarks, world, Handedness.RIGHT, score=1.0)


def test_distance_is_3d_euclidean() -> None:
    assert distance(ORIGIN, WorldPoint(0.0, 0.03, 0.04)) == pytest.approx(0.05)


def test_palm_size_is_wrist_to_middle_knuckle() -> None:
    hand = make_hand({L.WRIST: ORIGIN, L.MIDDLE_FINGER_MCP: WorldPoint(0.0, 0.08, 0.0)})
    assert palm_size(hand) == pytest.approx(0.08)


def test_straight_finger_has_straightness_one() -> None:
    hand = make_hand(
        {
            L.INDEX_FINGER_MCP: WorldPoint(0.0, 0.00, 0.0),
            L.INDEX_FINGER_PIP: WorldPoint(0.0, 0.03, 0.0),
            L.INDEX_FINGER_DIP: WorldPoint(0.0, 0.05, 0.0),
            L.INDEX_FINGER_TIP: WorldPoint(0.0, 0.07, 0.0),
        }
    )
    assert straightness(hand, Finger.INDEX) == pytest.approx(1.0)


def test_curled_finger_has_low_straightness() -> None:
    # The finger goes up 3 cm, then bends forward and back down:
    # the tip ends 1 cm from the knuckle after 7 cm of finger.
    hand = make_hand(
        {
            L.INDEX_FINGER_MCP: WorldPoint(0.0, 0.00, 0.00),
            L.INDEX_FINGER_PIP: WorldPoint(0.0, 0.03, 0.00),
            L.INDEX_FINGER_DIP: WorldPoint(0.0, 0.03, 0.02),
            L.INDEX_FINGER_TIP: WorldPoint(0.0, 0.01, 0.02),
        }
    )
    assert straightness(hand, Finger.INDEX) < 0.4


def test_thumb_to_pinky_ratio_is_relative_to_palm_size() -> None:
    hand = make_hand(
        {
            L.WRIST: ORIGIN,
            L.MIDDLE_FINGER_MCP: WorldPoint(0.0, 0.08, 0.0),
            L.PINKY_MCP: WorldPoint(-0.02, 0.07, 0.0),
            L.THUMB_TIP: WorldPoint(0.06, 0.07, 0.0),
        }
    )
    assert thumb_to_pinky_ratio(hand) == pytest.approx(0.08 / 0.08)


def test_pinch_ratio_uses_screen_plane_relative_to_palm() -> None:
    # Square image: palm is 0.2 tall on screen, fingertips are 0.05 apart.
    hand = make_hand(
        screen={
            L.WRIST: Landmark(0.5, 0.8, 0.0),
            L.MIDDLE_FINGER_MCP: Landmark(0.5, 0.6, 0.0),
            L.THUMB_TIP: Landmark(0.45, 0.4, 0.0),
            L.INDEX_FINGER_TIP: Landmark(0.50, 0.4, 0.0),
        }
    )
    assert pinch_ratio(hand, aspect_ratio=1.0) == pytest.approx(0.05 / 0.2)


def test_pinch_ratio_ignores_depth() -> None:
    # Fingertips touch on screen but MediaPipe guessed different depths.
    hand = make_hand(
        screen={
            L.WRIST: Landmark(0.5, 0.8, 0.0),
            L.MIDDLE_FINGER_MCP: Landmark(0.5, 0.6, 0.0),
            L.THUMB_TIP: Landmark(0.5, 0.4, -0.10),
            L.INDEX_FINGER_TIP: Landmark(0.5, 0.4, 0.05),
        }
    )
    assert pinch_ratio(hand, aspect_ratio=1.0) == pytest.approx(0.0)


def test_pinch_ratio_corrects_for_aspect_ratio() -> None:
    # On a 4:3 image a horizontal gap of 0.15 is as long as a vertical 0.2.
    hand = make_hand(
        screen={
            L.WRIST: Landmark(0.5, 0.8, 0.0),
            L.MIDDLE_FINGER_MCP: Landmark(0.5, 0.6, 0.0),
            L.THUMB_TIP: Landmark(0.35, 0.4, 0.0),
            L.INDEX_FINGER_TIP: Landmark(0.50, 0.4, 0.0),
        }
    )
    assert pinch_ratio(hand, aspect_ratio=4 / 3) == pytest.approx(1.0)


def test_every_finger_has_four_joints() -> None:
    assert all(len(finger.joints) == 4 for finger in Finger)
