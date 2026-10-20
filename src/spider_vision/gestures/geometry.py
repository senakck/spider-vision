"""Hand-shape measurements used for gesture recognition.

All measurements use world landmarks (meters) and are ratios, so they do not
depend on hand size, distance from the camera or image resolution.
"""

import math
from enum import Enum

from spider_vision.vision.landmarks import Hand, HandLandmark, WorldPoint

L = HandLandmark


class Finger(Enum):
    """A finger and its joints, from the knuckle to the tip."""

    THUMB = (L.THUMB_CMC, L.THUMB_MCP, L.THUMB_IP, L.THUMB_TIP)
    INDEX = (
        L.INDEX_FINGER_MCP,
        L.INDEX_FINGER_PIP,
        L.INDEX_FINGER_DIP,
        L.INDEX_FINGER_TIP,
    )
    MIDDLE = (
        L.MIDDLE_FINGER_MCP,
        L.MIDDLE_FINGER_PIP,
        L.MIDDLE_FINGER_DIP,
        L.MIDDLE_FINGER_TIP,
    )
    RING = (L.RING_FINGER_MCP, L.RING_FINGER_PIP, L.RING_FINGER_DIP, L.RING_FINGER_TIP)
    PINKY = (L.PINKY_MCP, L.PINKY_PIP, L.PINKY_DIP, L.PINKY_TIP)

    @property
    def joints(self) -> tuple[HandLandmark, ...]:
        return self.value


def distance(a: WorldPoint, b: WorldPoint) -> float:
    """Straight-line 3D distance between two points, in meters."""
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def palm_size(hand: Hand) -> float:
    """Wrist to middle-finger knuckle. Stays constant whatever the fingers do,
    so it is a good reference length for ratios."""
    return distance(hand.world(L.WRIST), hand.world(L.MIDDLE_FINGER_MCP))


def straightness(hand: Hand, finger: Finger) -> float:
    """How straight a finger is: 1.0 when fully straight, smaller when curled.

    It is the knuckle-to-tip distance divided by the length of the finger
    measured along its joints.
    """
    points = [hand.world(joint) for joint in finger.joints]
    along_joints = sum(distance(a, b) for a, b in zip(points, points[1:]))
    if along_joints == 0:
        return 0.0
    return distance(points[0], points[-1]) / along_joints


def pinch_ratio(hand: Hand) -> float:
    """Thumb tip to index fingertip, relative to palm size. Small when pinching."""
    gap = distance(hand.world(L.THUMB_TIP), hand.world(L.INDEX_FINGER_TIP))
    return gap / palm_size(hand)


def thumb_spread(hand: Hand) -> float:
    """Thumb tip to index knuckle, relative to palm size.

    Large when the thumb sticks out, small when it is folded across the palm.
    """
    gap = distance(hand.world(L.THUMB_TIP), hand.world(L.INDEX_FINGER_MCP))
    return gap / palm_size(hand)
