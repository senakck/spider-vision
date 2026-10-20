"""Hand-shape measurements used for gesture recognition.

Every measurement is a ratio, so it does not depend on hand size, distance
from the camera or image resolution.

Most use world landmarks (meters, 3D). The pinch is the exception: when two
fingertips touch, MediaPipe's depth estimate for them is unreliable, so the
pinch is measured on the screen plane instead. These choices were made from
measurements of a real hand (see ROADMAP.md, M3).
"""

import math
from enum import Enum

from spider_vision.vision.landmarks import Hand, HandLandmark, Landmark, WorldPoint

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


def thumb_to_pinky_ratio(hand: Hand) -> float:
    """Thumb tip to pinky knuckle, relative to palm size.

    Small when the thumb is folded across the palm, large when it sticks out.
    """
    gap = distance(hand.world(L.THUMB_TIP), hand.world(L.PINKY_MCP))
    return gap / palm_size(hand)


def _screen_distance(a: Landmark, b: Landmark, aspect_ratio: float) -> float:
    """2D distance on the screen plane, corrected for the image's aspect ratio.

    Normalized x is a fraction of the width and y of the height, so x is
    scaled by width / height to make both axes use the same unit.
    """
    return math.dist((a.x * aspect_ratio, a.y), (b.x * aspect_ratio, b.y))


def pinch_ratio(hand: Hand, aspect_ratio: float) -> float:
    """Thumb tip to index fingertip on screen, relative to on-screen palm size.

    Around 0.1 in a natural pinch (the finger pads touch, the tips stay a
    little apart) and around 1 when the fingers are apart. Webcam tracking is
    not precise enough to tell "touching" from "1-2 cm apart", so a small
    value means "close together". ``aspect_ratio`` is image width / height.
    """
    gap = _screen_distance(hand[L.THUMB_TIP], hand[L.INDEX_FINGER_TIP], aspect_ratio)
    palm = _screen_distance(hand[L.WRIST], hand[L.MIDDLE_FINGER_MCP], aspect_ratio)
    if palm == 0:
        return math.inf
    return gap / palm
