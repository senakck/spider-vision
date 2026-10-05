"""Domain model for a tracked hand.

These types are what the rest of the app sees. They do not depend on
MediaPipe, so gesture code and tests can use them without loading a model.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class HandLandmark(IntEnum):
    """The 21 hand landmarks, numbered the way MediaPipe numbers them."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Handedness(StrEnum):
    """Which hand it is, from the user's point of view."""

    LEFT = "Left"
    RIGHT = "Right"


@dataclass(frozen=True)
class Landmark:
    """One landmark in normalized image coordinates.

    x and y are in [0, 1] relative to the image width and height
    (0, 0 is the top-left corner); values slightly outside that range mean
    the point is just off-screen. z is depth relative to the wrist: smaller
    (more negative) values are closer to the camera, on roughly the same
    scale as x.
    """

    x: float
    y: float
    z: float

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert to pixel coordinates for an image of the given size."""
        return round(self.x * width), round(self.y * height)


@dataclass(frozen=True)
class Hand:
    """A single detected hand."""

    landmarks: tuple[Landmark, ...]
    handedness: Handedness
    score: float
    """How confident the model is about the handedness, from 0 to 1."""

    def __post_init__(self) -> None:
        if len(self.landmarks) != len(HandLandmark):
            raise ValueError(
                f"a hand needs {len(HandLandmark)} landmarks, got {len(self.landmarks)}"
            )

    def __getitem__(self, landmark: HandLandmark) -> Landmark:
        return self.landmarks[landmark]


# Pairs of landmarks joined by a bone, used for drawing the hand skeleton.
HAND_CONNECTIONS: tuple[tuple[HandLandmark, HandLandmark], ...] = (
    # palm
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
    # thumb
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    # index
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    # middle
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    # ring
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    # pinky
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
)
