"""Draws a tracked hand's skeleton onto a frame.

This is a debug overlay: it shows what the tracker sees. It only knows the
``Hand`` domain model, not MediaPipe.
"""

import cv2

from spider_vision.camera import Frame
from spider_vision.vision.landmarks import HAND_CONNECTIONS, Hand, HandLandmark

# Colors are BGR (OpenCV's channel order).
BONE_COLOR = (255, 255, 255)  # white
JOINT_COLOR = (0, 0, 220)  # red
LABEL_COLOR = (0, 255, 0)  # green
BONE_THICKNESS = 2
JOINT_RADIUS = 4


def draw_hand(frame: Frame, hand: Hand) -> None:
    """Draw bones, joints and a handedness label onto the frame (in place)."""
    height, width = frame.shape[:2]
    points = [landmark.to_pixel(width, height) for landmark in hand.landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], BONE_COLOR, BONE_THICKNESS)

    for point in points:
        cv2.circle(frame, point, JOINT_RADIUS, JOINT_COLOR, thickness=-1)

    wrist_x, wrist_y = points[HandLandmark.WRIST]
    cv2.putText(
        frame,
        f"{hand.handedness} {hand.score:.2f}",
        org=(wrist_x - 40, wrist_y + 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=LABEL_COLOR,
        thickness=2,
    )
