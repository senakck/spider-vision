"""Measure hand-shape values to choose gesture thresholds.

Run:  python scripts/measure_hand.py

Hold a pose and press its key to print the current measurements:
    o = OPEN_HAND   f = FIST   p = POINT   n = PINCH   s = SPIDER_MAN
    ESC or q = quit

The values are printed to the terminal so they can be copied and compared.
"""

import cv2

from spider_vision.camera import Camera, Frame
from spider_vision.config import AppConfig
from spider_vision.effects import draw_hand
from spider_vision.gestures.geometry import (
    Finger,
    palm_size,
    pinch_ratio,
    straightness,
    thumb_to_pinky_ratio,
)
from spider_vision.vision import Hand, HandTracker

WINDOW_NAME = "Spider Vision - measure"
POSE_KEYS = {
    ord("o"): "OPEN_HAND",
    ord("f"): "FIST",
    ord("p"): "POINT",
    ord("n"): "PINCH",
    ord("s"): "SPIDER_MAN",
}
EXIT_KEYS = {27, ord("q")}


def measure(hand: Hand, frame: Frame) -> dict[str, float]:
    height, width = frame.shape[:2]
    values = {
        finger.name.lower(): straightness(hand, finger)
        for finger in Finger
        if finger is not Finger.THUMB
    }
    values["thumb"] = thumb_to_pinky_ratio(hand)
    values["pinch"] = pinch_ratio(hand, aspect_ratio=width / height)
    # Sanity check of the world scale: an adult palm is roughly 8-10 cm.
    values["palm_cm"] = palm_size(hand) * 100
    return values


def format_values(values: dict[str, float]) -> str:
    return "  ".join(f"{name}={value:.2f}" for name, value in values.items())


def draw_values(frame: Frame, values: dict[str, float]) -> None:
    for row, (name, value) in enumerate(values.items()):
        cv2.putText(
            frame,
            f"{name:>7}: {value:.2f}",
            org=(10, 30 + row * 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 255, 255),  # BGR: yellow
            thickness=2,
        )


def main() -> None:
    config = AppConfig()
    print(__doc__)
    with Camera(config.camera) as camera, HandTracker(
        config.hand, mirrored=config.camera.mirror
    ) as tracker:
        try:
            while True:
                frame = camera.read()
                if frame is None:
                    continue

                hands = tracker.detect(frame)
                values = measure(hands[0], frame) if hands else None
                if values is not None:
                    draw_hand(frame, hands[0])
                    draw_values(frame, values)

                cv2.imshow(WINDOW_NAME, frame)
                # Lower-case the key so the tool also works with Caps Lock on.
                key = ord(chr(cv2.waitKey(1) & 0xFF).lower())
                if key in EXIT_KEYS:
                    break
                if key in POSE_KEYS:
                    if values is None:
                        print("no hand visible, try again")
                    else:
                        print(f"{POSE_KEYS[key]:<10}  {format_values(values)}")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
