"""Entry point. Run with:  python -m spider_vision

Owns the things that belong to the whole application rather than to one
layer: logging setup, the main loop, the window and keyboard handling.
"""

import logging

import cv2

from spider_vision.camera import Camera, CameraError, Frame
from spider_vision.config import AppConfig
from spider_vision.effects import draw_hand
from spider_vision.fps import FpsCounter
from spider_vision.vision import HandTracker, VisionError

logger = logging.getLogger(__name__)

WINDOW_NAME = "Spider Vision"
ESC_KEY = 27
EXIT_KEYS = {ESC_KEY, ord("q"), ord("Q")}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def draw_fps(frame: Frame, fps: float) -> None:
    """Write the FPS value in the top-left corner of the frame (in place)."""
    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 255, 0),  # BGR: green
        thickness=2,
    )


def window_was_closed() -> bool:
    """True if the user closed the window with its X button."""
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def run(config: AppConfig) -> int:
    """Run the main loop. Returns the process exit code."""
    fps = FpsCounter()
    failures = 0

    with Camera(config.camera) as camera, HandTracker(
        config.hand, mirrored=config.camera.mirror
    ) as tracker:
        try:
            while True:
                frame = camera.read()
                if frame is None:
                    failures += 1
                    if failures >= config.camera.max_read_failures:
                        logger.error("No frames from camera, stopping.")
                        return 1
                    continue
                failures = 0

                for hand in tracker.detect(frame):
                    draw_hand(frame, hand)

                fps.tick()
                draw_fps(frame, fps.fps)
                cv2.imshow(WINDOW_NAME, frame)

                # waitKey also lets OpenCV redraw the window; 1 ms keeps it fast.
                key = cv2.waitKey(1) & 0xFF
                if key in EXIT_KEYS or window_was_closed():
                    logger.info("Exit requested.")
                    return 0
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    config = AppConfig()
    setup_logging(config.log_level)
    logger.info("Starting Spider Vision. Press ESC or Q to quit.")
    try:
        return run(config)
    except (CameraError, VisionError) as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted with Ctrl+C.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
