"""MediaPipe hand tracking.

This is the only module that talks to MediaPipe's hand landmarker. It turns
MediaPipe's results into the ``Hand`` domain model, so no other layer depends
on MediaPipe's types.
"""

import logging
import time
from collections.abc import Callable
from types import TracebackType
from typing import Self

import cv2
import mediapipe as mp

from spider_vision.camera import Frame
from spider_vision.config import HandTrackingConfig
from spider_vision.vision.landmarks import Hand, Handedness, Landmark

logger = logging.getLogger(__name__)

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult


class VisionError(RuntimeError):
    """Raised when a vision model cannot be loaded or used."""


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class MonotonicTimestamp:
    """Produces strictly increasing millisecond timestamps.

    MediaPipe's VIDEO mode rejects a frame whose timestamp is not larger than
    the previous one, which can happen when two frames arrive within the same
    millisecond.
    """

    def __init__(self, clock_ms: Callable[[], int] = _monotonic_ms) -> None:
        self._clock_ms = clock_ms
        self._last = -1

    def next(self) -> int:
        now = max(self._clock_ms(), self._last + 1)
        self._last = now
        return now


def hands_from_result(result: HandLandmarkerResult) -> list[Hand]:
    """Convert a MediaPipe result into domain ``Hand`` objects."""
    hands: list[Hand] = []
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        best = categories[0]
        hands.append(
            Hand(
                landmarks=tuple(Landmark(p.x, p.y, p.z) for p in landmarks),
                handedness=Handedness(best.category_name),
                score=best.score,
            )
        )
    return hands


class HandTracker:
    """Detects and tracks hands in a stream of frames.

    Use it as a context manager so the model is always released:

        with HandTracker(config.hand) as tracker:
            hands = tracker.detect(frame)
    """

    def __init__(
        self,
        config: HandTrackingConfig,
        timestamps: MonotonicTimestamp | None = None,
    ) -> None:
        self._config = config
        self._timestamps = timestamps or MonotonicTimestamp()
        self._landmarker: mp.tasks.vision.HandLandmarker | None = None

    def open(self) -> None:
        model_path = self._config.model_path
        if not model_path.is_file():
            raise VisionError(
                f"Hand model not found at '{model_path}'. "
                "Run: python scripts/download_models.py"
            )

        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._config.max_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_hand_presence_confidence=self._config.min_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info("Hand tracker ready (max %d hand(s))", self._config.max_hands)

    def detect(self, frame: Frame) -> list[Hand]:
        """Return the hands found in a BGR frame (empty list if none)."""
        if self._landmarker is None:
            raise VisionError("Hand tracker is not open. Call open() first.")

        # OpenCV delivers BGR, MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._timestamps.next())
        return hands_from_result(result)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("Hand tracker closed")

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
