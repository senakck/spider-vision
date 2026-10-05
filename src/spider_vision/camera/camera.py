"""Webcam access.

This module only opens the camera and delivers frames. It knows nothing about
MediaPipe, gestures or drawing.
"""

import logging
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np
import numpy.typing as npt

from spider_vision.config import CameraConfig

logger = logging.getLogger(__name__)

Frame = npt.NDArray[np.uint8]
"""A BGR image as delivered by OpenCV, shape (height, width, 3)."""


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened."""


class VideoSource(Protocol):
    """The part of ``cv2.VideoCapture`` that ``Camera`` uses.

    Declaring it lets tests pass a fake source instead of a real webcam.
    """

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, Frame | None]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def get(self, prop_id: int) -> float: ...
    def release(self) -> None: ...


class Camera:
    """Opens a webcam and reads frames from it.

    Use it as a context manager so the camera is always released:

        with Camera(config.camera) as camera:
            frame = camera.read()
    """

    def __init__(
        self,
        config: CameraConfig,
        source_factory: Callable[[int], VideoSource] = cv2.VideoCapture,
    ) -> None:
        self._config = config
        self._source_factory = source_factory
        self._source: VideoSource | None = None

    def open(self) -> None:
        source = self._source_factory(self._config.index)
        if not source.isOpened():
            source.release()
            raise CameraError(
                f"Could not open camera {self._config.index}. "
                "Is it used by another app, or is camera access disabled?"
            )

        source.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        source.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._source = source

        # The camera may not support the requested size, so log what we got.
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %d opened at %dx%d", self._config.index, width, height)

    def read(self) -> Frame | None:
        """Return the next frame, or None if the camera did not deliver one."""
        if self._source is None:
            raise CameraError("Camera is not open. Call open() first.")

        ok, frame = self._source.read()
        if not ok or frame is None:
            return None
        if self._config.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self._source is not None:
            self._source.release()
            self._source = None
            logger.info("Camera %d released", self._config.index)

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
