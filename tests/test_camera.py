"""Tests for the camera layer, using a fake video source instead of a webcam."""

import numpy as np
import pytest

from spider_vision.camera import Camera, CameraError, Frame
from spider_vision.config import CameraConfig


class FakeSource:
    """Pretends to be cv2.VideoCapture and records what happened to it."""

    def __init__(self, opened: bool = True, frame: Frame | None = None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Frame | None]:
        return (self.frame is not None, self.frame)

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return 0.0

    def release(self) -> None:
        self.released = True


def make_frame() -> Frame:
    """A 1x2 image: left pixel black, right pixel white."""
    frame = np.zeros((1, 2, 3), dtype=np.uint8)
    frame[0, 1] = 255
    return frame


def test_mirror_flips_frame_horizontally() -> None:
    source = FakeSource(frame=make_frame())
    with Camera(CameraConfig(mirror=True), lambda _: source) as camera:
        frame = camera.read()
    assert frame is not None
    assert frame[0, 0, 0] == 255 and frame[0, 1, 0] == 0


def test_frame_unchanged_without_mirror() -> None:
    source = FakeSource(frame=make_frame())
    with Camera(CameraConfig(mirror=False), lambda _: source) as camera:
        frame = camera.read()
    assert frame is not None
    assert frame[0, 0, 0] == 0 and frame[0, 1, 0] == 255


def test_read_returns_none_when_no_frame() -> None:
    source = FakeSource(frame=None)
    with Camera(CameraConfig(), lambda _: source) as camera:
        assert camera.read() is None


def test_open_failure_raises_and_releases_source() -> None:
    source = FakeSource(opened=False)
    with pytest.raises(CameraError):
        Camera(CameraConfig(), lambda _: source).open()
    assert source.released


def test_camera_is_released_even_after_error() -> None:
    source = FakeSource(frame=make_frame())
    with pytest.raises(ZeroDivisionError):
        with Camera(CameraConfig(), lambda _: source):
            1 / 0
    assert source.released


def test_read_before_open_raises() -> None:
    with pytest.raises(CameraError):
        Camera(CameraConfig(), lambda _: FakeSource()).read()
