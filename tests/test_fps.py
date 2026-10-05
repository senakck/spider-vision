"""Tests for FpsCounter, using a fake clock so no real time passes."""

import pytest

from spider_vision.fps import FpsCounter


class FakeClock:
    """A clock that only moves when the test tells it to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def run_frames(counter: FpsCounter, clock: FakeClock, count: int, fps: float) -> None:
    for _ in range(count):
        counter.tick()
        clock.advance(1 / fps)


def test_fps_is_zero_before_two_frames() -> None:
    clock = FakeClock()
    counter = FpsCounter(clock=clock)
    assert counter.fps == 0.0
    counter.tick()
    assert counter.fps == 0.0


def test_steady_frame_rate_is_measured() -> None:
    clock = FakeClock()
    counter = FpsCounter(window=30, clock=clock)
    run_frames(counter, clock, count=60, fps=30)
    assert counter.fps == pytest.approx(30)


def test_only_recent_frames_count() -> None:
    clock = FakeClock()
    counter = FpsCounter(window=10, clock=clock)
    run_frames(counter, clock, count=20, fps=60)  # fast at first
    run_frames(counter, clock, count=20, fps=15)  # then slow
    assert counter.fps == pytest.approx(15)


def test_invalid_window_is_rejected() -> None:
    with pytest.raises(ValueError):
        FpsCounter(window=0)
