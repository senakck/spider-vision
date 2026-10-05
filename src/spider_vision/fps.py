"""Frames-per-second measurement for the main loop."""

import time
from collections import deque
from collections.abc import Callable


class FpsCounter:
    """Measures FPS as the average over the most recent frames.

    Averaging over a window keeps the displayed number stable instead of
    jumping on every frame.
    """

    def __init__(
        self,
        window: int = 30,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._clock = clock
        # N frames are N intervals, which needs N + 1 timestamps.
        self._timestamps: deque[float] = deque(maxlen=window + 1)

    def tick(self) -> None:
        """Call once per processed frame."""
        self._timestamps.append(self._clock())

    @property
    def fps(self) -> float:
        """Average FPS over the window, or 0.0 before two frames are seen."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
