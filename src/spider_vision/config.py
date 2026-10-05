"""Central configuration for Spider Vision.

All tunable values (thresholds, sizes, timings) live here instead of being
hard-coded in the layers that use them. Each milestone adds its own section
as a separate frozen dataclass and attaches it to ``AppConfig``.

Dataclasses are frozen so that no layer can change a setting at runtime by
accident.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    """Webcam settings (M1)."""

    index: int = 0
    """Which webcam to open. 0 is usually the built-in camera."""

    width: int = 640
    height: int = 480
    """Requested resolution. The camera may pick the closest size it supports."""

    mirror: bool = True
    """Flip frames horizontally so the view behaves like a mirror."""

    max_read_failures: int = 30
    """Stop after this many frames in a row fail to arrive (camera unplugged)."""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"camera index must be >= 0, got {self.index}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"camera resolution must be positive, got {self.width}x{self.height}"
            )
        if self.max_read_failures < 1:
            raise ValueError(
                f"max_read_failures must be >= 1, got {self.max_read_failures}"
            )


@dataclass(frozen=True)
class HandTrackingConfig:
    """MediaPipe hand landmarker settings (M2)."""

    model_path: Path = Path("models/hand_landmarker.task")
    """Relative to the project root. Download with scripts/download_models.py."""

    max_hands: int = 1
    """Each extra hand costs processing time. Raise when a feature needs two."""

    min_detection_confidence: float = 0.5
    """How sure the model must be to report a new hand."""

    min_presence_confidence: float = 0.5
    """How sure the model must be that a tracked hand is still there."""

    min_tracking_confidence: float = 0.5
    """Below this, tracking is dropped and the hand is searched for again."""

    def __post_init__(self) -> None:
        if self.max_hands < 1:
            raise ValueError(f"max_hands must be >= 1, got {self.max_hands}")
        for name in (
            "min_detection_confidence",
            "min_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class AppConfig:
    """Top-level application settings."""

    log_level: str = "INFO"
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand: HandTrackingConfig = field(default_factory=HandTrackingConfig)
