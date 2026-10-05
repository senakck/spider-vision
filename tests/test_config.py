"""Tests for the configuration module. No camera required."""

import dataclasses

import pytest

from spider_vision.config import AppConfig, CameraConfig, HandTrackingConfig


def test_default_log_level_is_info() -> None:
    assert AppConfig().log_level == "INFO"


def test_config_is_immutable() -> None:
    config = AppConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.log_level = "DEBUG"  # type: ignore[misc]


def test_values_can_be_overridden_at_creation() -> None:
    assert AppConfig(log_level="DEBUG").log_level == "DEBUG"


def test_camera_defaults_match_tested_webcam() -> None:
    camera = AppConfig().camera
    assert (camera.index, camera.width, camera.height) == (0, 640, 480)


@pytest.mark.parametrize(
    "kwargs",
    [{"index": -1}, {"width": 0}, {"height": -480}, {"max_read_failures": 0}],
)
def test_invalid_camera_settings_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)


def test_hand_tracking_tracks_one_hand_by_default() -> None:
    assert AppConfig().hand.max_hands == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_hands": 0},
        {"min_detection_confidence": -0.1},
        {"min_presence_confidence": 1.5},
        {"min_tracking_confidence": 2.0},
    ],
)
def test_invalid_hand_tracking_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        HandTrackingConfig(**kwargs)
