"""Tests for the configuration module. No camera required."""

import dataclasses

import pytest

from spider_vision.config import AppConfig, CameraConfig


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
    [{"index": -1}, {"width": 0}, {"height": -480}],
)
def test_invalid_camera_settings_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)
