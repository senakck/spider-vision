"""Tests for the configuration module. No camera required."""

import dataclasses

import pytest

from spider_vision.config import AppConfig


def test_default_log_level_is_info() -> None:
    assert AppConfig().log_level == "INFO"


def test_config_is_immutable() -> None:
    config = AppConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.log_level = "DEBUG"  # type: ignore[misc]


def test_values_can_be_overridden_at_creation() -> None:
    assert AppConfig(log_level="DEBUG").log_level == "DEBUG"
