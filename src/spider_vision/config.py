"""Central configuration for Spider Vision.

All tunable values (thresholds, sizes, timings) live here instead of being
hard-coded in the layers that use them. Each milestone adds its own section
as a separate frozen dataclass and attaches it to ``AppConfig``.

Dataclasses are frozen so that no layer can change a setting at runtime by
accident.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Top-level application settings."""

    log_level: str = "INFO"
