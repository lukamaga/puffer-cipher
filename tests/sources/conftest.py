"""Fixtures and fakes shared by the frame-source tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from puffer.sources.synthetic import SyntheticConfig


class PeriodicLed:
    """Deterministic stand-in for the LED challenge: on/off every ``period`` seconds."""

    def __init__(self, period: float = 0.3) -> None:
        self.period = period

    def state_at(self, t: float) -> bool:
        return int(np.floor(t / self.period)) % 2 == 0


@pytest.fixture
def periodic_led() -> Callable[[float], PeriodicLed]:
    """Factory for :class:`PeriodicLed` schedules."""
    return PeriodicLed


@pytest.fixture
def quiet_config() -> Callable[..., SyntheticConfig]:
    """Factory for a static, noise-free tank (only what a test enables moves)."""

    def make(**overrides: Any) -> SyntheticConfig:
        base: dict[str, Any] = {
            "seed": 7,
            "n_fish": 0,
            "caustics": False,
            "bubbles": False,
            "noise_sigma_day": 0.0,
            "noise_sigma_night": 0.0,
            "shot_noise": False,
        }
        base.update(overrides)
        return SyntheticConfig(**base)

    return make
