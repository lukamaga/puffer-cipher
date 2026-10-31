"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy generator for reproducible tests (never for secrets)."""
    return np.random.default_rng(20261031)
