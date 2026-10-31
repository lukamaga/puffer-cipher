"""Tests for the camera sensor noise model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from puffer.sources._sensor import SensorModel, _normal_quantiles


def _model(seed: int = 1, **kwargs: Any) -> SensorModel:
    params: dict[str, Any] = {"read_sigma": 2.0, "shot_gain": 0.05}
    params.update(kwargs)
    return SensorModel(np.random.default_rng(seed), **params)


def test_quantile_table_is_a_unit_normal() -> None:
    q = _normal_quantiles()
    assert q.shape == (65536,)
    assert abs(float(q.mean())) < 1e-6
    assert float(q.std()) == pytest.approx(1.0, abs=1e-6)
    assert 4.0 < float(np.max(q)) < 4.5  # tails truncated at ~4.3 sigma


@pytest.mark.parametrize(
    ("level", "read_sigma", "gain"), [(100, 2.0, 0.05), (40, 6.0, 0.15), (200, 1.0, 0.0)]
)
def test_noise_level_matches_model(level: int, read_sigma: float, gain: float) -> None:
    model = _model(read_sigma=read_sigma, shot_gain=gain)
    flat = np.full((200, 300, 3), level, np.uint8)
    noisy = model.apply(flat).astype(float)
    expected = math.sqrt(read_sigma**2 + gain * level + 1.0 / 12.0)  # + rounding
    assert noisy.std() == pytest.approx(expected, rel=0.03)
    assert noisy.mean() == pytest.approx(level, abs=0.05)
    assert model.expected_sigma(level) == pytest.approx(math.sqrt(read_sigma**2 + gain * level))


def test_noise_is_gaussian_shaped() -> None:
    model = _model(read_sigma=10.0, shot_gain=0.0)
    noisy = model.apply(np.full((300, 300), 128, np.uint8)).astype(float) - 128.0
    for k in (1, 2, 3):
        # Integer outputs: |x| <= 10k  <=>  |noise| < 10k + 0.5 before rounding.
        expected = math.erf((k + 0.05) / math.sqrt(2.0))
        assert np.mean(np.abs(noisy) <= 10.0 * k) == pytest.approx(expected, abs=0.006)


def test_shot_noise_grows_with_intensity() -> None:
    model = _model(read_sigma=1.0, shot_gain=0.2)
    dark = model.apply(np.full((100, 100), 20, np.uint8)).std()
    bright = model.apply(np.full((100, 100), 220, np.uint8)).std()
    assert bright > 2.0 * dark


def test_shot_noise_can_be_disabled() -> None:
    model = _model(read_sigma=2.0, shot_gain=0.5, shot_noise=False)
    assert model.apply(np.full((200, 200), 200, np.uint8)).std() == pytest.approx(
        math.sqrt(4 + 1 / 12), rel=0.05
    )


def test_same_seed_same_noise_and_shapes_preserved() -> None:
    image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    a, b = _model(seed=5).apply(image), _model(seed=5).apply(image)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _model(seed=6).apply(image))
    gray = _model().apply(image[..., 0])
    assert gray.shape == (48, 64)
    assert gray.dtype == np.uint8


def test_saturation_instead_of_wraparound() -> None:
    model = _model(read_sigma=20.0, shot_gain=0.0)
    assert np.min(model.apply(np.full((50, 50), 250, np.uint8))) > 150
    assert np.max(model.apply(np.full((50, 50), 3, np.uint8))) < 100


def test_zero_noise_is_identity_and_jpeg_round_trip() -> None:
    image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    assert np.array_equal(_model(read_sigma=0.0, shot_gain=0.0).apply(image), image)
    jpeg = _model(read_sigma=0.0, shot_gain=0.0, jpeg_quality=60).apply(image)
    assert jpeg.shape == image.shape
    assert not np.array_equal(jpeg, image)
    gray = _model(read_sigma=0.0, shot_gain=0.0, jpeg_quality=60).apply(image[..., 1])
    assert gray.shape == (48, 64)


def test_rejects_negative_parameters() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        _model(read_sigma=-1.0)
