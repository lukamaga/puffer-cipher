"""LedDetector: brightness definition and the adaptive two-cluster threshold."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from puffer.liveness import LedDetector, roi_brightness
from puffer.types import Frame, LightMode


def two_level_signal(
    rng: np.random.Generator,
    n: int,
    *,
    off: float = 60.0,
    on: float = 220.0,
    noise: float = 4.0,
    drift: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Random Manchester-like on/off states and their noisy brightness."""
    bits = rng.integers(0, 2, size=n // 2).astype(bool)
    states = np.empty(n, dtype=bool)
    states[0::2], states[1::2] = bits, ~bits
    b = np.where(states, on, off) + rng.normal(0.0, noise, size=n)
    if drift is not None:
        b = b + drift
    return states, b


def accuracy(det: LedDetector, states: npt.NDArray[np.bool_], b: npt.NDArray[np.float64]) -> float:
    obs = [det.update(i / 30, float(v)) for i, v in enumerate(b)]
    hits = [o.on == s for o, s in zip(obs, states, strict=True)]
    return float(np.mean(hits[10:]))


def test_thresholding_with_noise(rng):
    states, b = two_level_signal(rng, 3000)
    det = LedDetector()
    assert accuracy(det, states, b) > 0.999
    lo, hi = det.levels or (0.0, 0.0)
    assert lo == pytest.approx(60.0, abs=2.0)
    assert hi == pytest.approx(220.0, abs=2.0)
    assert det.threshold == pytest.approx(140.0, abs=2.0)


def test_tracks_slow_brightness_drift(rng):
    n = 6000  # 200 s at 30 fps
    t = np.arange(n) / 30
    drift = 50.0 * np.sin(2 * np.pi * t / 60.0) + 0.3 * t  # lamps dimming, sun coming up
    states, b = two_level_signal(rng, n, off=50.0, on=150.0, noise=3.0, drift=drift)
    det = LedDetector()
    assert accuracy(det, states, b) > 0.995
    # the levels followed the light: off-level now near 50 + drift at the end
    lo, _ = det.levels or (0.0, 0.0)
    assert lo == pytest.approx(50.0 + drift[-1], abs=8.0)


def test_level_jump_does_not_strand_a_cluster(rng):
    """Lights out: the on-level drops below the old midpoint and must be re-learnt."""
    s1, b1 = two_level_signal(rng, 600, off=60.0, on=250.0)
    s2, b2 = two_level_signal(rng, 900, off=20.0, on=120.0)
    det = LedDetector()
    accuracy(det, s1, b1)
    obs = [det.update(i / 30, float(v)) for i, v in enumerate(b2)]
    late = [o.on == s for o, s in zip(obs[300:], s2[300:], strict=True)]
    assert np.mean(late) > 0.99


def test_constant_led_is_never_confident(rng):
    det = LedDetector()
    obs = [det.update(i / 30, 80.0 + rng.normal(0, 3.0)) for i in range(600)]
    assert not any(o.confident for o in obs)


def test_covered_led_loses_confidence(rng):
    _states, b = two_level_signal(rng, 600)
    det = LedDetector()
    first = [det.update(i / 30, float(v)) for i, v in enumerate(b)]
    assert np.mean([o.confident for o in first[20:]]) > 0.95
    covered = [det.update(20 + i / 30, 70.0 + rng.normal(0, 3.0)) for i in range(300)]
    assert not any(o.confident for o in covered[-150:])
    assert det.contrast < det.min_contrast


def test_hysteresis_holds_state_inside_the_band():
    det = LedDetector(hysteresis=0.3)
    for i in range(40):  # learn levels 0 / 100
        det.update(i, 100.0 if i % 2 else 0.0)
    assert det.update(100, 100.0).on
    # just below the midpoint but inside the band: stays on
    assert det.update(101, 47.0).on
    assert det.update(102, 47.0).on
    assert not det.update(103, 30.0).on  # clearly below: switches
    assert not det.update(104, 53.0).on  # just above, inside the band: stays off


def test_transition_frames_are_not_confident():
    det = LedDetector()
    for i in range(60):
        det.update(i, 200.0 if i % 2 else 40.0)
    half_exposed = det.update(61, 120.0)
    assert not half_exposed.confident
    assert det.update(62, 200.0).confident


def colour_frame(value_bgr: tuple[int, int, int], *, grey: bool = False) -> Frame:
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[5:10, 10:20] = value_bgr
    if grey:
        img[...] = img.max(axis=2, keepdims=True)
    return Frame(0, 0.0, 0.0, img, LightMode.NIGHT if grey else LightMode.DAY)


def test_brightness_is_the_max_channel():
    roi = (10, 5, 10, 5)
    red_led = colour_frame((30, 30, 250))
    assert roi_brightness(red_led.image, roi) == pytest.approx(250.0)
    # a luminance average would dilute a red LED to ~95; V keeps its full contrast
    assert roi_brightness(colour_frame((0, 0, 0)).image, roi) == 0.0
    assert roi_brightness(colour_frame((200, 200, 200), grey=True).image, roi) == 200.0
    assert roi_brightness(red_led.image, (8, 5, 4, 5)) == pytest.approx(125.0)
    single_channel = np.full((4, 4), 77, dtype=np.uint8)  # raw IR sensor plane
    assert roi_brightness(single_channel, (0, 0, 4, 4)) == 77.0


def test_observe_uses_the_roi_and_requires_one():
    det = LedDetector()
    with pytest.raises(RuntimeError, match="ROI"):
        det.observe(colour_frame((0, 0, 255)))
    det.roi = (10, 5, 10, 5)
    obs = det.observe(colour_frame((0, 0, 255)))
    assert obs.brightness == pytest.approx(255.0)
    assert obs.t == 0.0


def test_moving_the_roi_resets_levels():
    det = LedDetector((0, 0, 2, 2))
    for i in range(20):
        det.update(i, 10.0 * (i % 2))
    assert det.levels is not None
    det.roi = (1, 1, 2, 2)
    assert det.levels is None
    assert det.threshold is None
    assert det.contrast == 0.0


@pytest.mark.parametrize("roi", [(-1, 0, 3, 3), (0, 0, 0, 3), (25, 0, 10, 3), (0, 18, 3, 3)])
def test_invalid_rois(roi):
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="ROI"):
        roi_brightness(img, roi)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"adapt_rate": 0.0}, "adapt_rate"),
        ({"hysteresis": 1.0}, "hysteresis"),
        ({"confidence_margin": -0.1}, "confidence_margin"),
        ({"min_contrast": -1.0}, "min_contrast"),
        ({"stale_frames": 0}, "stale_frames"),
    ],
)
def test_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LedDetector(**kwargs)


def test_non_finite_brightness_rejected():
    with pytest.raises(ValueError, match="finite"):
        LedDetector().update(0.0, float("nan"))
