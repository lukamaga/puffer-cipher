"""calibrate_roi: find the LED, ignore a decoy that blinks independently."""

from __future__ import annotations

import numpy as np
import pytest

from puffer.liveness import (
    CalibrationAccumulator,
    ChallengeSchedule,
    calibrate_roi,
    commanded_states,
    lag_grid,
    roi_brightness,
)
from puffer.types import LightMode

from .helpers import Scene, overlap, render, seeded_schedule


def roi_center(roi: tuple[int, int, int, int]) -> tuple[float, float]:
    return roi[0] + roi[2] / 2, roi[1] + roi[3] / 2


@pytest.mark.parametrize("mode", [LightMode.DAY, LightMode.NIGHT])
@pytest.mark.parametrize("latency", [0.0, 0.3])
def test_finds_true_led_and_ignores_decoy(mode, latency):
    challenge, decoy = seeded_schedule(1), seeded_schedule(2)
    scene = Scene(mode=mode, noise_sigma=5.0 if mode is LightMode.NIGHT else 3.0)
    frames = list(
        render(
            challenge.state_at,
            rng=np.random.default_rng(3),
            duration_s=4.0,
            latency_s=latency,
            scene=scene,
            decoy=decoy.state_at,
        )
    )
    result = calibrate_roi(frames, challenge)
    assert result.score > 0.95
    # the ROI lies (almost) entirely on the LED disc, nowhere near the decoy
    assert overlap(result.roi, scene.led_roi) > 0.5 or overlap(scene.led_roi, result.roi) > 0.5
    cx, cy = roi_center(result.roi)
    assert np.hypot(cx - scene.led_center[0], cy - scene.led_center[1]) <= scene.led_radius
    assert result.lag_s == pytest.approx(latency, abs=1 / 30)


def test_decoy_scores_low_against_the_challenge():
    """The decoy blinks as much as the LED; only the *fresh challenge* tells them apart."""
    challenge, decoy = seeded_schedule(4), seeded_schedule(5)
    scene = Scene(decoy_center=(26, 72))
    frames = list(
        render(
            challenge.state_at, rng=np.random.default_rng(6), duration_s=4.0, decoy=decoy.state_at
        )
    )
    decoy_roi = (20, 68, 8, 8)
    against_decoy = calibrate_roi(frames, decoy)
    assert overlap(against_decoy.roi, decoy_roi) > 0.5  # it is a perfectly good blinker...
    result = calibrate_roi(frames, challenge)  # ...but not the one we asked for
    assert overlap(result.roi, decoy_roi) == 0.0
    assert overlap(result.roi, scene.led_roi) > 0.0

    acc = CalibrationAccumulator()
    for f in frames:
        acc.add(f)
    series = acc.roi_series(decoy_roi)
    times = acc.times
    correlations = [
        np.corrcoef(series, commanded_states(challenge, times - lag))[0, 1]
        for lag in lag_grid(0.5, 1 / 120)
    ]
    assert max(np.abs(correlations)) < 0.4 < result.score


def test_no_led_gives_a_low_score():
    challenge, decoy = seeded_schedule(7), seeded_schedule(8)
    frames = list(
        render(
            challenge.state_at,
            rng=np.random.default_rng(9),
            duration_s=4.0,
            decoy=decoy.state_at,
            covered=lambda t: True,
        )
    )
    assert calibrate_roi(frames, challenge).score < 0.5


def test_replayed_led_gives_a_low_score():
    old, fresh = seeded_schedule(10), seeded_schedule(11)
    frames = list(render(old.state_at, rng=np.random.default_rng(12), duration_s=4.0))
    assert calibrate_roi(frames, fresh).score < 0.6


def test_accumulator_sliding_window_and_roi_series():
    challenge = seeded_schedule(13)
    frames = list(render(challenge.state_at, rng=np.random.default_rng(14), duration_s=3.0))
    acc = CalibrationAccumulator(cell=16, resolution=4, keep_s=1.0)
    for f in frames:
        acc.add(f)
    assert acc.span <= 1.0
    assert acc.times[0] >= frames[-1].timestamp - 1.0
    kept = frames[-len(acc) :]
    roi = (96, 20, 8, 4)
    expected = [roi_brightness(f.image, roi) for f in kept]
    assert np.allclose(acc.roi_series(roi), expected, rtol=0.0, atol=1e-9)
    with pytest.raises(ValueError, match="aligned"):
        acc.roi_series((97, 20, 8, 4))
    acc.clear()
    assert len(acc) == 0
    assert acc.span == 0.0


def test_accumulator_rejects_bad_input():
    challenge = seeded_schedule(15)
    small = next(
        render(
            challenge.state_at,
            rng=np.random.default_rng(0),
            scene=Scene(
                width=8, height=8, led_center=(4, 4), led_radius=2, decoy_center=None, fish=False
            ),
        )
    )
    with pytest.raises(ValueError, match="smaller than one cell"):
        CalibrationAccumulator(cell=16).add(small)
    acc = CalibrationAccumulator()
    with pytest.raises(ValueError, match="at least"):
        acc.solve(challenge)
    with pytest.raises(ValueError, match="divide"):
        CalibrationAccumulator(cell=10, resolution=4)
    frames = render(challenge.state_at, rng=np.random.default_rng(0), duration_s=0.2)
    acc.add(next(frames))
    with pytest.raises(ValueError, match="size changed"):
        acc.add(
            next(render(challenge.state_at, rng=np.random.default_rng(0), scene=Scene(width=64)))
        )


def test_pattern_without_transitions_scores_zero():
    """Before t0 the LED is simply off: nothing to correlate with, nothing found."""
    late = ChallengeSchedule(t0=1e6, random_bytes=np.random.default_rng(17).bytes)
    frames = list(
        render(seeded_schedule(18).state_at, rng=np.random.default_rng(19), duration_s=2.0)
    )
    assert calibrate_roi(frames, late).score == 0.0


def test_duplicate_timestamps_are_rejected():
    frame = next(render(seeded_schedule(20).state_at, rng=np.random.default_rng(21)))
    with pytest.raises(ValueError, match="distinct timestamps"):
        calibrate_roi([frame] * 10, seeded_schedule(20))
    acc = CalibrationAccumulator()
    with pytest.raises(ValueError, match="no frames"):
        acc.roi_series((0, 0, 4, 4))


def test_lag_grid_and_commanded_states():
    assert lag_grid(0.5, 0.1).tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert lag_grid(0.0, 0.1).tolist() == [0.0]
    with pytest.raises(ValueError, match="lag step"):
        lag_grid(0.5, 0.0)
    with pytest.raises(ValueError, match="max_lag_s"):
        lag_grid(-0.5, 0.1)

    class Blink:
        """A plain LedSchedule (not a ChallengeSchedule): on during odd seconds."""

        def state_at(self, t: float) -> bool:
            return int(t) % 2 == 1

    times = np.array([[0.5, 1.5], [2.5, 3.5]])
    assert commanded_states(Blink(), times).tolist() == [[False, True], [False, True]]
    sch = seeded_schedule(16)
    assert commanded_states(sch, times).tolist() == [
        [sch.state_at(float(t)) for t in row] for row in times
    ]
