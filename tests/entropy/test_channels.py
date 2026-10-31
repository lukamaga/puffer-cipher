"""Sensor and fish channels, the rolling MCV estimate and the credit formula."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from puffer.entropy.channels import (
    FishChannel,
    RollingEstimate,
    SensorChannel,
    SensorChannelConfig,
    compute_credit,
)
from puffer.entropy.estimators import most_common_value

H, W = 48, 64


def noise(rng: np.random.Generator, h: int = H, w: int = W) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# -- sensor channel ------------------------------------------------------------


def test_first_frame_has_no_samples_then_temporal_differences(make_frame, rng):
    channel = SensorChannel(SensorChannelConfig(n_sites=500, site_seed=b"sites"))
    a, b = noise(rng), noise(rng)
    first = channel.process(make_frame(a, index=0))
    assert first.channel == "sensor"
    assert first.samples.dtype == np.uint8
    assert first.samples.size == 0
    second = channel.process(make_frame(b, index=1))
    ys, xs = channel.sites
    expected = ((b[ys, xs, 1].astype(int) - a[ys, xs, 1].astype(int)) % 256).astype(np.uint8)
    assert second.samples.dtype == np.uint8
    np.testing.assert_array_equal(second.samples, expected)
    assert second.sample_owner is None


def test_absorb_is_header_plus_raw_frame(make_frame, rng):
    image = noise(rng)
    out = SensorChannel().process(make_frame(image, index=1234, t=56.78))
    assert struct.unpack(">Qd", out.absorb[:16]) == (1234, 56.78)
    assert out.absorb[16:] == image.tobytes()
    assert (out.frame_index, out.t) == (1234, 56.78)


def test_sites_avoid_border_and_excluded_roi(make_frame, rng):
    roi = (10, 5, 20, 12)  # x, y, w, h
    channel = SensorChannel(
        SensorChannelConfig(n_sites=10_000, border=4, exclude_rois=[roi], site_seed=b"s")
    )
    channel.process(make_frame(noise(rng)))
    ys, xs = channel.sites
    assert np.all((ys >= 4) & (ys < H - 4))
    assert np.all((xs >= 4) & (xs < W - 4))
    in_roi = (xs >= 10) & (xs < 30) & (ys >= 5) & (ys < 17)
    assert not in_roi.any()
    # More sites requested than available: every valid pixel is used exactly once.
    assert len(set(zip(ys.tolist(), xs.tolist(), strict=True))) == ys.size
    assert ys.size == (H - 8) * (W - 8) - 20 * 12


def test_sites_are_reproducible_with_a_seed_and_random_without(make_frame, rng):
    frame = make_frame(noise(rng))

    def sites(config: SensorChannelConfig) -> list[int]:
        channel = SensorChannel(config)
        channel.process(frame)
        ys, xs = channel.sites
        return (ys * W + xs).tolist()

    assert sites(SensorChannelConfig(n_sites=100, site_seed=b"a")) == sites(
        SensorChannelConfig(n_sites=100, site_seed=b"a")
    )
    assert sites(SensorChannelConfig(n_sites=100, site_seed=b"a")) != sites(
        SensorChannelConfig(n_sites=100, site_seed=b"b")
    )
    assert sites(SensorChannelConfig(n_sites=100)) != sites(SensorChannelConfig(n_sites=100))


def test_frame_size_change_reinitialises_sites(make_frame, rng):
    channel = SensorChannel(SensorChannelConfig(n_sites=200, site_seed=b"x"))
    channel.process(make_frame(noise(rng)))
    assert channel.process(make_frame(noise(rng), index=1)).samples.size == 200
    out = channel.process(make_frame(noise(rng, 32, 40), index=2))
    assert out.samples.size == 0  # no previous frame of this size
    ys, xs = channel.sites
    assert np.all(ys < 32 - 8)
    assert np.all(xs < 40 - 8)
    assert channel.process(make_frame(noise(rng, 32, 40), index=3)).samples.size == 200


def test_static_scene_gives_zero_differences(make_frame, rng):
    image = noise(rng)
    channel = SensorChannel(SensorChannelConfig(n_sites=64, site_seed=b"x"))
    channel.process(make_frame(image))
    assert not channel.process(make_frame(image.copy(), index=1)).samples.any()


def test_no_valid_sites_is_an_error(make_frame, rng):
    channel = SensorChannel(SensorChannelConfig(exclude_rois=[(0, 0, W, H)]))
    with pytest.raises(ValueError, match="no sample sites"):
        channel.process(make_frame(noise(rng)))


def test_sensor_config_validation():
    with pytest.raises(ValueError, match="n_sites"):
        SensorChannelConfig(n_sites=0)
    with pytest.raises(ValueError, match="exclude_rois"):
        SensorChannelConfig(exclude_rois=[(0, 0, -1, 5)])


# -- fish channel --------------------------------------------------------------


@pytest.mark.parametrize(
    ("innovation", "symbol"),
    [
        ((0.0, 0.0), 0x00),
        ((0.125, -0.125), 0x1F),  # q = (1, -1): -1 & 0xF = 15
        ((2.0, 1.0), 0x08),  # q = (16, 8): 16 & 0xF = 0
        ((0.0625, 0.1875), 0x02),  # q = (0.5, 1.5) -> rounds half to even: (0, 2)
        ((-2.5, 3.0), 0xC8),  # q = (-20, 24) -> (12, 8)
    ],
)
def test_fish_innovation_quantisation(make_tracking, make_observation, innovation, symbol):
    result = make_tracking(1.0, observations=[make_observation(7, innovation, "f1")])
    out = FishChannel().process(result)
    assert out.samples.dtype == np.uint8
    assert out.samples.tolist() == [symbol]


def test_fish_absorb_owners_and_metadata(make_tracking, make_observation):
    obs = [make_observation(3, (0.5, 0.25), "f3"), make_observation(9, (-1.0, 0.0))]
    out = FishChannel().process(make_tracking(2.5, observations=obs, frame_index=77))
    assert (out.channel, out.frame_index, out.t) == ("fish", 77, 2.5)
    assert out.sample_owner == ("f3", "track-9")
    expected = b"".join(
        struct.pack(
            ">10d",
            o.track_id,
            o.cx,
            o.cy,
            o.vx,
            o.vy,
            o.heading,
            o.accel,
            o.turn_rate,
            *o.innovation,
        )
        for o in obs
    )
    assert out.absorb == expected
    assert out.samples.tolist() == [(4 << 4) | 2, (8 << 4) | 0]


def test_fish_without_observations(make_tracking):
    out = FishChannel().process(make_tracking(0.0))
    assert out.absorb == b""
    assert out.samples.size == 0
    assert out.sample_owner == ()


def test_non_finite_innovation_is_absorbed_but_never_sampled(make_tracking, make_observation):
    obs = [make_observation(1, (float("nan"), 0.0), "a"), make_observation(2, (0.125, 0.0), "b")]
    out = FishChannel().process(make_tracking(0.0, observations=obs))
    assert len(out.absorb) == 2 * 10 * 8
    assert out.samples.tolist() == [0x10]
    assert out.sample_owner == ("b",)


# -- rolling estimate and credit ---------------------------------------------


def test_rolling_estimate_tracks_the_last_window(rng):
    from puffer.entropy.channels import _min_entropy_estimate

    est = RollingEstimate(h_cap=4.0, window=4096, refresh_every=1)
    stream = rng.integers(0, 256, size=20_000, dtype=np.uint8)
    for chunk in np.array_split(stream, 37):  # uneven chunks wrap the ring buffer
        est.feed(chunk)
    last = stream[-4096:]
    assert est.h_est == pytest.approx(_min_entropy_estimate(last))
    assert est.h_est <= most_common_value(last).min_entropy
    assert est.samples_seen == stream.size


def test_rolling_estimate_sees_bit_level_structure(rng):
    """Small symmetric differences look rich per byte but are poor per bit."""
    from puffer.entropy.estimators import collision, to_bitstring

    noise = (np.round(rng.normal(0, 2.0, 4096)).astype(np.int64) % 256).astype(np.uint8)
    est = RollingEstimate(h_cap=4.0)
    assert est.feed(noise)
    bitwise = 8 * collision(to_bitstring(noise, 8)).min_entropy
    assert est.h_est <= bitwise + 1e-9
    assert est.h_est < most_common_value(noise).min_entropy - 0.5


def test_rolling_estimate_waits_for_min_samples_then_refreshes_every_n_frames(rng):
    est = RollingEstimate(h_cap=3.0, refresh_every=30, min_samples=1024)
    refreshed = [est.feed(rng.integers(0, 256, 100, dtype=np.uint8)) for _ in range(80)]
    first = refreshed.index(True)
    assert first == 10  # 11th frame brings the window to 1100 samples
    assert [i for i, r in enumerate(refreshed) if r] == [10, 40, 70]
    assert est.ready
    assert 5.0 < est.h_est <= 8.0  # uncapped estimate
    assert est.credit(100) == pytest.approx(100 * 3.0 * 0.5)  # capped at H_cap, times 0.5


def test_rolling_estimate_is_zero_before_ready_and_for_constant_data():
    est = RollingEstimate(h_cap=4.0, min_samples=1024)
    assert not est.feed(np.zeros(1000, np.uint8))
    assert (est.ready, est.h_est, est.credit(1000)) == (False, 0.0, 0.0)
    assert est.feed(np.zeros(100, np.uint8))
    assert est.h_est == 0.0


def test_compute_credit():
    assert compute_credit(4096, 3.0, 4.0) == 4096 * 3.0 * 0.5
    assert compute_credit(4096, 6.5, 4.0) == 4096 * 4.0 * 0.5  # capped
    assert compute_credit(100, 2.0, 3.0, multiplier=0.5) == 50.0  # DEGRADED
    assert compute_credit(100, -1.0, 3.0) == 0.0
    assert compute_credit(0, 8.0, 3.0) == 0.0
    with pytest.raises(ValueError, match="multiplier"):
        compute_credit(10, 2.0, 3.0, multiplier=2.0)
    with pytest.raises(ValueError, match="finite"):
        compute_credit(10, float("nan"), 3.0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_credit(-1, 2.0, 3.0)


def test_rolling_estimate_sees_through_a_deterministic_cycle():
    """MCV alone credits 0,1,2,0,1,2,... with ~1.5 bits/sample; predictors see 0."""
    from puffer.entropy.channels import RollingEstimate
    from puffer.entropy.estimators import most_common_value

    cycle = (np.arange(4096) % 3).astype(np.uint8)
    assert most_common_value(cycle).min_entropy > 1.4
    est = RollingEstimate(h_cap=3.0)
    assert est.feed(cycle)
    assert est.h_est < 0.05
    assert est.credit(4096) < 1.0


def test_rolling_estimate_keeps_credit_for_noise(rng):
    """Gaussian pixel differences still earn credit, at their bit-level rate.

    For differences with sigma 2.8 stored mod 256, MCV sees ~2.8 bits per
    sample, but the high bits copy the sign, and the bitstring Collision
    estimate settles around 1.2 bits per sample. Credit follows the lower one.
    """
    from puffer.entropy.channels import RollingEstimate

    noise = (np.round(rng.normal(0, 2.8, 4096)).astype(np.int64) % 256).astype(np.uint8)
    est = RollingEstimate(h_cap=4.0)
    assert est.feed(noise)
    assert 0.8 < est.h_est < 2.0
    assert est.credit(4096) > 1000
