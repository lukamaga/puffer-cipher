"""End-to-end wiring of the entropy core, as the engine is expected to drive it."""

from __future__ import annotations

import numpy as np

from puffer.entropy import (
    DEFAULT_H_CAP,
    EntropyPool,
    FishChannel,
    HealthMonitor,
    PufferRNG,
    ReceiptContext,
    RollingEstimate,
    SensorChannel,
    SensorChannelConfig,
)
from puffer.types import HealthState


def test_frames_to_key_with_receipt(make_frame, make_tracking, make_observation, rng):
    monitor = HealthMonitor()
    sensor = SensorChannel(SensorChannelConfig(n_sites=1024, site_seed=b"pipeline"))
    fish = FishChannel()
    estimates = {name: RollingEstimate(h_cap=cap) for name, cap in DEFAULT_H_CAP.items()}
    pool = EntropyPool()
    scene = rng.integers(20, 230, size=(120, 160, 3)).astype(np.float64)

    for i in range(60):
        image = np.clip(scene + rng.normal(0.0, 2.0, scene.shape), 0, 255).astype(np.uint8)
        frame = make_frame(image, index=i)
        observations = [
            make_observation(k, tuple(rng.normal(0.0, 1.5, 2)), f"f{k}", frame.timestamp)
            for k in range(3)
        ]
        tracking = make_tracking(frame.timestamp, 3, observations, frame_index=i)
        monitor.observe_frame(frame)
        monitor.observe_tracking(tracking)
        for out in (sensor.process(frame), fish.process(tracking)):
            monitor.observe_samples(out.channel, out.samples, out.t)
            estimate = estimates[out.channel]
            if estimate.feed(out.samples):
                monitor.observe_entropy_estimate(out.channel, estimate.h_est, out.t)
            if monitor.usable(out.channel):
                credit = estimate.credit(out.samples.size, monitor.credit_multiplier(out.channel))
                pool.absorb(out, credit)

    assert monitor.state("sensor") is HealthState.OK
    assert monitor.state("fish") is HealthState.STARTUP  # 180 fish samples < 1024
    assert pool.credits()["sensor"] == 4096.0  # saturated
    assert pool.credits()["fish"] == 0.0

    context = ReceiptContext(
        mode="day", health={name: r.state for name, r in monitor.reports().items()}
    )
    key, receipt = PufferRNG(pool, context=lambda: context).generate_key()
    assert len(key) == 32
    d = receipt.to_dict()
    assert d["channels"]["sensor"] == {"credited_bits": 4096.0, "share": 1.0, "health": "ok"}
    assert d["channels"]["fish"]["health"] == "startup"
    assert d["frames"] == [0, 59]
