"""End to end on the seeded simulator: liveness, credit, receipts, day versus night.

These are the proof that the system works as a whole: frames from the
simulator flow through liveness, tracking, both entropy channels, health
tests and the pool into PufferRNG, and the receipt tells the truth about it.
"""

from __future__ import annotations

import pytest

from puffer.types import CHANNEL_FISH, CHANNEL_SENSOR, HealthState, LivenessState

from .helpers import Run, run_frames

WINDOW_S = 64 * 0.2  # liveness_window_bits * symbol_s: the earliest possible verdict
COOLDOWN_S = 5.0  # HealthConfig.cooldown_s after the liveness check clears


def test_liveness_verifies_after_one_window(day_run: Run) -> None:
    verified = day_run.first(lambda r: r.liveness is LivenessState.VERIFIED)
    assert verified is not None
    # The first complete 64-bit window ends 12.8 s after the start, plus the
    # simulated 50 ms camera latency; this seeded run verifies at t = 12.87 s.
    assert WINDOW_S <= verified.t <= WINDOW_S + 1.0
    later = [r for r in day_run.records if r.t >= verified.t]
    assert all(r.liveness is LivenessState.VERIFIED for r in later), "verdict must stay stable"
    assert day_run.last.liveness is LivenessState.VERIFIED


def test_no_camera_credit_before_liveness_and_cooldown(day_run: Run) -> None:
    verified = day_run.first(lambda r: r.liveness is LivenessState.VERIFIED)
    assert verified is not None
    gate = verified.t + COOLDOWN_S
    for record in day_run.records:
        if record.t < gate:
            assert record.credited[CHANNEL_SENSOR] == 0.0
            assert record.credited[CHANNEL_FISH] == 0.0
            assert record.states[CHANNEL_SENSOR] is HealthState.FAILED
            assert record.checks[CHANNEL_SENSOR].get("liveness") in ("failed", "ok")
    first_credit = day_run.first(lambda r: r.credited[CHANNEL_SENSOR] > 0)
    assert first_credit is not None
    assert first_credit.t >= gate


def test_sensor_and_fish_channels_earn_credit(day_run: Run) -> None:
    last = day_run.last
    assert last.states[CHANNEL_SENSOR] is HealthState.OK
    assert last.states[CHANNEL_FISH] is HealthState.OK
    assert last.credited[CHANNEL_SENSOR] > 100_000  # ~6.8 kbit per frame, saturating the pool
    assert last.credited[CHANNEL_FISH] > 1_000
    assert last.bits_per_s[CHANNEL_FISH] > 100
    snap = day_run.engine.snapshot()
    assert snap is not None
    assert snap.pool[CHANNEL_SENSOR] == snap.pool_capacity_bits  # saturated: nobody drew yet
    h_fish = snap.channels[CHANNEL_FISH].h_per_sample
    assert h_fish is not None
    assert h_fish > 0.5


def test_key_receipt_attributes_sensor_and_named_fish(day_run: Run) -> None:
    engine = day_run.engine
    key, receipt = engine.rng.generate_key()
    d = receipt.to_dict()
    assert len(key) == 32
    assert d["purpose"] == "key"
    assert d["mode"] == "day"
    assert d["liveness"]["state"] == "verified"
    assert d["liveness"]["confidence_bits"] >= 24.0
    sensor, fish = d["channels"]["sensor"], d["channels"]["fish"]
    assert sensor["credited_bits"] == 4096.0
    assert sensor["health"] == "ok"
    assert fish["credited_bits"] > 100
    assert fish["health"] == "ok"
    assert 0 < fish["share"] < 1
    assert sensor["share"] + fish["share"] == pytest.approx(1, abs=1e-3)
    assert d["channels"]["system"]["credited_bits"] == 256.0
    named = [f for f in d["fish"] if not f["fish_id"].startswith("track-")]
    assert len(named) >= 3, d["fish"]
    assert all(f["name"] and f["name"] != f["fish_id"] for f in named)
    assert sum(f["bits"] for f in d["fish"]) == pytest.approx(fish["credited_bits"], rel=1e-3)
    first, last = d["frames"]
    assert 0 < first <= last == day_run.last.index
    # The draw consumed everything: the next key gets (almost) nothing from the camera.
    _, again = engine.rng.generate_key()
    assert again.camera_bits == 0.0


def test_outputs_have_requested_length_and_never_repeat(day_run: Run) -> None:
    engine = day_run.engine
    run_frames(engine, 15)  # refill the pool a little
    keys = [engine.rng.generate_key()[0] for _ in range(64)]
    assert all(len(k) == 32 for k in keys)
    assert len(set(keys)) == len(keys)
    data, receipt = engine.rng.generate(3 * 2**20 + 7, "bytes")  # spans four 1 MiB chunks
    assert len(data) == 3 * 2**20 + 7
    assert receipt.n_bytes == len(data)
    chunks = {data[i : i + 2**20] for i in range(0, 3 * 2**20, 2**20)}
    assert len(chunks) == 3
    nonces = {engine.rng.nonce() for _ in range(32)}
    assert len(nonces) == 32
    assert all(len(n) == 24 for n in nonces)
    assert len(engine.rng.salt()) == 16


def test_night_fish_rate_lower_but_sensor_credited(day_run: Run, night_run: Run) -> None:
    day, night = day_run.last, night_run.last
    # The LED challenge is visible to the IR camera too.
    assert night.liveness is LivenessState.VERIFIED
    assert night.states[CHANNEL_SENSOR] is HealthState.OK
    assert night.credited[CHANNEL_SENSOR] > 100_000
    assert night.bits_per_s[CHANNEL_SENSOR] > 0
    # Sleeping fish under IR: fewer confirmed tracks, so fewer innovation samples.
    assert night.states[CHANNEL_FISH] in (HealthState.OK, HealthState.DEGRADED)
    assert 0 < night.bits_per_s[CHANNEL_FISH] < 0.75 * day.bits_per_s[CHANNEL_FISH]
    assert _rate(night_run, CHANNEL_FISH, 5.0) < 0.75 * _rate(day_run, CHANNEL_FISH, 5.0)
    snap = night_run.engine.snapshot()
    assert snap is not None
    assert snap.mode.value == "night"


def _rate(run: Run, channel: str, seconds: float) -> float:
    """Mean credited bits per second over the last ``seconds`` of the run."""
    end = run.last
    start = next(r for r in run.records if r.t >= end.t - seconds)
    return (end.credited[channel] - start.credited[channel]) / (end.t - start.t)
