"""Health in the running engine: frozen video, transient failures and recovery, stalls."""

from __future__ import annotations

from pathlib import Path

from puffer.engine import EngineState, EntropyEngine
from puffer.runtime import build_engine
from puffer.sources import SyntheticAquarium, SyntheticConfig
from puffer.types import CHANNEL_FISH, CHANNEL_SENSOR, HealthState

from .helpers import (
    HEIGHT,
    WIDTH,
    FakeMonotonic,
    FreezeWindow,
    FrozenSource,
    ScriptedSource,
    config,
    run_frames,
    states_sequence,
    wait_for,
)


def _sim(seed: int) -> SyntheticAquarium:
    return SyntheticAquarium(
        SyntheticConfig(seed=seed, width=WIDTH, height=HEIGHT, led_xy=(WIDTH - 32, 32))
    )


def test_frozen_source_fails_the_camera_channels(tmp_path: Path) -> None:
    image = _sim(5).read()
    assert image is not None
    engine = build_engine(
        config(tmp_path, liveness="off", led="none"), source=FrozenSource(image.image)
    )
    try:
        run = run_frames(engine, 40)
        last = run.last
        for channel in (CHANNEL_SENSOR, CHANNEL_FISH):
            assert last.states[channel] is HealthState.FAILED
            assert last.checks[channel]["frozen_frame"] == "failed"
        assert any("frozen video" in reason for reason in last.reasons[CHANNEL_SENSOR])
        # A still picture also has zero temporal noise, which the RCT catches at once.
        assert last.checks[CHANNEL_SENSOR]["repetition_count"] == "failed"
        frozen_since = run.first(lambda r: r.checks[CHANNEL_SENSOR]["frozen_frame"] == "failed")
        assert frozen_since is not None
        assert frozen_since.index == 15
        assert all(r.credited[CHANNEL_SENSOR] == 0.0 for r in run.records)
        _, receipt = engine.rng.generate_key()
        assert receipt.camera_bits == 0.0
    finally:
        engine.close()


def test_sensor_recovers_after_a_transient_freeze(tmp_path: Path) -> None:
    """2/3 s of repeated frames: FAILED, 5 s cooldown, a fresh start-up, OK again."""
    source = FreezeWindow(_sim(6), start=30, stop=50)
    engine = build_engine(config(tmp_path, liveness="off", led="none"), source=source)
    try:
        run = run_frames(engine, 240, until=lambda r: r.index >= 215)
        assert states_sequence(run.records, CHANNEL_SENSOR) == [
            "startup",
            "ok",
            "failed",
            "startup",
            "ok",
        ]
        failed = run.first(lambda r: r.states[CHANNEL_SENSOR] is HealthState.FAILED)
        recovered = run.first(lambda r: r.index > 50 and r.states[CHANNEL_SENSOR] is HealthState.OK)
        assert failed is not None
        assert recovered is not None
        assert failed.index == 31  # first repeated frame: all differences zero
        # The cooldown runs from the moment the frozen check cleared (frame 50, t = 1.67 s).
        assert recovered.t >= 50 / 30 + 5.0
        during = [r for r in run.records if failed.index <= r.index < recovered.index]
        assert all(r.credited[CHANNEL_SENSOR] == failed.credited[CHANNEL_SENSOR] for r in during)
        assert any(r.checks[CHANNEL_SENSOR]["frozen_frame"] == "failed" for r in during)
        assert all(r.states[CHANNEL_FISH] is not HealthState.OK for r in during)
        assert run.last.credited[CHANNEL_SENSOR] > recovered.credited[CHANNEL_SENSOR]
    finally:
        engine.close()


def test_stalled_live_source_times_out_while_blocked(tmp_path: Path) -> None:
    """A camera stuck inside read(): the watchdog fails the channels after 5 s."""
    clock = FakeMonotonic()
    source = ScriptedSource(stall_after=40, clock=clock)
    engine = EntropyEngine(config(tmp_path, liveness="off"), source, live_source=True, clock=clock)
    try:
        engine.start()
        assert wait_for(source.stalled.is_set)
        snap = engine.snapshot()
        assert snap is not None
        assert snap.health[CHANNEL_SENSOR].checks["source_timeout"] == "ok"
        clock.advance(6.0)  # six seconds without a frame
        assert wait_for(
            lambda: (
                (s := engine.snapshot()) is not None
                and s.health[CHANNEL_SENSOR].checks.get("source_timeout") == "failed"
            )
        )
        snap = engine.snapshot()
        assert snap is not None
        assert snap.channels[CHANNEL_SENSOR].health is HealthState.FAILED
        assert any("no frame for" in reason for reason in snap.health[CHANNEL_SENSOR].reasons)
        assert engine.run_state is EngineState.RUNNING  # still waiting for the camera
    finally:
        source.close()  # unblocks the read
        engine.close()
