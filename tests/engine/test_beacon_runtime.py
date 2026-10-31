"""The public beacon wired to a real engine: valid chain, honest summaries, restarts."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from puffer.beacon import verify_chain
from puffer.runtime import BEACON_DB, BEACON_KEY, build_beacon, build_engine

from .helpers import config, run_frames, seeded_schedule

T0_MS = 1_793_454_000_000  # 2026-10-31T16:20:00Z, a slot boundary for 60 s periods


class FakeClockMs:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _integer_only(value: Any) -> bool:
    """Ints, strings, bools, lists and dicts only: the DESIGN.md 3.4 summary rule."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and _integer_only(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_integer_only(v) for v in value)
    return isinstance(value, str | bool | int) and value is not None


def test_beacon_chain_from_engine_survives_restart(tmp_path: Path) -> None:
    cfg = config(tmp_path, liveness="off", beacon_period_s=60)
    engine = build_engine(cfg, schedule=seeded_schedule(21))
    try:
        clock = FakeClockMs(T0_MS + 1_234)
        # A new beacon key is itself drawn from PufferRNG (a fish-born identity).
        service = build_beacon(cfg, engine, clock_ms=clock)
        public_key = service.public_key
        # Sensor credit arrives at once; fish credit only after the fish channel's
        # 1024-sample start-up window, whose length depends on how soon tracks are
        # classified as fish. Wait for it (bounded) instead of a fixed frame count.
        warmup = run_frames(
            engine, 1500, until=lambda _: engine.pool.credits().get("fish", 0.0) > 0
        )
        genesis = service.tick()
        run_frames(engine, 30)
        clock.now += 60_000
        second = service.tick()
        assert (genesis.index, second.index) == (0, 1)
        assert genesis.timestamp_ms == T0_MS
        assert second.timestamp_ms == T0_MS + 60_000
        assert genesis.period_ms == 60_000
        pulses = service.store.range(0, 10)
        report = verify_chain(pulses, public_key)
        assert report.ok, report.errors

        summary = json.loads(genesis.summary)
        assert _integer_only(summary)
        assert "null" not in genesis.summary
        assert summary["sensor_bits"] == 4096
        assert summary["system_bits"] == 256
        assert summary["fish_bits"] > 0
        assert summary["fish_count"] >= 1
        assert summary["health"] == {"sensor": "ok", "fish": "ok", "system": "ok"}
        assert summary["liveness"] == "disabled"
        assert summary["mode"] == "day"
        top = summary["top_fish"]
        assert 1 <= len(top) <= 3
        assert all(f["name"] and f["name"] != f["fish_id"] for f in top if f["fish_id"][0] == "f")
        assert all(isinstance(f["bits"], int) for f in top)
        first, last = summary["frames"]
        assert 0 <= first <= last <= warmup.records[-1].index
        # Pulse 1 reveals the value drawn at genesis right after pulse 0's own draw,
        # so it honestly reports (almost) no fresh camera credit.
        assert json.loads(second.summary)["sensor_bits"] <= 4096
        service.store.close()

        # Restart: same key from beacon.key, the chain continues from the stored reveal.
        restarted = build_beacon(cfg, engine, clock_ms=clock)
        assert restarted.public_key == public_key
        assert restarted.store.count() == 2
        clock.now += 3 * 60_000  # downtime: two slots missed
        third = restarted.tick()
        assert third.index == 2
        assert third.timestamp_ms == T0_MS + 4 * 60_000
        report = verify_chain(restarted.store.range(0, 10), public_key)
        assert report.ok, report.errors
        assert report.checked == 3
        restarted.store.close()
    finally:
        engine.close()
    assert stat.S_IMODE((tmp_path / BEACON_KEY).stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / BEACON_DB).stat().st_mode) == 0o600


def test_beacon_keeps_publishing_without_camera_credit(tmp_path: Path) -> None:
    """Liveness required and not yet verified: pulses still come, from os.urandom-keyed output."""
    cfg = config(tmp_path)
    engine = build_engine(cfg, schedule=seeded_schedule(22))
    try:
        run_frames(engine, 10)
        clock = FakeClockMs(T0_MS)
        service = build_beacon(cfg, engine, clock_ms=clock)
        pulse = service.tick()
        summary = json.loads(pulse.summary)
        assert summary["sensor_bits"] == summary["fish_bits"] == 0
        assert summary["liveness"] == "calibrating"
        assert summary["health"]["sensor"] == "failed"
        assert summary["system_bits"] == 256
        assert verify_chain([pulse], service.public_key).ok
        service.store.close()
    finally:
        engine.close()
