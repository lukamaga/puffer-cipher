"""The replay attack: a perfect recording of the tank must never earn camera credit.

The attacker records the (seeded) simulator to a lossless file while its LED
shows challenge *A*, then feeds the file to the engine. Liveness is forced to
``required`` and the verifier is even told exactly where the LED is - the
most favourable setup for the attacker - but the engine draws a *fresh*
challenge from ``os.urandom``, which did not exist when the file was made.
The recording is bit-exact (FFV1), so nothing but freshness distinguishes it
from the live tank; the control test shows the very same file verifies
against the challenge it actually recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from puffer.liveness import ChallengeSchedule
from puffer.runtime import build_engine
from puffer.sources import SyntheticAquarium, SyntheticConfig, record
from puffer.types import CHANNEL_FISH, CHANNEL_SENSOR, HealthState, LivenessState

from .helpers import HEIGHT, WIDTH, config, run_frames, seeded_schedule

RECORD_S = 14.0  # one full 64-bit window (12.8 s) plus a second of verdicts


@dataclass(frozen=True)
class Recording:
    path: Path
    led_roi: tuple[int, int, int, int]
    schedule: ChallengeSchedule  # challenge A, shown while recording
    frames: int


def _record(directory: Path, seconds: float) -> Recording:
    schedule = seeded_schedule(101)
    sim = SyntheticAquarium(
        SyntheticConfig(seed=3, width=WIDTH, height=HEIGHT, led_xy=(WIDTH - 32, 32)),
        led_schedule=schedule,
    )
    path = directory / "tank.mkv"
    frames = record(sim, path, seconds=seconds, fourcc="FFV1")
    return Recording(path, sim.led_roi, schedule, frames)


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Recording:
    return _record(tmp_path_factory.mktemp("replay"), RECORD_S)


def test_replayed_recording_never_verifies_and_earns_no_camera_credit(
    recording: Recording, tmp_path: Path
) -> None:
    cfg = config(
        tmp_path,
        source=f"video:{recording.path}",
        liveness="required",
        led_roi=recording.led_roi,
    )
    engine = build_engine(cfg)  # a fresh challenge: os.urandom bits drawn now
    try:
        assert engine.liveness_policy == "required"
        run = run_frames(engine, recording.frames + 10)
        assert len(run.records) == recording.frames  # the whole file was processed
        assert all(r.liveness is not LivenessState.VERIFIED for r in run.records)
        assert run.last.liveness is LivenessState.FAILED  # a full window was judged
        for record_ in run.records:
            assert record_.credited[CHANNEL_SENSOR] == 0.0
            assert record_.credited[CHANNEL_FISH] == 0.0
        for channel in (CHANNEL_SENSOR, CHANNEL_FISH):
            assert run.last.states[channel] is HealthState.FAILED
            assert run.last.checks[channel]["liveness"] == "failed"
        snap = engine.snapshot()
        assert snap is not None
        assert snap.liveness.confidence_bits < 24.0  # indistinguishable from chance
        assert snap.liveness.match_ratio < 0.9
        assert engine.pool.camera_credit() == 0.0

        # Outputs are still produced, keyed by os.urandom, and honestly labelled.
        key, receipt = engine.rng.generate_key()
        d = receipt.to_dict()
        assert len(key) == 32
        assert key != engine.rng.generate_key()[0]
        assert receipt.camera_bits == 0.0
        assert d["channels"]["sensor"] == {"credited_bits": 0.0, "share": None, "health": "failed"}
        assert d["channels"]["fish"] == {"credited_bits": 0.0, "share": None, "health": "failed"}
        assert d["channels"]["system"]["credited_bits"] == 256.0
        assert d["liveness"]["state"] == "failed"
        assert d["fish"] == []
    finally:
        engine.close()


def test_video_files_default_to_liveness_off(recording: Recording, tmp_path: Path) -> None:
    """``auto`` credits recorded fish without a verifier (the documented MVP behaviour)."""
    engine = build_engine(config(tmp_path, source=f"video:{recording.path}"))
    try:
        assert engine.liveness_policy == "off"
        assert engine.liveness_verifier is None
        run = run_frames(engine, 40)
        assert run.last.liveness is LivenessState.DISABLED
        assert run.last.credited[CHANNEL_SENSOR] > 0
        _, receipt = engine.rng.generate_key()
        assert receipt.to_dict()["liveness"]["state"] == "disabled"
    finally:
        engine.close()


def test_advisory_liveness_reports_without_gating_credit(
    recording: Recording, tmp_path: Path
) -> None:
    """``advisory``: the verdict is shown, but the unverified replay is credited anyway."""
    cfg = config(
        tmp_path,
        source=f"video:{recording.path}",
        liveness="advisory",
        led_roi=recording.led_roi,
    )
    engine = build_engine(cfg)
    try:
        run = run_frames(engine, 60)
        assert all(r.liveness is not LivenessState.VERIFIED for r in run.records)
        assert run.last.checks[CHANNEL_SENSOR]["liveness"] == "ok"
        assert run.last.states[CHANNEL_SENSOR] is HealthState.OK
        assert run.last.credited[CHANNEL_SENSOR] > 0
        _, receipt = engine.rng.generate_key()
        d = receipt.to_dict()
        assert d["liveness"]["state"] == "calibrating"
        assert receipt.camera_bits > 0
    finally:
        engine.close()


@pytest.mark.slow
def test_control_same_recording_verifies_against_its_own_challenge(
    tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> None:
    """Control: replayed against challenge A itself, the file passes.

    This proves the replay test fails *because the challenge is fresh*, not
    because of the codec or the detector. It also exercises auto-calibration
    on a video (no ROI given): the sensor channel must be rebuilt to exclude
    the LED region the verifier found.
    """
    rec = _record(tmp_path_factory.mktemp("control"), 22.0)
    cfg = config(tmp_path, source=f"video:{rec.path}", liveness="required")
    engine = build_engine(cfg, schedule=rec.schedule)
    try:
        run = run_frames(engine, rec.frames)
        verified = run.first(lambda r: r.liveness is LivenessState.VERIFIED)
        assert verified is not None
        # 4 s of calibration (discarded), then one full 64-bit window.
        assert 4.0 + 12.8 <= verified.t <= 4.0 + 12.8 + 1.5
        roi = engine.liveness_verifier.roi if engine.liveness_verifier else None
        assert roi is not None
        x, y, w, h = roi
        lx, ly, lw, lh = rec.led_roi
        assert lx <= x + w / 2 <= lx + lw
        assert ly <= y + h / 2 <= ly + lh
        ys, xs = engine.sensor_channel.sites
        inside = (xs >= x - 4) & (xs < x + w + 4) & (ys >= y - 4) & (ys < y + h + 4)
        assert not inside.any()
        assert run.last.credited[CHANNEL_SENSOR] > 0
    finally:
        engine.close()
