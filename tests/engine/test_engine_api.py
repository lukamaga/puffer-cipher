"""The engine's public surface: thread, subscribers, JPEG, fish cards, collect, errors."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from puffer.engine import (
    EngineSnapshot,
    EngineState,
    EntropyEngine,
    chaos_fraction,
)
from puffer.hardware import SimulatedLed
from puffer.liveness import LivenessVerifier
from puffer.runtime import FISH_GALLERY, STATS_DB, build_engine
from puffer.sources import SyntheticAquarium, SyntheticConfig
from puffer.types import (
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    ChannelOutput,
    Frame,
    HealthState,
    LivenessState,
    TrackingResult,
)
from puffer.vision import FishTracker

from .helpers import (
    HEIGHT,
    WIDTH,
    Run,
    ScriptedSource,
    config,
    run_frames,
    seeded_schedule,
    sim_spec,
    wait_for,
)

FISH_CARD_KEYS = {"fish_id", "name", "color", "present", "track_id", "stats"}
FISH_STATS_KEYS = {
    "speed_avg",
    "speed_max",
    "activity",
    "chaos_score",
    "bits_total",
    "bits_today",
    "distance_px",
    "seen_s",
}


# --------------------------------------------------------------------------- thread


def test_thread_runs_subscribers_and_survives_a_failing_one(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off"))
    seen: list[tuple[int, str]] = []
    failures = 0

    def good(snap: EngineSnapshot) -> None:
        seen.append((snap.frame_index, threading.current_thread().name))

    def bad(snap: EngineSnapshot) -> None:
        nonlocal failures
        failures += 1
        raise RuntimeError("subscriber bug")

    unsubscribe_bad = engine.subscribe(bad)
    unsubscribe_good = engine.subscribe(good)
    try:
        engine.start()
        engine.start()  # idempotent
        assert engine.running
        assert engine.run_state is EngineState.RUNNING
        with pytest.raises(RuntimeError, match="thread is running"):
            engine.step()
        assert wait_for(lambda: len(seen) >= 20)
        # Readers work concurrently with the frame loop.
        for _ in range(5):
            assert engine.snapshot() is not None
            assert engine.jpeg(max_width=160) is not None
            engine.fish_cards()
            assert len(engine.rng.generate(32, "bytes")[0]) == 32
        unsubscribe_bad()
        engine.stop()
        assert not engine.running
        assert engine.run_state is EngineState.STOPPED
        count = len(seen)
        assert {name for _, name in seen} == {"puffer-engine"}
        indices = [i for i, _ in seen]
        assert indices == list(range(indices[0], indices[0] + count))  # every frame, in order
        assert failures >= 20
        assert engine.errors == 0
        unsubscribe_good()
        engine.start()  # restartable
        assert wait_for(lambda: (s := engine.snapshot()) is not None and s.frame_index > count + 5)
        assert len(seen) == count  # unsubscribed
    finally:
        engine.close()
    assert engine.run_state is EngineState.CLOSED
    engine.close()  # idempotent


def test_frame_errors_are_counted_and_the_loop_keeps_going(tmp_path: Path) -> None:
    source = ScriptedSource(fail=lambda i: i % 4 == 3)
    engine = EntropyEngine(config(tmp_path, liveness="off"), source, max_consecutive_errors=3)
    try:
        engine.start()
        assert wait_for(lambda: (s := engine.snapshot()) is not None and s.frame_index >= 30)
        engine.stop()
        snap = engine.snapshot()
        assert snap is not None
        assert engine.errors >= 8
        assert snap.errors == engine.errors
        assert snap.last_error is not None
        assert "scripted read failure" in snap.last_error
        assert engine.run_state is EngineState.STOPPED
    finally:
        engine.close()


def test_repeated_failures_stop_the_thread_with_a_visible_error(tmp_path: Path) -> None:
    source = ScriptedSource(fail=lambda i: True)
    engine = EntropyEngine(config(tmp_path, liveness="off"), source, max_consecutive_errors=3)
    try:
        engine.start()
        assert wait_for(lambda: not engine.running)
        assert engine.run_state is EngineState.ERROR
        snap = engine.snapshot()
        assert snap is not None  # visible although no frame ever succeeded
        assert snap.frame_index == -1
        assert snap.run_state is EngineState.ERROR
        assert snap.errors == 3
        assert "scripted read failure" in (snap.last_error or "")
        assert set(snap.channels) == {CHANNEL_SENSOR, CHANNEL_FISH, CHANNEL_SYSTEM}
    finally:
        engine.close()


class BrokenTracker(FishTracker):
    """A vision pipeline with a bug that fires on every frame."""

    def process(self, frame: Frame) -> TrackingResult:
        raise ZeroDivisionError("vision bug")


def test_a_broken_tracker_costs_the_fish_channel_not_the_sensor(tmp_path: Path) -> None:
    sim = SyntheticAquarium(
        SyntheticConfig(seed=9, width=WIDTH, height=HEIGHT, led_xy=(WIDTH - 32, 32))
    )
    engine = EntropyEngine(config(tmp_path), sim, tracker=BrokenTracker())
    try:
        run = run_frames(engine, 10)
        assert len(run.records) == 10  # every frame survived
        assert run.last.states[CHANNEL_SENSOR] is HealthState.OK
        assert run.last.credited[CHANNEL_SENSOR] > 0
        assert run.last.credited[CHANNEL_FISH] == 0.0
        assert engine.errors == 10
        assert engine.last_error == "vision: ZeroDivisionError: vision bug"
        snap = engine.snapshot()
        assert snap is not None
        assert snap.errors == 10
        assert snap.tracks == ()
    finally:
        engine.close()


def test_finite_source_ends_the_thread(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, source=sim_spec(seed=2), liveness="off"))
    try:
        run_frames(engine, 3)
        engine.source.close()  # the simulator then reports "exhausted"
        engine.start()
        assert wait_for(lambda: not engine.running)
        assert engine.run_state is EngineState.ENDED
        assert engine.step() is None
    finally:
        engine.close()


# --------------------------------------------------------------------------- readers


def test_jpeg_is_downscaled_and_cached_per_frame(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        assert engine.jpeg() is None  # no frame yet
        run_frames(engine, 2)
        full = engine.jpeg()
        small = engine.jpeg(max_width=160, quality=50)
        assert full is not None
        assert small is not None
        decoded = cv2.imdecode(np.frombuffer(full, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (HEIGHT, WIDTH, 3)
        decoded = cv2.imdecode(np.frombuffer(small, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (HEIGHT // 2, WIDTH // 2, 3)
        assert engine.jpeg(max_width=160, quality=50) is small  # same frame: cached bytes
        run_frames(engine, 1)
        assert engine.jpeg(max_width=160, quality=50) is not small
        with pytest.raises(ValueError, match="quality"):
            engine.jpeg(quality=0)
    finally:
        engine.close()


def test_fish_cards_match_the_design_shape(day_run: Run) -> None:
    cards = day_run.engine.fish_cards()
    assert len(cards) >= 4
    for card in cards:
        assert set(card) == FISH_CARD_KEYS
        assert set(card["stats"]) == FISH_STATS_KEYS
        assert re.fullmatch(r"#[0-9a-f]{6}", card["color"])
        assert isinstance(card["present"], bool)
        assert card["track_id"] is None or isinstance(card["track_id"], int)
        assert card["present"] == (card["track_id"] is not None)
        stats = card["stats"]
        assert 0.0 <= stats["activity"] <= 1.0
        assert 0.0 <= stats["chaos_score"] < 1.0
        assert 0.0 <= stats["speed_avg"] <= stats["speed_max"]
        assert stats["bits_today"] <= stats["bits_total"] + 1e-9
        assert all(isinstance(v, float | int) for v in stats.values())
    ids = [int(c["fish_id"][1:]) for c in cards]
    assert ids == sorted(ids)  # natural order: f2 before f10
    assert any(c["present"] for c in cards)
    assert sum(c["stats"]["bits_total"] for c in cards) > 0
    # The snapshot carries the labels the video overlay needs.
    snap = day_run.engine.snapshot()
    assert snap is not None
    for card in cards:
        assert snap.fish[card["fish_id"]].name == card["name"]


def test_leaderboard_ranks_by_chaos(day_run: Run) -> None:
    board = day_run.engine.leaderboard(days=7)
    assert [e["rank"] for e in board] == list(range(1, len(board) + 1))
    assert all(set(e) == {"rank", "fish_id", "name", "chaos_score", "bits"} for e in board)
    scores = [e["chaos_score"] for e in board if e["chaos_score"] is not None]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s < 1 for s in scores)
    with pytest.raises(ValueError, match="positive integer"):
        day_run.engine.leaderboard(days=0)


def test_entropy_history_has_the_entropy_message_shape(day_run: Run) -> None:
    history = day_run.engine.entropy_history(seconds=10)
    assert 19 <= len(history) <= 22  # 2 Hz
    times = [item["t"] for item in history]
    assert times == sorted(times)
    item = history[-1]
    assert set(item) == {"t", "mode", "channels", "pool"}
    assert set(item["channels"]) == {CHANNEL_SENSOR, CHANNEL_FISH, CHANNEL_SYSTEM}
    assert set(item["pool"]) == {CHANNEL_SENSOR, CHANNEL_FISH, "capacity_bits"}
    assert item["channels"][CHANNEL_SYSTEM]["h_per_sample"] is None
    assert set(item["channels"][CHANNEL_SENSOR]) == {
        "bits_per_s",
        "h_per_sample",
        "credited_total",
        "health",
    }
    assert len(day_run.engine.entropy_history(seconds=10_000)) <= 1201


def test_receipt_context_always_reports_camera_health(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        context = engine.receipt_context()
        assert context.mode == "unknown"
        assert set(context.health) == {CHANNEL_SENSOR, CHANNEL_FISH}
        assert context.liveness.state is LivenessState.DISABLED
        run_frames(engine, 3)
        assert engine.receipt_context().mode == "day"
    finally:
        engine.close()


def test_output_listeners_see_every_channel_output(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off"))
    outputs: list[tuple[str, int, float, HealthState]] = []

    def listener(out: ChannelOutput, credit: float, state: HealthState) -> None:
        outputs.append((out.channel, out.frame_index, credit, state))

    def broken(out: ChannelOutput, credit: float, state: HealthState) -> None:
        raise ZeroDivisionError("analysis hook bug")

    engine.add_output_listener(broken)
    unsubscribe = engine.add_output_listener(listener)
    try:
        run_frames(engine, 5)
        unsubscribe()
        run_frames(engine, 2)
        assert [(c, i) for c, i, _, _ in outputs] == [
            (channel, i) for i in range(5) for channel in (CHANNEL_SENSOR, CHANNEL_FISH)
        ]
        sensor = [o for o in outputs if o[0] == CHANNEL_SENSOR]
        assert sensor[0][2] == 0.0
        assert sensor[0][3] is HealthState.STARTUP
        assert sensor[-1][2] > 0
        assert sensor[-1][3] is HealthState.OK
    finally:
        engine.close()


# --------------------------------------------------------------------------- collect


def test_collect_steps_until_enough_camera_credit(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        assert engine.collect(1024, timeout_s=60, max_frames=20)
        assert engine.pool.camera_credit() >= 1024
        frames = engine.snapshot()
        assert frames is not None
        assert frames.frame_index < 5
        _, receipt = engine.rng.generate_key()
        assert receipt.camera_bits >= 1024
        with pytest.raises(ValueError, match="saturate"):
            engine.collect(10_000, timeout_s=1)
    finally:
        engine.close()


def test_collect_gives_up_without_liveness(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path), schedule=seeded_schedule(31))
    try:
        assert not engine.collect(256, timeout_s=60, max_frames=30)  # liveness not verified yet
        assert engine.pool.camera_credit() == 0.0
        assert not engine.collect(256, timeout_s=0.0)
    finally:
        engine.close()


def test_rng_wait_for_credit_pumps_frames(tmp_path: Path) -> None:
    """``PufferRNG.wait_for_credit`` steps a synchronous engine through its pump hook."""
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        assert engine.rng.wait_for_credit(2048, timeout=60)
        snap = engine.snapshot()
        assert snap is not None
        assert snap.frame_index >= 2
    finally:
        engine.close()


# --------------------------------------------------------------------------- vision


def test_led_is_masked_from_the_tracker(tmp_path: Path) -> None:
    """A blinking LED is a perfect motion blob; it must never become a credited 'fish'."""
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        roi = engine.led_roi
        assert roi is not None
        x, y, w, h = roi
        for _ in range(150):  # unmasked, the LED is a confirmed track within ~1 s
            snap = engine.step()
            assert snap is not None
            for track in snap.tracks:
                cx, cy = track.center
                assert not (x - 8 <= cx <= x + w + 8 and y - 8 <= cy <= y + h + 8), track
        ys, xs = engine.sensor_channel.sites
        inside = (xs >= x - 4) & (xs < x + w + 4) & (ys >= y - 4) & (ys < y + h + 4)
        assert xs.size == 4096
        assert not inside.any()
    finally:
        engine.close()


def test_wide_frames_are_tracked_downscaled_and_reported_in_frame_pixels(tmp_path: Path) -> None:
    cfg = config(tmp_path, source=sim_spec(seed=7, width=640), liveness="off", vision_max_width=320)
    engine = build_engine(cfg)
    try:
        xs: list[float] = []
        snap = None
        for _ in range(75):  # 30 frames of background learning, then tracks
            snap = engine.step()
            assert snap is not None
            for track in snap.tracks:
                x1, y1, x2, y2 = track.bbox
                # Kalman boxes may overshoot the border by a little, never by a scale factor.
                assert -8 <= x1 < x2 <= 648
                assert -8 <= y1 < y2 <= 488
                assert x1 <= track.center[0] <= x2
                xs.append(track.center[0])
        assert snap is not None
        assert snap.width == 640
        assert max(xs) > 400  # coordinates are frame pixels, not tracker pixels
        cards = engine.fish_cards()
        assert cards
        assert max(c["stats"]["speed_max"] for c in cards) > 0
    finally:
        engine.close()


def test_calibrated_led_region_rebuilds_the_sensor_channel(tmp_path: Path) -> None:
    schedule = seeded_schedule(41)
    sim = SyntheticAquarium(
        SyntheticConfig(seed=8, width=WIDTH, height=HEIGHT, led_xy=(WIDTH - 32, 32)),
        led_schedule=schedule,
    )
    verifier = LivenessVerifier(schedule, None)  # no ROI: calibrate from the video
    engine = EntropyEngine(
        config(tmp_path), sim, liveness=verifier, liveness_policy="advisory", sensor_site_seed=b"x"
    )
    try:
        assert engine.led_roi is None
        before = engine.sensor_channel
        run = run_frames(engine, 200, until=lambda r: engine.led_roi is not None)
        assert engine.led_roi is not None
        assert engine.led_roi == verifier.roi
        assert run.last.t >= 4.0  # one calibration window
        assert engine.sensor_channel is not before
        x, y, w, h = engine.led_roi
        lx, ly, lw, lh = sim.led_roi
        assert lx <= x + w / 2 <= lx + lw
        assert ly <= y + h / 2 <= ly + lh
        run_frames(engine, 2)
        ys, xs = engine.sensor_channel.sites
        inside = (xs >= x - 4) & (xs < x + w + 4) & (ys >= y - 4) & (ys < y + h + 4)
        assert xs.size == 4096
        assert not inside.any()
    finally:
        engine.close()


# --------------------------------------------------------------------------- lifecycle


def test_close_switches_the_led_off_and_persists_state(tmp_path: Path) -> None:
    source = ScriptedSource()  # an unknown source counts as a live camera
    cfg = config(tmp_path, liveness="advisory", led="simulated", led_roi=(10, 10, 8, 8))
    engine = build_engine(cfg, source=source)
    assert engine.live_source
    assert engine.led is not None
    driver = engine.led.driver
    assert isinstance(driver, SimulatedLed)
    run_frames(engine, 3)
    assert wait_for(lambda: driver.n_commands > 0)
    engine.close()
    engine.close()
    assert source.closed
    assert not driver.is_on
    assert not engine.led.running
    assert (tmp_path / FISH_GALLERY).exists()
    assert (tmp_path / STATS_DB).exists()
    with pytest.raises(RuntimeError, match="closed"):
        engine.step()
    with pytest.raises(RuntimeError, match="closed"):
        engine.start()


def test_constructor_rejects_inconsistent_liveness(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    verifier = LivenessVerifier(seeded_schedule(1), (1, 1, 4, 4))
    with pytest.raises(ValueError, match="needs a LivenessVerifier"):
        EntropyEngine(cfg, ScriptedSource(), liveness_policy="required")
    with pytest.raises(ValueError, match="takes no verifier"):
        EntropyEngine(cfg, ScriptedSource(), liveness=verifier, liveness_policy="off")


def test_chaos_fraction_is_monotone_and_bounded() -> None:
    scores = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 1e9]
    fractions = [chaos_fraction(s) for s in scores]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0
    assert fractions[3] == 0.5
    assert all(0 <= f < 1 for f in fractions)
    assert chaos_fraction(float("nan")) == 0.0
    assert chaos_fraction(-1.0) == 0.0
