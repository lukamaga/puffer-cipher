"""LivenessVerifier end to end on synthetic video: live, replayed, degraded."""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from puffer.liveness import (
    ChallengeSchedule,
    LedDetector,
    LivenessVerifier,
    challenge_commitment,
    confidence_bits,
    observation_commitment,
)
from puffer.types import Frame, LightMode, LivenessState, LivenessStatus

from .helpers import Scene, fraction, render, run, seeded_schedule

VERIFIED, FAILED, CALIBRATING = (
    LivenessState.VERIFIED,
    LivenessState.FAILED,
    LivenessState.CALIBRATING,
)
# calibration (4 s) + one full 64-bit window (12.8 s) + latency + one symbol
FIRST_VERDICT_S = 18.0


def frames_for(
    led: ChallengeSchedule,
    seed: int,
    *,
    duration_s: float = 21.0,
    decoy: bool = True,
    **kwargs: Any,
) -> list[Frame]:
    """Live video of ``led``, with an independent decoy blinking elsewhere."""
    decoy_fn = seeded_schedule(seed + 500).state_at if decoy else None
    return list(
        render(
            led.state_at,
            rng=np.random.default_rng(seed),
            duration_s=duration_s,
            decoy=decoy_fn,
            **kwargs,
        )
    )


def after(frames: list[Frame], statuses: list[LivenessStatus], t: float) -> list[LivenessStatus]:
    return [s for f, s in zip(frames, statuses, strict=True) if f.timestamp >= t]


# --------------------------------------------------------------------- live


@pytest.mark.parametrize("fps", [30.0, 15.0])
@pytest.mark.parametrize("latency", [0.0, 0.05, 0.3])
def test_live_video_is_verified(fps, latency):
    challenge = seeded_schedule(1)
    frames = frames_for(challenge, 2, fps=fps, latency_s=latency)
    verifier = LivenessVerifier(challenge)
    statuses = run(verifier, frames)

    assert len(statuses) == len(frames)  # a status for every frame
    assert statuses[0].state is CALIBRATING
    tail = after(frames, statuses, FIRST_VERDICT_S)
    assert fraction(tail, VERIFIED) == 1.0
    last = statuses[-1]
    # the effective latency includes half the 4 ms exposure
    assert abs(last.lag_s - (latency + 0.002)) <= 1.0 / fps
    assert last.match_ratio == 1.0
    assert last.confidence_bits > 50
    assert last.expected == last.observed
    assert len(last.expected) == 32
    assert last.window_end - last.window_start == pytest.approx(64 * 0.2)
    assert last.window_end <= frames[-1].timestamp


def test_never_verified_before_a_full_window():
    challenge = seeded_schedule(3)
    frames = frames_for(challenge, 4)
    statuses = run(LivenessVerifier(challenge), frames)
    early = [s for f, s in zip(frames, statuses, strict=True) if f.timestamp < 4.0 + 12.8]
    assert all(s.state is CALIBRATING for s in early)


def test_configured_roi_skips_calibration():
    challenge = seeded_schedule(5)
    scene = Scene()
    frames = frames_for(challenge, 6, duration_s=16.0, latency_s=0.1)
    verifier = LivenessVerifier(challenge, scene.led_roi, auto_calibrate=False)
    statuses = run(verifier, frames)
    assert verifier.calibration is None
    assert verifier.schedule is challenge
    assert verifier.roi == scene.led_roi
    assert fraction(after(frames, statuses, 14.0), VERIFIED) == 1.0
    # a pre-built detector works the same way
    detector = LedDetector(scene.led_roi)
    assert LivenessVerifier(challenge, detector).detector is detector


def test_night_grayscale_video():
    challenge = seeded_schedule(7)
    frames = frames_for(
        challenge, 8, latency_s=0.05, scene=Scene(mode=LightMode.NIGHT, noise_sigma=6.0)
    )
    assert all(np.array_equal(f.image[..., 0], f.image[..., 2]) for f in frames[:5])
    statuses = run(LivenessVerifier(challenge), frames)
    assert fraction(after(frames, statuses, FIRST_VERDICT_S), VERIFIED) == 1.0


def test_day_to_night_switch_keeps_verifying():
    challenge = seeded_schedule(9)
    day = frames_for(challenge, 10, duration_s=22.0, latency_s=0.05)
    night = list(
        render(
            challenge.state_at,
            rng=np.random.default_rng(11),
            start_s=22.0,
            duration_s=8.0,
            latency_s=0.05,
            scene=Scene(mode=LightMode.NIGHT, noise_sigma=5.0),
        )
    )
    statuses = run(LivenessVerifier(challenge), day + night)
    assert fraction(statuses[-len(night) :], VERIFIED) > 0.95


def test_slow_lighting_drift():
    challenge = seeded_schedule(12)
    frames = frames_for(
        challenge, 13, duration_s=40.0, scene=Scene(drift_amplitude=45.0, drift_period_s=25.0)
    )
    statuses = run(LivenessVerifier(challenge), frames)
    assert fraction(after(frames, statuses, FIRST_VERDICT_S), VERIFIED) == 1.0


@pytest.mark.parametrize("fps", [30.0, 15.0])
def test_frame_drops_and_jittered_timestamps(fps):
    challenge = seeded_schedule(14)
    frames = frames_for(
        challenge,
        15,
        fps=fps,
        duration_s=24.0,
        latency_s=0.12,
        drop_prob=0.15,
        capture_jitter_s=0.2 / fps,
        timestamp_noise_s=0.002,
    )
    statuses = run(LivenessVerifier(challenge), frames)
    tail = after(frames, statuses, FIRST_VERDICT_S + 1.0)
    assert fraction(tail, VERIFIED) > 0.98
    assert abs(statuses[-1].lag_s - 0.122) <= 1.0 / fps


@pytest.mark.parametrize(("fps", "min_verified"), [(30.0, 0.99), (15.0, 0.9)])
@pytest.mark.parametrize("seed", [16, 17])
def test_low_false_reject_with_corrupted_frames(fps, min_verified, seed):
    """5% of frames hit by occlusion, glare or heavy noise.

    At 15 fps a 100 ms chip holds only ~1.5 frames, so one bad frame can cost
    a bit; the 64-bit window absorbs up to six such losses.
    """
    challenge = seeded_schedule(seed)
    frames = frames_for(challenge, seed + 1, fps=fps, duration_s=35.0, corrupt_prob=0.05)
    statuses = run(LivenessVerifier(challenge), frames)
    assert fraction(after(frames, statuses, FIRST_VERDICT_S), VERIFIED) >= min_verified


def test_nrz_encoding_also_verifies():
    challenge = seeded_schedule(18, encoding="nrz")
    frames = frames_for(challenge, 19, duration_s=19.0, latency_s=0.05)
    statuses = run(LivenessVerifier(challenge), frames)
    assert statuses[-1].state is VERIFIED


# ------------------------------------------------------------------- attacks


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_replayed_video_fails(seed):
    """Video of schedule A (a recording) checked against the fresh schedule B."""
    recorded, fresh = seeded_schedule(seed), seeded_schedule(seed + 1000)
    frames = frames_for(recorded, seed, latency_s=0.05)
    statuses = run(LivenessVerifier(fresh), frames)
    assert not any(s.state is VERIFIED for s in statuses)
    assert statuses[-1].state is FAILED


def replay_trials(n_trials: int) -> tuple[int, int, float, list[float]]:
    """Replays judged at the bit level (LED region known, so every window is scored)."""
    scene = Scene(fish=False, decoy_center=None, width=64, height=48, led_center=(40, 20))
    verified = judged = 0
    max_conf = 0.0
    ratios = []
    for seed in range(n_trials):
        recorded, fresh = seeded_schedule(100 + seed), seeded_schedule(5000 + seed)
        frames = frames_for(recorded, seed, decoy=False, duration_s=16.0, scene=scene)
        statuses = run(LivenessVerifier(fresh, scene.led_roi), frames)
        scored = [s for s in statuses if s.state is not CALIBRATING]
        judged += len(scored)
        verified += sum(s.state is VERIFIED for s in scored)
        max_conf = max(max_conf, *(s.confidence_bits for s in scored))
        ratios.append(scored[-1].match_ratio)
    return verified, judged, max_conf, ratios


def test_replay_false_accept_is_near_zero():
    verified, judged, max_conf, ratios = replay_trials(6)
    assert judged > 500
    assert verified == 0
    assert max_conf < 16  # nowhere near the 24-bit bar
    assert 0.3 < float(np.mean(ratios)) < 0.7  # chance level


@pytest.mark.slow
def test_replay_false_accept_many_trials():
    verified, _judged, max_conf, ratios = replay_trials(60)
    assert verified == 0
    assert max_conf < 20
    assert abs(float(np.mean(ratios)) - 0.5) < 0.05


def test_lag_beyond_bound_is_rejected():
    challenge = seeded_schedule(23)
    frames = frames_for(challenge, 24, latency_s=0.8)  # pre-rendered, too old to be live
    statuses = run(LivenessVerifier(challenge, max_lag_s=0.5), frames)
    assert not any(s.state is VERIFIED for s in statuses)
    # the same video is fine for a verifier that accepts such a slow camera
    statuses = run(LivenessVerifier(challenge, max_lag_s=1.0), frames)
    assert statuses[-1].state is VERIFIED


def test_led_covered_after_verification_fails_quickly():
    challenge = seeded_schedule(25)
    frames = frames_for(challenge, 26, duration_s=30.0, covered=lambda t: t >= 20.0)
    statuses = run(LivenessVerifier(challenge), frames)
    assert all(s.state is VERIFIED for s in after(frames, statuses, FIRST_VERDICT_S)[:30])
    # > 6 of 64 bits lost after ~1.4 s; allow 3 s
    assert all(s.state is FAILED for s in after(frames, statuses, 23.0))


def test_led_covered_from_the_start_fails():
    challenge = seeded_schedule(27)
    frames = frames_for(challenge, 28, covered=lambda t: True)
    statuses = run(LivenessVerifier(challenge), frames)
    assert not any(s.state is VERIFIED for s in statuses)
    assert statuses[-1].state is FAILED  # "LED hidden", not "still calibrating"


def test_led_covered_with_configured_roi_fails_after_one_window():
    challenge = seeded_schedule(29)
    frames = frames_for(challenge, 30, duration_s=16.0, covered=lambda t: True)
    statuses = run(LivenessVerifier(challenge, Scene().led_roi), frames)
    assert statuses[-1].state is FAILED
    assert statuses[-1].observed == "?" * 32  # nothing decodable


def test_frozen_feed_fails():
    """Frames stop arriving (or repeat): the gap holds no evidence and counts against."""
    challenge = seeded_schedule(31)
    frames = frames_for(challenge, 32, duration_s=20.0)
    verifier = LivenessVerifier(challenge)
    assert run(verifier, frames)[-1].state is VERIFIED
    # the same last image keeps coming with fresh timestamps: the LED never changes
    stuck = [
        Frame(f.index + i, frames[-1].timestamp + i / 30, 0.0, frames[-1].image)
        for i, f in enumerate(frames[:90], start=1)
    ]
    assert run(verifier, stuck)[-1].state is FAILED


def test_recovery_from_failed_to_verified():
    live, recording = seeded_schedule(33), seeded_schedule(34)

    def led(t: float) -> bool:
        return (recording if 22.0 <= t < 34.0 else live).state_at(t)

    frames = list(
        render(
            led,
            rng=np.random.default_rng(35),
            duration_s=52.0,
            latency_s=0.08,
            decoy=seeded_schedule(36).state_at,
        )
    )
    statuses = run(LivenessVerifier(live), frames)
    states = [s.state for s in statuses]
    at = {round(f.timestamp, 3): s for f, s in zip(frames, states, strict=True)}
    assert at[21.0] is VERIFIED
    assert at[30.0] is FAILED  # the replayed segment
    assert at[51.0] is VERIFIED  # live feed back: a fresh window of good bits
    # the transitions happen once each, without flapping
    changes = [b for a, b in itertools.pairwise(states) if a is not b]
    assert changes == [VERIFIED, FAILED, VERIFIED]


def test_recalibrates_after_the_camera_moves():
    challenge = seeded_schedule(37)
    before = frames_for(challenge, 38, duration_s=22.0)
    moved_scene = Scene(led_center=(60, 40), decoy_center=None, fish=False)
    moved = list(
        render(
            challenge.state_at,
            rng=np.random.default_rng(39),
            start_s=22.0,
            duration_s=33.0,
            scene=moved_scene,
        )
    )
    verifier = LivenessVerifier(challenge, recalibrate_after_s=5.0)
    statuses = run(verifier, before + moved)
    assert statuses[len(before) - 1].state is VERIFIED
    assert FAILED in {s.state for s in statuses[len(before) :]}
    assert statuses[-1].state is VERIFIED
    assert verifier.roi is not None
    x, y, w, h = verifier.roi
    assert abs(x + w / 2 - 60) <= 5
    assert abs(y + h / 2 - 40) <= 5


@pytest.mark.parametrize("roi", [Scene().led_roi, None])
def test_timestamps_older_than_retention_fail_without_crashing(roi):
    challenge = ChallengeSchedule(retention_s=60.0, random_bytes=np.random.default_rng(40).bytes)
    challenge.state_at(10_000.0)  # the LED controller is far ahead of these "frames"
    frames = frames_for(seeded_schedule(41), 42, duration_s=6.0)
    statuses = run(LivenessVerifier(challenge, roi), frames)
    assert statuses[-1].state is FAILED


# --------------------------------------------------------------- mechanics


def test_non_increasing_timestamps_are_ignored():
    challenge = seeded_schedule(43)
    frames = frames_for(challenge, 44, duration_s=1.0)
    verifier = LivenessVerifier(challenge)
    status = verifier.process(frames[10])
    assert verifier.process(frames[5]) is status
    assert verifier.process(frames[10]) is status


def test_frames_that_cannot_be_checked_never_inherit_verified():
    """The status is the verdict on *this* frame, and callers credit its entropy on it.

    A frame whose timestamp does not advance is not checked against the
    challenge, so returning the previous VERIFIED would let a feed replayed
    with old timestamps (or a camera whose clock stopped) earn credit forever.
    """
    challenge = seeded_schedule(53)
    frames = frames_for(challenge, 54, duration_s=20.0)
    live, continuation = frames[:-30], frames[-30:]
    verifier = LivenessVerifier(challenge)
    assert run(verifier, live)[-1].state is VERIFIED

    blank = [dataclasses.replace(f, image=np.zeros_like(f.image)) for f in live[-90:]]
    stale = run(verifier, blank)  # black frames carrying already-seen timestamps
    assert all(s.state is FAILED for s in stale)
    assert verifier.status.state is FAILED
    frozen_clock = dataclasses.replace(live[-1], timestamp=math.nan)
    assert verifier.process(frozen_clock).state is FAILED

    # the ignored frames taught the verifier nothing: the live feed carries on
    assert all(s.state is VERIFIED for s in run(verifier, continuation))


def test_an_error_while_processing_does_not_leave_a_stale_verdict(monkeypatch):
    challenge = seeded_schedule(55)
    frames = frames_for(challenge, 56, duration_s=19.0)
    verifier = LivenessVerifier(challenge)
    assert run(verifier, frames)[-1].state is VERIFIED

    def broken(frame: Frame) -> None:
        raise RuntimeError("camera driver returned garbage")

    monkeypatch.setattr(verifier.detector, "observe", broken)
    later = dataclasses.replace(frames[-1], timestamp=frames[-1].timestamp + 1 / 30)
    with pytest.raises(RuntimeError, match="garbage"):
        verifier.process(later)
    assert verifier.status.state is FAILED


def test_resolution_change_fails_closed_and_recalibrates():
    """A smaller picture cannot contain the old LED region: FAILED at once, never an exception."""
    challenge = seeded_schedule(57)
    before = frames_for(challenge, 58, duration_s=20.0)
    smaller = Scene(width=96, height=64, led_center=(30, 30), decoy_center=None, fish=False)
    after_change = list(
        render(
            challenge.state_at,
            rng=np.random.default_rng(59),
            start_s=20.0,
            duration_s=30.0,
            scene=smaller,
        )
    )
    verifier = LivenessVerifier(challenge, recalibrate_after_s=5.0)
    statuses = run(verifier, before + after_change)
    assert statuses[len(before) - 1].state is VERIFIED
    assert statuses[len(before)].state is FAILED
    assert statuses[-1].state is VERIFIED
    x, y, w, h = verifier.roi or (0, 0, 0, 0)
    assert abs(x + w / 2 - 30) <= 5
    assert abs(y + h / 2 - 30) <= 5


def test_configured_roi_outside_a_smaller_picture_fails_without_raising():
    challenge = seeded_schedule(60)
    frames = frames_for(challenge, 61, duration_s=16.0)
    verifier = LivenessVerifier(challenge, Scene().led_roi)
    assert run(verifier, frames)[-1].state is VERIFIED
    small = dataclasses.replace(
        frames[-1], timestamp=frames[-1].timestamp + 1 / 30, image=frames[-1].image[:16, :16]
    )
    assert verifier.process(small).state is FAILED
    assert verifier.status.state is FAILED


def test_resolution_change_during_calibration_restarts_it():
    challenge = seeded_schedule(62)
    first = frames_for(challenge, 63, duration_s=2.0)
    bigger = Scene(width=160, height=120, led_center=(120, 30), decoy_center=None)
    rest = list(
        render(
            challenge.state_at,
            rng=np.random.default_rng(64),
            start_s=2.0,
            duration_s=20.0,
            scene=bigger,
        )
    )
    statuses = run(LivenessVerifier(challenge), first + rest)
    assert statuses[-1].state is VERIFIED


def test_processing_is_cheap_per_frame():
    challenge = seeded_schedule(45)
    frames = frames_for(challenge, 46, duration_s=20.0)
    verifier = LivenessVerifier(challenge)
    durations = []
    for f in frames:
        start = time.perf_counter()
        verifier.process(f)
        durations.append(time.perf_counter() - start)
    steady = durations[-200:]  # full window, lag searches included
    assert float(np.median(steady)) < 0.005


def test_reset_starts_over():
    challenge = seeded_schedule(47)
    frames = frames_for(challenge, 48, duration_s=19.0)
    verifier = LivenessVerifier(challenge)
    assert run(verifier, frames)[-1].state is VERIFIED
    verifier.reset()
    assert verifier.roi is None  # it came from calibration
    assert verifier.status.state is CALIBRATING
    assert verifier.proof() is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_bits": 0}, "window_bits"),
        ({"min_match": 0.0}, "min_match"),
        ({"min_confidence_bits": -1.0}, "min_confidence_bits"),
        ({"max_lag_s": -0.1}, "max_lag_s"),
        ({"lag_step_s": 0.0}, "lag_step_s"),
        ({"lag_oversample": 0}, "lag_oversample"),
        ({"calibration_s": 0.0}, "calibration_s"),
        ({"edge_guard_s": 0.06}, "edge_guard_s"),
        ({"recalibrate_after_s": 0.0}, "recalibrate_after_s"),
        ({"window_bits": 5000}, "retention"),
        ({"auto_calibrate": False}, "auto_calibrate"),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LivenessVerifier(seeded_schedule(0), **kwargs)


# ---------------------------------------------------------- confidence math


def direct_confidence(matches: int, n: int, n_lags: int) -> float:
    """Exact rational binomial tail, independent of the implementation under test."""
    tail = Fraction(sum(math.comb(n, k) for k in range(matches, n + 1)), 2**n)
    return max(0.0, -math.log2(min(Fraction(1), tail * n_lags)))


@pytest.mark.parametrize(
    ("matches", "n", "n_lags"),
    [
        (64, 64, 1),
        (64, 64, 61),
        (58, 64, 61),
        (58, 64, 31),
        (40, 64, 1),
        (32, 64, 1),
        (10, 10, 4),
        (5, 10, 2),
        (900, 1000, 100),
        (1000, 1000, 1),
    ],
)
def test_confidence_bits_matches_direct_binomial(matches, n, n_lags):
    assert confidence_bits(matches, n, n_lags) == pytest.approx(
        direct_confidence(matches, n, n_lags), rel=1e-9, abs=1e-9
    )


def test_confidence_bits_edge_cases():
    assert confidence_bits(0, 64, 1) == 0.0
    assert confidence_bits(0, 0, 1) == 0.0
    assert confidence_bits(3, 64, 10) == 0.0  # clamped: never negative
    assert confidence_bits(2000, 2000) == pytest.approx(2000.0)  # no underflow
    for bad, message in [((-1, 4, 1), "matches"), ((5, 4, 1), "matches"), ((2, 4, 0), "n_lags")]:
        with pytest.raises(ValueError, match=message):
            confidence_bits(*bad)


def test_status_confidence_uses_the_lag_count():
    challenge = seeded_schedule(49)
    frames = frames_for(challenge, 50)
    verifier = LivenessVerifier(challenge)
    last = run(verifier, frames)[-1]
    assert verifier.n_lags == 61  # 0..0.5 s in quarter-frame steps at 30 fps
    matches = round(last.match_ratio * 64)
    assert last.confidence_bits == pytest.approx(confidence_bits(matches, 64, 61))


# -------------------------------------------------------------------- proof


def test_proof_binds_the_window():
    challenge = seeded_schedule(51)
    frames = frames_for(challenge, 52, latency_s=0.05)
    verifier = LivenessVerifier(challenge)
    statuses = []
    for f in frames:
        statuses.append(verifier.process(f))
        if statuses[-1].state is CALIBRATING:
            assert verifier.proof() is None
    proof = verifier.proof()
    assert proof is not None
    assert verifier.proof() is proof  # cached until the next window
    last = statuses[-1]
    assert proof.n_bits == 64
    assert proof.matches == 64
    assert proof.window_start == pytest.approx(last.window_start)
    assert proof.lag_s == pytest.approx(last.lag_s)
    bits = challenge.bits(proof.first_symbol, proof.first_symbol + proof.n_bits)
    assert proof.challenge_commitment == challenge_commitment(proof.first_symbol, bits)
    assert proof.observation_commitment == observation_commitment(proof.first_symbol, bits)
    assert "".join(map(str, bits[-32:])) == last.expected

    summary = proof.to_dict()
    assert all(isinstance(v, int | str) for v in summary.values())
    assert summary["confidence_bits"] == math.floor(proof.confidence_bits)
    assert summary["n_bits"] == 64
