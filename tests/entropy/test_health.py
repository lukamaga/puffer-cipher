"""SP 800-90B continuous tests, the channel state machine and the domain checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom

from puffer.entropy.health import (
    AdaptiveProportionTest,
    ChannelHealth,
    CheckResult,
    FrozenFrameCheck,
    HealthConfig,
    HealthMonitor,
    RepetitionCountTest,
    apt_cutoff,
    rct_cutoff,
)
from puffer.types import HealthState, LivenessState, LivenessStatus

OK, STARTUP, DEGRADED, FAILED = (
    HealthState.OK,
    HealthState.STARTUP,
    HealthState.DEGRADED,
    HealthState.FAILED,
)


def uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 256, size=n, dtype=np.uint8)


# -- cutoffs -----------------------------------------------------------------


@pytest.mark.parametrize(("h", "cutoff"), [(2.0, 11), (1.0, 21), (0.5, 41), (8.0, 4), (3.0, 8)])
def test_rct_cutoff(h, cutoff):
    assert rct_cutoff(h) == cutoff


@pytest.mark.parametrize(
    ("h", "cutoff"), [(0.5, 410), (1.0, 311), (2.0, 177), (4.0, 62), (8.0, 13)]
)
def test_apt_cutoff_matches_sp800_90b_table_2(h, cutoff):
    assert apt_cutoff(h, 512) == cutoff


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0, 3.5, 8.0])
def test_apt_cutoff_is_critbinom(h):
    """C - 1 is the smallest k with P[Binomial(W, 2^-H) <= k] >= 1 - 2^-20."""
    k = apt_cutoff(h) - 1
    q = 1 - 2.0**-20
    assert binom.cdf(k, 512, 2.0**-h) >= q
    assert binom.cdf(k - 1, 512, 2.0**-h) < q


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_cutoffs_reject_bad_entropy(bad):
    with pytest.raises(ValueError, match="min-entropy"):
        rct_cutoff(bad)
    with pytest.raises(ValueError, match="min-entropy"):
        apt_cutoff(bad)


# -- repetition count test ---------------------------------------------------


def test_rct_fails_exactly_at_cutoff_run_length():
    base = np.arange(100, dtype=np.uint8)  # no repeats at all
    ten = np.concatenate((base, np.full(10, 7, np.uint8), base))
    eleven = np.concatenate((base, np.full(11, 7, np.uint8), base))
    assert RepetitionCountTest(2.0).update(ten) == 0
    assert RepetitionCountTest(2.0).update(eleven) == 1


def test_rct_signals_once_per_run_and_once_per_separate_run():
    t = RepetitionCountTest(2.0)
    stuck = np.concatenate((np.zeros(1000, np.uint8), [1], np.zeros(11, np.uint8)))
    assert t.update(stuck) == 2
    assert t.failures == 2


def test_rct_run_spanning_batches():
    t = RepetitionCountTest(2.0)
    assert t.update(np.array([1, 2, 3, 3, 3, 3, 3, 3], np.uint8)) == 0
    assert t.update(np.array([3, 3, 3, 3], np.uint8)) == 0  # run is now 10
    assert t.update(np.array([3], np.uint8)) == 1  # 11th repetition
    assert t.update(np.array([3, 3, 3], np.uint8)) == 0  # same run, already signalled


def test_rct_rejects_float_samples():
    with pytest.raises(TypeError, match="integer"):
        RepetitionCountTest(2.0).update(np.array([0.5, 1.5]))


# -- adaptive proportion test ------------------------------------------------


def test_apt_counts_occurrences_of_the_windows_first_sample():
    cutoff = apt_cutoff(2.0)  # 177

    def window_starting_with_zeros(k: int) -> np.ndarray:
        w = np.resize(np.arange(1, 256, dtype=np.uint8), 512)  # no zeros, each symbol <= 3x
        w[:k] = 0
        return w

    assert AdaptiveProportionTest(2.0).update(window_starting_with_zeros(cutoff - 1)) == 0
    assert AdaptiveProportionTest(2.0).update(window_starting_with_zeros(cutoff)) == 1
    # Mostly zeros, but the window's first sample is another symbol: nothing to count.
    other = np.zeros(512, np.uint8)
    other[0] = 9
    assert AdaptiveProportionTest(2.0).update(other) == 0


def test_apt_detects_biased_source(rng):
    # True min-entropy 1 bit (P[0] = 1/2) while the channel is assessed at 2 bits.
    x = np.where(rng.random(512 * 40) < 0.5, 0, rng.integers(1, 256, 512 * 40)).astype(np.uint8)
    windows_starting_with_zero = int(np.count_nonzero(x.reshape(40, 512)[:, 0] == 0))
    assert AdaptiveProportionTest(2.0).update(x) == windows_starting_with_zero > 10


def test_stuck_source_trips_both_tests():
    stuck = np.full(4096, 42, np.uint8)
    assert RepetitionCountTest(2.0).update(stuck) == 1
    assert AdaptiveProportionTest(2.0).update(stuck) == 8  # one per window


def test_no_false_positives_on_uniform_bytes(rng):
    x = uniform(rng, 1_000_000)
    assert RepetitionCountTest(2.0).update(x) == 0
    assert AdaptiveProportionTest(2.0).update(x) == 0


def test_false_positive_rate_is_calibrated_when_entropy_matches_assessment(rng):
    """Four equiprobable symbols have exactly H = 2: failures stay near alpha = 2^-20."""
    x = rng.integers(0, 4, size=1_000_000, dtype=np.uint8)
    assert RepetitionCountTest(2.0).update(x) <= 5
    assert AdaptiveProportionTest(2.0).update(x) <= 1


def _runs(pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([np.full(n, s, np.uint8) for s, n in pairs] or [np.empty(0, np.uint8)])


@settings(derandomize=True, max_examples=200, deadline=None)
@given(
    pairs=st.lists(st.tuples(st.integers(0, 3), st.integers(1, 30)), max_size=200),
    cuts=st.lists(st.integers(0, 6000), max_size=12),
)
def test_batching_equals_one_shot(pairs, cuts):
    x = _runs(pairs)
    pieces = np.split(x, sorted(c % (x.size + 1) for c in cuts))
    for make in (
        lambda: RepetitionCountTest(2.0),
        lambda: AdaptiveProportionTest(1.0, window=16, alpha_exp=4),
    ):
        one_shot, batched = make(), make()
        expected = one_shot.update(x)
        assert sum(batched.update(p) for p in pieces) == expected
        assert batched.failures == one_shot.failures


def _rct_literal(x: np.ndarray, cutoff: int) -> int:
    """SP 800-90B 4.4.1 sample by sample; one signal per run (the run's B reaching C)."""
    failures, a, b = 0, None, 0
    for s in x.tolist():
        a, b = (a, b + 1) if s == a else (s, 1)
        failures += b == cutoff
    return failures


def _apt_literal(x: np.ndarray, window: int, cutoff: int) -> int:
    """SP 800-90B 4.4.2 window by window (a trailing partial window counts too)."""
    failures = 0
    for start in range(0, x.size, window):
        w = x[start : start + window].tolist()
        failures += any(k + 1 >= cutoff for k in np.cumsum([s == w[0] for s in w]) - 1)
    return failures


@settings(derandomize=True, max_examples=200, deadline=None)
@given(
    pairs=st.lists(st.tuples(st.integers(0, 3), st.integers(1, 25)), max_size=120),
    cuts=st.lists(st.integers(0, 3000), max_size=10),
)
def test_continuous_tests_match_the_literal_sp800_90b_loops(pairs, cuts):
    x = _runs(pairs)
    pieces = np.split(x, sorted(c % (x.size + 1) for c in cuts))
    rct = RepetitionCountTest(2.0)
    apt = AdaptiveProportionTest(1.0, window=16, alpha_exp=4)
    assert sum(rct.update(p) for p in pieces) == _rct_literal(x, rct.cutoff)
    assert sum(apt.update(p) for p in pieces) == _apt_literal(x, 16, apt.cutoff)


def test_batching_equals_one_shot_on_large_mixed_stream(rng):
    x = np.concatenate([uniform(rng, 5000), np.zeros(700, np.uint8), uniform(rng, 3000)] * 3)
    cuts = np.sort(rng.integers(0, x.size, size=40))
    for make in (lambda: RepetitionCountTest(2.0), lambda: AdaptiveProportionTest(2.0)):
        expected = make().update(x)
        batched = make()
        assert sum(batched.update(p) for p in np.split(x, cuts)) == expected > 0


# -- channel state machine ---------------------------------------------------


def test_channel_lifecycle_startup_ok_failed_cooldown_startup_ok(rng):
    ch = ChannelHealth("sensor", 2.0)
    assert ch.state is STARTUP
    ch.update(uniform(rng, 512), 0.0)
    assert ch.state is STARTUP
    assert ch.report(0.0).reasons == ("start-up: 512/1024 samples",)
    ch.update(uniform(rng, 512), 0.1)
    assert ch.state is STARTUP  # this batch completed start-up but is not credited
    assert ch.credit_multiplier == 0.0
    ch.update(uniform(rng, 512), 0.2)
    assert ch.state is OK
    assert ch.credit_multiplier == 1.0

    assert ch.update(np.zeros(600, np.uint8), 1.0) > 0
    report = ch.report(1.0)
    assert report.state is FAILED
    assert report.checks["repetition_count"] == "failed"
    assert "cooldown: 5.0 s left" in report.reasons
    assert ch.credit_multiplier == 0.0

    ch.update(uniform(rng, 2048), 5.9)  # healthy again, but still cooling down
    assert ch.state is FAILED
    assert ch.report(6.0).state is STARTUP  # cooldown over: fresh start-up window
    assert ch.report(6.0).reasons == ("start-up: 0/1024 samples",)
    ch.update(uniform(rng, 1024), 6.1)
    assert ch.state is STARTUP
    ch.update(uniform(rng, 100), 6.2)
    assert ch.state is OK


def _health_op(kind: str) -> st.SearchStrategy[tuple]:
    if kind == "good":
        return st.tuples(st.just(kind), st.integers(1, 100))
    if kind == "stuck":
        return st.tuples(st.just(kind), st.integers(11, 100))
    if kind == "check":
        return st.tuples(
            st.just(kind),
            st.sampled_from(["frozen_frame", "liveness"]),
            st.sampled_from([OK, DEGRADED, FAILED]),
        )
    return st.tuples(st.just(kind))


# Mostly healthy batches, so that credit is actually reached between failures.
_OPS = st.sampled_from(["good"] * 6 + ["stuck", "check", "report"]).flatmap(_health_op)


@settings(derandomize=True, max_examples=150, deadline=None)
@given(
    steps=st.lists(st.tuples(st.floats(0.0, 0.5), _OPS), min_size=10, max_size=60),
    seed=st.integers(0, 2**32 - 1),
)
def test_credit_is_never_granted_while_failed_or_starting_up(steps, seed):
    """Credit needs: no failing check, a full cooldown since the last bad event,
    and a full start-up window of clean samples in *earlier* batches after it."""
    gen = np.random.default_rng(seed)
    cfg = HealthConfig(startup_samples=128, cooldown_s=1.0)
    ch = ChannelHealth("fish", 2.0, cfg)
    t, last_bad, clean = 0.0, -math.inf, 0
    failing: set[str] = set()
    for dt, op in steps:
        t += dt
        if op[0] == "check":
            _, name, state = op
            ch.apply(CheckResult(name, state, "x"), t)
            if state is FAILED or name in failing:
                last_bad, clean = t, 0
            failing.discard(name)
            if state is FAILED:
                failing.add(name)
        elif op[0] == "report":
            ch.report(t)
        else:
            kind, n = op
            x = uniform(gen, n) if kind == "good" else np.zeros(n, np.uint8)
            if ch.update(x, t):
                last_bad, clean = t, 0
                assert ch.credit_multiplier == 0.0
                continue
            if ch.credit_multiplier > 0:
                assert not failing
                assert t >= last_bad + cfg.cooldown_s
                assert clean >= cfg.startup_samples
                assert ch.state in (OK, DEGRADED)
            if t >= last_bad + cfg.cooldown_s:
                clean += n
        if ch.state in (STARTUP, FAILED):
            assert ch.credit_multiplier == 0.0


def test_failures_during_cooldown_restart_it():
    ch = ChannelHealth("fish", 2.0)
    ch.update(np.zeros(20, np.uint8), 0.0)
    assert ch.state is FAILED
    ch.update(np.zeros(512, np.uint8), 4.0)  # still stuck: APT fails again
    assert ch.report(8.0).state is FAILED
    assert ch.report(9.0).state is STARTUP


def test_failure_during_startup(rng):
    ch = ChannelHealth("sensor", 2.0)
    ch.update(uniform(rng, 100), 0.0)
    ch.update(np.full(11, 3, np.uint8), 0.1)
    assert ch.state is FAILED


def test_domain_checks_degrade_and_fail(rng):
    cfg = HealthConfig(startup_samples=0)
    ch = ChannelHealth("fish", 2.0, cfg)
    ch.update(uniform(rng, 10), 0.0)
    ch.update(uniform(rng, 10), 0.1)
    assert ch.state is OK
    ch.apply(CheckResult("fish_presence", DEGRADED, "1 of 3 fish visible"), 0.2)
    assert ch.state is DEGRADED
    assert ch.credit_multiplier == 0.5
    assert ch.report(0.2).reasons == ("1 of 3 fish visible",)
    ch.apply(CheckResult("fish_presence", FAILED, "no fish for 10 s"), 0.3)
    assert ch.state is FAILED
    # Cooldown counts from the moment the failing check clears, not from when it failed.
    ch.apply(CheckResult("fish_presence", OK), 20.0)
    assert ch.report(24.9).state is FAILED
    assert ch.report(25.0).state is STARTUP


def test_failing_check_blocks_recovery_until_cleared():
    ch = ChannelHealth("sensor", 2.0)
    ch.apply(CheckResult("liveness", FAILED, "liveness calibrating"), 0.0)
    assert ch.report(100.0).state is FAILED
    assert ch.report(100.0).reasons == ("liveness calibrating",)


# -- monitor: domain checks wired to channels ----------------------------------


def frozen_frames(make_frame, rng, n, start=0):
    image = uniform(rng, 48 * 64 * 3).reshape(48, 64, 3)
    return [make_frame(image, index=start + i) for i in range(n)]


def test_frozen_frame_fails_after_15_repeats(make_frame, rng):
    monitor = HealthMonitor()
    frames = frozen_frames(make_frame, rng, 16)
    for frame in frames[:15]:
        monitor.observe_frame(frame)
    assert monitor.state("sensor") is STARTUP  # 14 repeats so far
    monitor.observe_frame(frames[15])
    reports = monitor.reports()
    for channel in ("sensor", "fish"):
        assert reports[channel].state is FAILED
        assert reports[channel].checks["frozen_frame"] == "failed"
    assert not monitor.usable("sensor")


def test_near_identical_frames_count_as_frozen(make_frame, rng):
    monitor = HealthMonitor()
    base = uniform(rng, 48 * 64 * 3).reshape(48, 64, 3)
    for i in range(16):
        image = base.copy()
        image[0, 0, 0] ^= i & 1  # one subsampled pixel flips: MAD ~ 1e-3
        monitor.observe_frame(make_frame(image, index=i))
    assert monitor.state("sensor") is FAILED


@pytest.mark.parametrize("period", [2, 5, 600])
def test_looped_video_counts_as_frozen(make_frame, rng, period):
    """A bit-exact loop (``video:tank.mp4,loop=1``) never repeats the *previous* frame.

    Each frame differs from its predecessor, so only remembering recent
    frame digests reveals that the camera is showing stale frames again.
    """
    monitor = HealthMonitor()
    loop = [uniform(rng, 24 * 32 * 3).reshape(24, 32, 3) for _ in range(period)]
    for i in range(period + 14):  # one full pass, then 14 repeated frames
        monitor.observe_frame(make_frame(loop[i % period], index=i))
    assert monitor.state("sensor") is STARTUP
    monitor.observe_frame(make_frame(loop[(period + 14) % period], index=period + 14))
    reports = monitor.reports()
    for channel in ("sensor", "fish"):
        assert reports[channel].state is FAILED
        assert reports[channel].checks["frozen_frame"] == "failed"


def test_frozen_history_is_bounded(make_frame, rng):
    loop = [uniform(rng, 8 * 8 * 3).reshape(8, 8, 3) for _ in range(5)]
    check = FrozenFrameCheck(history=4)  # remembers fewer frames than the loop is long
    results = [check.observe(make_frame(loop[i % 5], index=i)) for i in range(60)]
    assert all(r.state is OK for r in results)
    assert check.consecutive == 0
    with pytest.raises(ValueError, match="frozen_history"):
        HealthConfig(frozen_history=0)


def test_duplicated_frames_are_tolerated(make_frame, rng):
    """Drivers and pulldown duplicate frames (A A B B B C C ...): never 15 in a row."""
    monitor = HealthMonitor()
    index = 0
    for _ in range(40):
        image = uniform(rng, 24 * 32 * 3).reshape(24, 32, 3)
        for _ in range(3):
            monitor.observe_frame(make_frame(image, index=index))
            index += 1
    assert monitor.reports()["sensor"].checks["frozen_frame"] == "ok"


def test_noisy_frames_are_not_frozen(make_frame, rng):
    monitor = HealthMonitor()
    for i in range(40):
        monitor.observe_frame(make_frame(uniform(rng, 48 * 64 * 3).reshape(48, 64, 3), index=i))
    assert monitor.reports()["sensor"].checks["frozen_frame"] == "ok"


def test_fish_presence(make_tracking):
    monitor = HealthMonitor(HealthConfig(expected_fish=3, startup_samples=0))
    monitor.observe_samples("fish", np.arange(10, dtype=np.uint8), 0.0)
    monitor.observe_tracking(make_tracking(0.0, n_confirmed=3))
    monitor.observe_samples("fish", np.arange(10, dtype=np.uint8), 0.1)
    assert monitor.state("fish") is OK
    monitor.observe_tracking(make_tracking(0.5, n_confirmed=2))
    assert monitor.state("fish") is DEGRADED
    assert monitor.credit_multiplier("fish") == 0.5
    assert monitor.state("sensor") is not DEGRADED  # presence only affects the fish channel
    monitor.observe_tracking(make_tracking(1.0, n_confirmed=0))
    assert monitor.state("fish") is DEGRADED  # briefly hidden
    monitor.observe_tracking(make_tracking(10.9, n_confirmed=0))
    assert monitor.state("fish") is FAILED
    assert "no fish for 10 s" in monitor.reports()["fish"].reasons


def test_source_timeout(make_frame, rng):
    monitor = HealthMonitor()
    monitor.observe_frame(make_frame(uniform(rng, 48 * 64 * 3).reshape(48, 64, 3), t=100.0))
    monitor.check_timeout(104.9)
    assert monitor.state("sensor") is STARTUP
    monitor.check_timeout(105.0)
    reports = monitor.reports()
    assert reports["sensor"].state is FAILED
    assert reports["fish"].checks["source_timeout"] == "failed"
    assert reports["sensor"].reasons == ("no frame for 5.0 s",)


def test_timeout_without_any_frame_counts_from_first_check():
    monitor = HealthMonitor()
    monitor.check_timeout(50.0)
    assert monitor.state("sensor") is STARTUP
    monitor.check_timeout(55.0)
    assert monitor.state("sensor") is FAILED


def test_low_entropy_estimate_fails_only_that_channel():
    monitor = HealthMonitor()
    monitor.observe_entropy_estimate("sensor", 0.3, 1.0)
    assert monitor.state("sensor") is FAILED
    assert monitor.state("fish") is STARTUP
    assert monitor.reports()["sensor"].reasons == ("low entropy: 0.30 bits/sample (min 0.5)",)


def test_liveness_gating(rng):
    monitor = HealthMonitor(HealthConfig(startup_samples=0))
    for t in (0.0, 0.1):
        monitor.observe_samples("sensor", uniform(rng, 100), t)
    assert monitor.state("sensor") is OK

    monitor.observe_liveness(LivenessStatus(LivenessState.FAILED), required=False)
    assert monitor.state("sensor") is OK  # not required: informational only

    monitor.observe_liveness(LivenessStatus(LivenessState.CALIBRATING), required=True, t=1.0)
    assert monitor.state("sensor") is FAILED
    assert monitor.state("fish") is FAILED
    assert monitor.reports()["sensor"].reasons == ("liveness calibrating",)

    monitor.observe_liveness(LivenessStatus(LivenessState.VERIFIED), required=True, t=2.0)
    monitor.observe_samples("sensor", uniform(rng, 100), 6.9)
    assert monitor.state("sensor") is FAILED  # cooldown runs until t = 7.0
    monitor.observe_samples("sensor", uniform(rng, 100), 7.0)  # fresh start-up (0 samples)
    monitor.observe_samples("sensor", uniform(rng, 100), 7.1)
    assert monitor.state("sensor") is OK


def test_reports_overall_and_system_channel(rng):
    monitor = HealthMonitor(HealthConfig(startup_samples=0))
    assert monitor.overall() is STARTUP
    reports = monitor.reports()
    assert set(reports) == {"sensor", "fish", "system"}
    assert reports["system"].state is OK
    assert monitor.usable("system")
    assert monitor.credit_multiplier("system") == 1.0

    for t in (0.0, 0.1):
        monitor.observe_samples("sensor", uniform(rng, 100), t)
        monitor.observe_samples("fish", uniform(rng, 100), t)
    assert monitor.overall() is OK
    monitor.observe_entropy_estimate("fish", 0.0, 0.2)
    assert monitor.overall() is DEGRADED  # sensor still delivers camera entropy
    monitor.observe_entropy_estimate("sensor", 0.0, 0.2)
    assert monitor.overall() is FAILED


def test_unknown_channel_is_rejected():
    monitor = HealthMonitor()
    with pytest.raises(KeyError, match="unknown camera channel"):
        monitor.observe_samples("lava", np.zeros(3, np.uint8), 0.0)
    with pytest.raises(KeyError):
        monitor.credit_multiplier("lava")


def test_config_validation():
    with pytest.raises(ValueError, match="expected_fish"):
        HealthConfig(min_fish=2, expected_fish=1)
    with pytest.raises(ValueError, match="cooldown_s"):
        HealthConfig(cooldown_s=-1)
    assert HealthConfig().assessed_h("fish") == 2.0
    with pytest.raises(KeyError):
        HealthConfig().assessed_h("system")
