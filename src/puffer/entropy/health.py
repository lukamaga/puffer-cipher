"""Continuous health tests for the camera entropy channels.

Why this module exists
----------------------
Entropy *credit* (DESIGN.md section 4.3) is only meaningful while the source
behaves like the source that was assessed. Cameras fail in boring ways - a USB
hiccup repeats the last frame, auto-exposure saturates the sensor, the fish
hide, someone points the camera at a looping video - and every one of those
failures produces bytes that still *look* fine after hashing. The pool would
happily keep crediting them. This module watches the raw sample streams and
the video itself, and withdraws credit the moment something looks wrong.

Two layers of tests run per channel:

* **SP 800-90B section 4.4 continuous tests.** The Repetition Count Test
  (4.4.1) catches a source that gets *stuck* on one symbol; the Adaptive
  Proportion Test (4.4.2) catches a large *loss of entropy* (one symbol
  becoming far too frequent). Both are parameterised by the channel's
  assessed min-entropy ``H`` and a false-positive rate ``alpha = 2^-20``, so
  a healthy source trips them only about once per million windows.
* **Domain checks** that the generic tests cannot see: frozen video,
  missing fish, a low rolling entropy estimate, a source that stopped
  delivering frames, and the LED anti-replay liveness gate.

Channel state machine
---------------------
::

    STARTUP --(start-up window passes)--> OK <--(check)--> DEGRADED
       ^                                   |                  |
       |                              any failure        any failure
       |                                   v                  v
       +-------- cooldown elapsed ------ FAILED <-------------+
                                    (failures while FAILED restart the cooldown)

Recovery from FAILED deliberately costs a cooldown *and* a fresh start-up
window with freshly reset tests: a source that flaps between broken and
working must re-earn trust each time instead of being credited the instant
one batch looks normal. Credit multipliers are OK 1.0, DEGRADED 0.5,
STARTUP/FAILED 0.0. A batch is credited only if the channel was already out
of start-up when the batch *began*; the batch that completes the start-up
window is therefore never credited (see :meth:`ChannelHealth.update`).

All timestamps are on the frame clock (``Frame.timestamp`` /
``TrackingResult.t``), never wall time, so recorded video and the simulator
behave exactly like a live camera.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import binom

from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    Frame,
    HealthReport,
    HealthState,
    LivenessState,
    LivenessStatus,
    TrackingResult,
)

__all__ = [
    "AdaptiveProportionTest",
    "ChannelHealth",
    "CheckResult",
    "FishPresenceCheck",
    "FrozenFrameCheck",
    "HealthConfig",
    "HealthMonitor",
    "LivenessCheck",
    "LowEntropyCheck",
    "RepetitionCountTest",
    "SourceTimeoutCheck",
    "apt_cutoff",
    "rct_cutoff",
]

# Check names exactly as listed in DESIGN.md section 4.2 (shown on the dashboard).
CHECK_RCT = "repetition_count"
CHECK_APT = "adaptive_proportion"
CHECK_FROZEN = "frozen_frame"
CHECK_FISH_PRESENCE = "fish_presence"
CHECK_LOW_ENTROPY = "low_entropy"
CHECK_SOURCE_TIMEOUT = "source_timeout"
CHECK_LIVENESS = "liveness"


def _symbols(samples: npt.ArrayLike) -> npt.NDArray[Any]:
    """Flatten ``samples`` and insist on integer symbols (empty input is allowed)."""
    x = np.asarray(samples).ravel()
    if x.size and not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"health tests need integer symbols, got dtype {x.dtype}")
    return x


def _check_h(h: float) -> None:
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"assessed min-entropy must be a positive number of bits, got {h!r}")


# --------------------------------------------------------------------------
# SP 800-90B section 4.4 continuous tests
# --------------------------------------------------------------------------


def rct_cutoff(h: float, alpha_exp: int = 20) -> int:
    """Repetition Count Test cutoff ``C = 1 + ceil(-log2(alpha) / H)``, alpha = 2^-alpha_exp.

    A source with min-entropy ``H`` per sample repeats a symbol ``C - 1``
    more times with probability at most ``2^(-H (C-1)) <= alpha``.
    """
    _check_h(h)
    if alpha_exp <= 0:
        raise ValueError("alpha_exp must be positive")
    return 1 + math.ceil(alpha_exp / h)


def apt_cutoff(h: float, window: int = 512, alpha_exp: int = 20) -> int:
    """Adaptive Proportion Test cutoff ``C = 1 + CRITBINOM(W, 2^-H, 1 - alpha)``.

    ``CRITBINOM(n, p, q)`` is the smallest ``k`` with ``P[Binomial(n, p) <= k] >= q``,
    which is exactly ``scipy.stats.binom.ppf(q, n, p)``. For ``W = 512`` this
    reproduces SP 800-90B Table 2 (H = 0.5, 1, 2, 4, 8 -> 410, 311, 177, 62, 13).
    """
    _check_h(h)
    if window < 2 or alpha_exp <= 0:
        raise ValueError("window must be >= 2 and alpha_exp positive")
    return 1 + int(binom.ppf(1.0 - 2.0**-alpha_exp, window, 2.0**-h))


class RepetitionCountTest:
    """SP 800-90B section 4.4.1 Repetition Count Test, stateful across batches.

    The standard's loop - keep the current value ``A`` and a run counter ``B``;
    increment ``B`` on a repeat, signal an error when ``B`` reaches ``C`` - is
    evaluated per *run* with numpy instead of per sample, so a 4096-sample
    sensor batch costs a handful of vector operations. A run that continues
    across ``update`` calls is carried over, so feeding a stream in any
    batching gives exactly the same failures as feeding it at once. Each run
    signals at most one failure (the moment ``B`` equals ``C``), which is the
    literal reading of the standard.
    """

    def __init__(self, h: float, alpha_exp: int = 20) -> None:
        self.h = h
        self.cutoff = rct_cutoff(h, alpha_exp)
        self.failures = 0
        self.samples_seen = 0
        self._last: int | None = None
        self._run = 0

    def reset(self) -> None:
        """Forget all state (used for the fresh start-up after a cooldown)."""
        self.failures = 0
        self.samples_seen = 0
        self._last = None
        self._run = 0

    def update(self, samples: npt.ArrayLike) -> int:
        """Feed a batch of symbols; return the number of failures it triggered."""
        x = _symbols(samples)
        if x.size == 0:
            return 0
        starts = np.concatenate(([0], np.flatnonzero(x[1:] != x[:-1]) + 1))
        lengths = np.diff(np.append(starts, x.size))
        offsets = np.zeros(lengths.size, dtype=np.int64)
        if self._last is not None and int(x[0]) == self._last:
            offsets[0] = self._run
        totals = offsets + lengths
        failures = int(np.count_nonzero((offsets < self.cutoff) & (totals >= self.cutoff)))
        self._last = int(x[-1])
        # Capping keeps the state bounded; any B >= C behaves identically.
        self._run = int(min(totals[-1], self.cutoff))
        self.failures += failures
        self.samples_seen += int(x.size)
        return failures


class AdaptiveProportionTest:
    """SP 800-90B section 4.4.2 Adaptive Proportion Test, stateful across batches.

    Samples are cut into consecutive, non-overlapping windows of ``W``
    samples. In each window the test counts how often the window's *first*
    sample occurs (including itself) and signals an error when the count
    reaches ``C``. Complete windows inside a batch are evaluated as one 2-D
    numpy comparison; a window that straddles batches keeps its reference
    symbol and partial count, so batching never changes the result. A window
    signals at most one failure, at the sample where its count reaches ``C``.
    """

    def __init__(self, h: float, window: int = 512, alpha_exp: int = 20) -> None:
        self.h = h
        self.window = window
        self.cutoff = apt_cutoff(h, window, alpha_exp)
        self.failures = 0
        self.samples_seen = 0
        self._ref = 0
        self._count = 0
        self._pos = 0  # samples consumed in the open window; 0 = no open window
        self._flagged = False

    def reset(self) -> None:
        """Forget all state (used for the fresh start-up after a cooldown)."""
        self.failures = 0
        self.samples_seen = 0
        self._ref = 0
        self._count = 0
        self._pos = 0
        self._flagged = False

    def update(self, samples: npt.ArrayLike) -> int:
        """Feed a batch of symbols; return the number of failures it triggered."""
        x = _symbols(samples)
        if x.size == 0:
            return 0
        w, c = self.window, self.cutoff
        failures = 0
        i = 0
        if self._pos:
            i = min(w - self._pos, x.size)
            self._count += int(np.count_nonzero(x[:i] == self._ref))
            self._pos = (self._pos + i) % w
            if not self._flagged and self._count >= c:
                failures += 1
                self._flagged = True
        rest = x[i:]
        n_full = rest.size // w
        if n_full:
            blocks = rest[: n_full * w].reshape(n_full, w)
            counts = np.count_nonzero(blocks == blocks[:, :1], axis=1)
            failures += int(np.count_nonzero(counts >= c))
        tail = rest[n_full * w :]
        if tail.size:
            self._ref = int(tail[0])
            self._count = int(np.count_nonzero(tail == tail[0]))
            self._pos = int(tail.size)
            self._flagged = self._count >= c
            failures += int(self._flagged)
        self.failures += failures
        self.samples_seen += int(x.size)
        return failures


# --------------------------------------------------------------------------
# Configuration and domain checks
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HealthConfig:
    """Health thresholds. Defaults are the values in DESIGN.md section 4.2.

    ``sensor_h`` / ``fish_h`` are the *assessed* per-sample min-entropies
    that parameterise RCT/APT. They are deliberately lower than what a good
    camera delivers: the continuous tests must not false-alarm on a healthy
    source, while still catching a stuck or heavily biased one.
    """

    sensor_h: float = 2.0
    fish_h: float = 2.0
    alpha_exp: int = 20
    apt_window: int = 512
    startup_samples: int = 1024
    cooldown_s: float = 5.0
    degraded_multiplier: float = 0.5
    frozen_mad_threshold: float = 0.05
    frozen_frames: int = 15
    min_fish: int = 1
    expected_fish: int | None = None
    fish_absence_s: float = 10.0
    min_bits_per_sample: float = 0.5
    source_timeout_s: float = 5.0
    frozen_history: int = 2048
    """Frames whose digests the frozen-frame check remembers (catches loops up to this length)."""

    def __post_init__(self) -> None:
        _check_h(self.sensor_h)
        _check_h(self.fish_h)
        if self.startup_samples < 0 or self.frozen_frames < 1 or self.min_fish < 0:
            raise ValueError("startup_samples, frozen_frames and min_fish must be non-negative")
        if self.frozen_history < 1:
            raise ValueError("frozen_history must be at least 1 frame")
        if self.expected_fish is not None and self.expected_fish < self.min_fish:
            raise ValueError("expected_fish must be >= min_fish")
        if not 0.0 <= self.degraded_multiplier <= 1.0:
            raise ValueError("degraded_multiplier must be in [0, 1]")
        for name in ("cooldown_s", "fish_absence_s", "source_timeout_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def assessed_h(self, channel: str) -> float:
        """Assessed per-sample min-entropy of a camera channel."""
        if channel == CHANNEL_SENSOR:
            return self.sensor_h
        if channel == CHANNEL_FISH:
            return self.fish_h
        raise KeyError(f"no assessed entropy for channel {channel!r}")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Verdict of one domain check. ``reason`` is a short dashboard string."""

    name: str
    state: HealthState
    reason: str = ""


class FrozenFrameCheck:
    """``frozen_frame``: the video stopped changing, or is replaying itself.

    Every real sensor has shot noise, so no two raw frames are ever
    bit-identical, and consecutive frames never differ by almost nothing. A
    frame whose 128-bit digest (shape included; cheap and exact) matches any
    of the last ``history`` frames, or whose mean absolute difference from
    the previous frame is below ``mad_threshold`` grey levels (measured on a
    2x2 subsampled frame, plenty of pixels for this threshold), counts as
    frozen; ``frames`` consecutive frozen frames fail the camera channels.
    Remembering more than the previous frame is what catches a looping
    video (``video:tank.mp4,loop=1``, a replayed clip, an alternating pair
    of stills): each frame differs from its predecessor, but every frame of
    the second pass repeats one already seen. Loops longer than ``history``
    frames are left to the liveness check. A single repeated frame is
    tolerated because some drivers and video files legitimately duplicate
    frames.
    """

    def __init__(self, mad_threshold: float = 0.05, frames: int = 15, history: int = 2048) -> None:
        if history < 1:
            raise ValueError("history must be at least 1 frame")
        self.mad_threshold = mad_threshold
        self.frames = frames
        self.consecutive = 0
        self._recent: deque[bytes] = deque()
        self._history = history
        self._seen: dict[bytes, int] = {}  # digest -> occurrences in _recent
        self._prev_small: npt.NDArray[np.uint8] | None = None

    def observe(self, frame: Frame) -> CheckResult:
        image = np.ascontiguousarray(frame.image)
        h = hashlib.blake2b(digest_size=16, person=b"puffer-frozen")
        h.update(repr(image.shape).encode("ascii"))
        h.update(image.data)
        digest = h.digest()
        small = image[::2, ::2]
        prev = self._prev_small
        if digest in self._seen:
            frozen = True
        elif prev is None or prev.shape != small.shape:
            frozen = False
        else:
            diff = np.abs(small.astype(np.int16) - prev.astype(np.int16))
            frozen = float(diff.mean()) < self.mad_threshold
        self.consecutive = self.consecutive + 1 if frozen else 0
        self._remember(digest)
        self._prev_small = small.copy()
        if self.consecutive >= self.frames:
            return CheckResult(
                CHECK_FROZEN, HealthState.FAILED, f"frozen video: {self.consecutive} frames"
            )
        return CheckResult(CHECK_FROZEN, HealthState.OK)

    def _remember(self, digest: bytes) -> None:
        self._recent.append(digest)
        self._seen[digest] = self._seen.get(digest, 0) + 1
        if len(self._recent) > self._history:
            old = self._recent.popleft()
            if self._seen[old] == 1:
                del self._seen[old]
            else:
                self._seen[old] -= 1


class FishPresenceCheck:
    """``fish_presence``: are there fish to watch?

    With fewer than ``min_fish`` confirmed tracks the fish channel produces
    almost no samples, and what it produces (tracker noise on an empty tank)
    is not fish behaviour. A short absence (a fish behind a plant) only
    degrades the channel; ``absence_s`` seconds of absence fail it. With an
    ``expected_fish`` count configured, a partial population degrades it.
    """

    def __init__(
        self, min_fish: int = 1, expected_fish: int | None = None, absence_s: float = 10.0
    ) -> None:
        self.min_fish = min_fish
        self.expected_fish = expected_fish
        self.absence_s = absence_s
        self._since: float | None = None  # last time enough fish were seen (or first look)

    def observe(self, result: TrackingResult) -> CheckResult:
        n = sum(1 for track in result.tracks if track.confirmed)
        if self._since is None or n >= self.min_fish:
            self._since = result.t
        if n < self.min_fish:
            absent = result.t - self._since
            if absent >= self.absence_s:
                return CheckResult(
                    CHECK_FISH_PRESENCE, HealthState.FAILED, f"no fish for {absent:.0f} s"
                )
            return CheckResult(
                CHECK_FISH_PRESENCE,
                HealthState.DEGRADED,
                f"{n} fish visible (need {self.min_fish})",
            )
        if self.expected_fish is not None and n < self.expected_fish:
            return CheckResult(
                CHECK_FISH_PRESENCE,
                HealthState.DEGRADED,
                f"{n} of {self.expected_fish} fish visible",
            )
        return CheckResult(CHECK_FISH_PRESENCE, HealthState.OK)


class LowEntropyCheck:
    """``low_entropy``: the rolling MCV estimate fell below a floor.

    The continuous tests are tuned to the *assessed* entropy and only catch
    gross failures; this check catches a slow drift (dim lighting, a
    saturated sensor region growing) long before RCT/APT would.
    """

    def __init__(self, min_bits_per_sample: float = 0.5) -> None:
        self.min_bits_per_sample = min_bits_per_sample

    def observe(self, h: float) -> CheckResult:
        if not (math.isfinite(h) and h >= self.min_bits_per_sample):
            return CheckResult(
                CHECK_LOW_ENTROPY,
                HealthState.FAILED,
                f"low entropy: {h:.2f} bits/sample (min {self.min_bits_per_sample:g})",
            )
        return CheckResult(CHECK_LOW_ENTROPY, HealthState.OK)


class SourceTimeoutCheck:
    """``source_timeout``: no frame arrived for ``timeout_s`` seconds.

    Without this check a dead camera would leave the last verdicts (and the
    last credit multiplier) frozen in place. The clock starts at the first
    frame or, if none ever arrives, at the first :meth:`check`.
    """

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self._last: float | None = None

    def frame_seen(self, t: float) -> None:
        self._last = t

    def check(self, now: float) -> CheckResult:
        if self._last is None:
            self._last = now
        silent = now - self._last
        if silent >= self.timeout_s:
            return CheckResult(
                CHECK_SOURCE_TIMEOUT, HealthState.FAILED, f"no frame for {silent:.1f} s"
            )
        return CheckResult(CHECK_SOURCE_TIMEOUT, HealthState.OK)


class LivenessCheck:
    """``liveness``: gate camera credit on the LED anti-replay challenge.

    When liveness is required, anything but VERIFIED (including CALIBRATING)
    fails the camera channels: until the fresh LED pattern is seen in the
    video, the frames may be a recording, and a recording's "entropy" is
    known to whoever recorded it.
    """

    def observe(self, status: LivenessStatus, required: bool) -> CheckResult:
        if required and status.state is not LivenessState.VERIFIED:
            return CheckResult(CHECK_LIVENESS, HealthState.FAILED, f"liveness {status.state}")
        return CheckResult(CHECK_LIVENESS, HealthState.OK)


# --------------------------------------------------------------------------
# Per-channel state machine
# --------------------------------------------------------------------------


class ChannelHealth:
    """Health state of one camera channel: continuous tests + domain verdicts.

    Not thread-safe on its own; :class:`HealthMonitor` serialises access.
    """

    def __init__(self, name: str, h_assessed: float, config: HealthConfig | None = None) -> None:
        self.name = name
        self.config = config or HealthConfig()
        self.rct = RepetitionCountTest(h_assessed, self.config.alpha_exp)
        self.apt = AdaptiveProportionTest(h_assessed, self.config.apt_window, self.config.alpha_exp)
        self._checks: dict[str, CheckResult] = {}
        self._startup_seen = 0
        self._started = False
        self._failed = False
        self._failed_until = -math.inf
        self._test_failures: set[str] = set()
        self._state = HealthState.STARTUP

    @property
    def state(self) -> HealthState:
        """State as of the last observation (call :meth:`report` to advance time)."""
        return self._state

    @property
    def credit_multiplier(self) -> float:
        """Factor applied to this channel's credit: OK 1.0, DEGRADED 0.5, else 0."""
        if self._state is HealthState.OK:
            return 1.0
        if self._state is HealthState.DEGRADED:
            return self.config.degraded_multiplier
        return 0.0

    def update(self, samples: npt.ArrayLike, t: float) -> int:
        """Run RCT and APT on a batch; return the number of failures in it.

        The STARTUP -> OK transition is taken at the *start* of a batch, once
        previous batches completed the start-up window, so the multiplier read
        after the batch that finished start-up is still 0. Samples keep
        flowing through the tests while the channel is FAILED, so a source
        that is still broken keeps restarting its cooldown.
        """
        x = _symbols(samples)
        self._advance(t)
        if not (self._failed or self._started) and (
            self._startup_seen >= self.config.startup_samples
        ):
            self._started = True
        rct_failures = self.rct.update(x)
        apt_failures = self.apt.update(x)
        if rct_failures:
            self._test_failures.add(CHECK_RCT)
        if apt_failures:
            self._test_failures.add(CHECK_APT)
        if rct_failures or apt_failures:
            self._fail(t)
        elif not (self._failed or self._started):
            self._startup_seen += int(x.size)
        self._state = self._evaluate()
        return rct_failures + apt_failures

    def apply(self, result: CheckResult, t: float) -> None:
        """Record a domain-check verdict observed at time ``t``.

        A FAILED verdict starts (or restarts) the cooldown; the cooldown also
        restarts when a failing check clears, so it always measures time
        since the source was last seen misbehaving.
        """
        previous = self._checks.get(result.name)
        self._checks[result.name] = result
        if result.state is HealthState.FAILED:
            self._fail(t)
        elif previous is not None and previous.state is HealthState.FAILED:
            self._failed_until = max(self._failed_until, t + self.config.cooldown_s)
        self._advance(t)
        self._state = self._evaluate()

    def report(self, t: float) -> HealthReport:
        """Advance the state machine to time ``t`` and describe it."""
        self._advance(t)
        self._state = self._evaluate()
        reasons: list[str] = []
        if self._failed:
            reasons.extend(
                f"{name.replace('_', ' ')} test failed" for name in self._failure_order()
            )
            # A still-failing check explains itself below; otherwise show the cooldown.
            if t < self._failed_until and not self._any_check(HealthState.FAILED):
                reasons.append(f"cooldown: {self._failed_until - t:.1f} s left")
        elif not self._started:
            reasons.append(
                f"start-up: {min(self._startup_seen, self.config.startup_samples)}"
                f"/{self.config.startup_samples} samples"
            )
        reasons.extend(c.reason for c in self._checks.values() if c.state is not HealthState.OK)
        checks = {name: self._test_state(name).value for name in (CHECK_RCT, CHECK_APT)}
        checks.update({name: c.state.value for name, c in self._checks.items()})
        return HealthReport(self.name, self._state, tuple(reasons), checks)

    # -- internals ---------------------------------------------------------

    def _fail(self, t: float) -> None:
        self._failed = True
        self._failed_until = max(self._failed_until, t + self.config.cooldown_s)

    def _advance(self, t: float) -> None:
        """Leave FAILED for a fresh start-up once the cooldown is over."""
        if not self._failed or t < self._failed_until:
            return
        if self._any_check(HealthState.FAILED):
            return
        self._failed = False
        self._started = False
        self._startup_seen = 0
        self._test_failures.clear()
        self.rct.reset()
        self.apt.reset()

    def _evaluate(self) -> HealthState:
        if self._failed:
            return HealthState.FAILED
        if not self._started:
            return HealthState.STARTUP
        if self._any_check(HealthState.DEGRADED):
            return HealthState.DEGRADED
        return HealthState.OK

    def _any_check(self, state: HealthState) -> bool:
        return any(c.state is state for c in self._checks.values())

    def _failure_order(self) -> list[str]:
        return [name for name in (CHECK_RCT, CHECK_APT) if name in self._test_failures]

    def _test_state(self, name: str) -> HealthState:
        if name in self._test_failures:
            return HealthState.FAILED
        if not (self._failed or self._started):
            return HealthState.STARTUP
        return HealthState.OK


# --------------------------------------------------------------------------
# Monitor: routes observations to channels
# --------------------------------------------------------------------------


class HealthMonitor:
    """Health of every channel, fed by the engine, read by the API.

    Thread-safe: the engine thread observes while API/WebSocket threads call
    :meth:`reports`. The ``system`` channel (``os.urandom``) is always OK -
    if the kernel RNG failed, ``os.urandom`` would raise rather than return
    weak bytes - and is reported so the dashboard can show all three.
    """

    def __init__(self, config: HealthConfig | None = None) -> None:
        self.config = config or HealthConfig()
        cfg = self.config
        self._channels = {
            name: ChannelHealth(name, cfg.assessed_h(name), cfg) for name in CAMERA_CHANNELS
        }
        self._frozen = FrozenFrameCheck(
            cfg.frozen_mad_threshold, cfg.frozen_frames, cfg.frozen_history
        )
        self._presence = FishPresenceCheck(cfg.min_fish, cfg.expected_fish, cfg.fish_absence_s)
        self._low_entropy = {
            name: LowEntropyCheck(cfg.min_bits_per_sample) for name in CAMERA_CHANNELS
        }
        self._timeout = SourceTimeoutCheck(cfg.source_timeout_s)
        self._liveness = LivenessCheck()
        self._now: float | None = None
        self._lock = threading.Lock()

    def observe_frame(self, frame: Frame) -> None:
        """Frozen-frame and source-timeout checks (both camera channels)."""
        t = frame.timestamp
        with self._lock:
            self._touch(t)
            self._timeout.frame_seen(t)
            self._apply_camera(self._frozen.observe(frame), t)
            self._apply_camera(self._timeout.check(t), t)

    def observe_tracking(self, result: TrackingResult) -> None:
        """Fish-presence check (fish channel)."""
        with self._lock:
            self._touch(result.t)
            self._channels[CHANNEL_FISH].apply(self._presence.observe(result), result.t)

    def observe_samples(self, channel: str, samples: npt.ArrayLike, t: float) -> int:
        """Continuous tests on a channel's sample batch; returns its failure count."""
        with self._lock:
            self._touch(t)
            return self._channel(channel).update(samples, t)

    def observe_entropy_estimate(self, channel: str, h: float, t: float) -> None:
        """Low-entropy check on a fresh rolling estimate (bits per sample)."""
        with self._lock:
            self._touch(t)
            health = self._channel(channel)
            health.apply(self._low_entropy[channel].observe(h), t)

    def observe_liveness(
        self, status: LivenessStatus, required: bool, t: float | None = None
    ) -> None:
        """Liveness gate (both camera channels).

        ``t`` defaults to the latest time seen by any other observation;
        ``LivenessStatus`` carries no reliable timestamp of its own.
        """
        with self._lock:
            when = t if t is not None else (self._now if self._now is not None else 0.0)
            self._touch(when)
            self._apply_camera(self._liveness.observe(status, required), when)

    def check_timeout(self, now: float) -> None:
        """Source-timeout check; call periodically, also when no frames arrive."""
        with self._lock:
            self._touch(now)
            self._apply_camera(self._timeout.check(now), now)

    def reports(self, now: float | None = None) -> dict[str, HealthReport]:
        """Reports for sensor, fish and system, as of ``now`` (default: latest time seen)."""
        with self._lock:
            t = now if now is not None else (self._now if self._now is not None else 0.0)
            out = {name: health.report(t) for name, health in self._channels.items()}
        out[CHANNEL_SYSTEM] = HealthReport(CHANNEL_SYSTEM, HealthState.OK)
        return out

    def state(self, channel: str) -> HealthState:
        """Current state of one channel."""
        if channel == CHANNEL_SYSTEM:
            return HealthState.OK
        with self._lock:
            return self._channel(channel).state

    def usable(self, channel: str) -> bool:
        """May the channel's output be absorbed? (Everything but FAILED.)

        STARTUP output is absorbed without credit: hashing more data into the
        pool can never hurt, only crediting it can.
        """
        return self.state(channel) is not HealthState.FAILED

    def credit_multiplier(self, channel: str) -> float:
        if channel == CHANNEL_SYSTEM:
            return 1.0
        with self._lock:
            return self._channel(channel).credit_multiplier

    def overall(self) -> HealthState:
        """One headline state for the camera entropy.

        FAILED only if *every* camera channel failed (no camera entropy at
        all); DEGRADED if some channel failed or is degraded; STARTUP while
        any channel is still starting; OK otherwise. System randomness keeps
        outputs secure in every case, so this is about camera entropy only.
        """
        with self._lock:
            states = [health.state for health in self._channels.values()]
        if all(s is HealthState.FAILED for s in states):
            return HealthState.FAILED
        if any(s in (HealthState.FAILED, HealthState.DEGRADED) for s in states):
            return HealthState.DEGRADED
        if any(s is HealthState.STARTUP for s in states):
            return HealthState.STARTUP
        return HealthState.OK

    # -- internals ---------------------------------------------------------

    def _channel(self, channel: str) -> ChannelHealth:
        try:
            return self._channels[channel]
        except KeyError:
            raise KeyError(f"unknown camera channel {channel!r}") from None

    def _apply_camera(self, result: CheckResult, t: float) -> None:
        for health in self._channels.values():
            health.apply(result, t)

    def _touch(self, t: float) -> None:
        self._now = t if self._now is None else max(self._now, t)
