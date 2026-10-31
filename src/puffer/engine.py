"""The entropy engine: frames in, credited entropy, receipts and live statistics out.

:class:`EntropyEngine` is where every subsystem meets (DESIGN.md sections 1 and
9.2). It owns one frame source and drives, per frame::

    source.read()
      -> health.observe_frame                 frozen video, source timeout
      -> liveness.process                     LED challenge verdict for *this* frame
      -> health.observe_liveness              gate (required) or display only (advisory)
      -> tracker.process                      fish tracks, identities, innovations
      -> SensorChannel / FishChannel          ChannelOutput (absorb + samples)
      -> health.observe_samples + RollingEstimate  RCT/APT, low-entropy check
      -> pool.absorb(output, credit)          only while the channel is not FAILED
      -> fish statistics, snapshot, subscribers

and hands out randomness only through :attr:`EntropyEngine.rng` (PufferRNG keyed
by ``os.urandom``), so nothing the engine does - or fails to do - can make an
output weaker than the operating system's generator (DESIGN.md rule 2).

Design decisions
----------------
* **Credit is computed here, from samples only.** ``credit = n * min(H_est,
  H_cap) * safety_factor * health_multiplier`` (DESIGN.md 4.3); the rolling
  estimate is refreshed from the very samples that were just health-tested.
  A FAILED channel's output is neither absorbed nor credited; STARTUP output
  is absorbed with zero credit (hashing more data never hurts).
* **The LED is not entropy.** The LED region (configured, the simulator's
  exact ROI, or whatever the verifier calibrates), padded by
  :data:`LED_ROI_PADDING` pixels, is excluded from the sensor sample sites:
  its brightness is our own public challenge. When the verifier learns or
  re-learns the ROI the sensor channel is rebuilt with fresh sites.
  The same region (wider padding) is masked out of the image the fish
  tracker sees: a blinking LED is a perfect motion blob and would otherwise
  be tracked, named and credited as a fish.
* **Fish entropy is measured where it was observed.** Frames wider than
  ``vision_max_width`` are downscaled for tracking. The fish channel samples
  the innovations in the *tracker's* pixel grid: scaling them up first would
  invent sub-pixel resolution the measurement never had. Boxes, trails,
  speeds and distances shown to people are scaled back to frame pixels.
* **Rates live on the frame clock.** ``bits_per_s`` and ``fps`` are
  exponential moving averages over frame timestamps, so a recorded video
  processed at 3x real time reports the rates the camera produced;
  ``processing_fps`` is the wall-clock throughput of the engine itself.
  ``bits_per_s`` and ``credited_total`` count the credit a channel *earned*;
  the pool's ledger saturates at ``capacity_bits`` until someone draws, and
  how much a particular output consumed is on its receipt.
* **A frame that fails does not stop the tank.** The background thread logs
  and counts exceptions and carries on with the next frame (backing off);
  only ``max_consecutive_errors`` failures in a row stop it, with the error
  in :meth:`EntropyEngine.snapshot`. Subscriber and listener exceptions are
  logged and never reach the loop.
* **Health keeps ticking when the camera does not.** While a live source is
  stuck inside ``read()`` (or keeps returning nothing), a watchdog calls
  ``health.check_timeout`` with the frame clock extrapolated by wall time, so
  ``source_timeout`` fails the camera channels after 5 s instead of the last
  verdict staying frozen on the dashboard.
* **Thread model.** ``step()`` and everything it touches (tracker, verifier,
  channels, re-ID gallery) run on one thread at a time (the engine thread, or
  the caller of :meth:`EntropyEngine.step` / :meth:`EntropyEngine.collect`).
  Readers (:meth:`snapshot`, :meth:`fish_cards`, :meth:`jpeg`,
  :meth:`receipt_context`, ``rng.generate``) only see immutable values
  published at the end of each frame, or thread-safe components (pool,
  health monitor, fish statistics), so they never block the frame loop for
  long and never observe a half-updated frame.
"""

from __future__ import annotations

import enum
import logging
import math
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal, Self

import cv2
import numpy as np
import numpy.typing as npt

from puffer.config import PufferConfig
from puffer.entropy import (
    DEFAULT_H_CAP,
    EntropyPool,
    FishChannel,
    HealthConfig,
    HealthMonitor,
    PufferRNG,
    ReceiptContext,
    RollingEstimate,
    SensorChannel,
    SensorChannelConfig,
    compute_credit,
)
from puffer.fishstats import FishStats, FishStatsError, FishTotals
from puffer.hardware import LedController
from puffer.liveness import LivenessVerifier
from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SYSTEM,
    ChannelOutput,
    Frame,
    FrameSource,
    HealthReport,
    HealthState,
    LightMode,
    LivenessState,
    LivenessStatus,
    TrackingResult,
    TrackSnapshot,
)
from puffer.vision import FishCharacter, FishTracker

__all__ = [
    "CHAOS_HALF_SCORE",
    "LED_ROI_PADDING",
    "SYSTEM_BITS_PER_DRAW",
    "VISION_LED_PADDING",
    "ChannelStats",
    "EngineSnapshot",
    "EngineState",
    "EntropyEngine",
    "FishLabel",
    "OutputListener",
    "ResolvedLiveness",
    "chaos_fraction",
]

logger = logging.getLogger(__name__)

ResolvedLiveness = Literal["required", "advisory", "off"]
"""Liveness policy after ``auto`` was resolved for the source (see :mod:`puffer.runtime`)."""

Roi = tuple[int, int, int, int]
OutputListener = Callable[[ChannelOutput, float, HealthState], None]
SnapshotCallback = Callable[["EngineSnapshot"], None]

LED_ROI_PADDING: Final = 4
"""Pixels added around the LED region before excluding it from sensor sampling (halo, blur)."""

VISION_LED_PADDING: Final = 8
"""Pixels (tracker resolution) added around the LED region masked from the fish tracker.

Wider than :data:`LED_ROI_PADDING` because the motion detector blurs and
dilates: an edge of the blinking halo left visible still becomes a blob."""

SYSTEM_BITS_PER_DRAW: Final = 256.0
"""Every pool draw is keyed with 32 fresh bytes of ``os.urandom``."""

CHAOS_HALF_SCORE: Final = 1.0
"""Vision chaos score that maps to a 0.5 chaos fraction (see :func:`chaos_fraction`)."""

RATE_TAU_S: Final = 5.0
"""Time constant (frame-clock seconds) of the ``bits_per_s`` moving averages."""

FPS_TAU_S: Final = 2.0
HISTORY_PERIOD_S: Final = 0.5
"""Spacing of :meth:`EntropyEngine.entropy_history` samples (the 2 Hz entropy message)."""

HISTORY_MAX_S: Final = 600.0

_STALL_CHECK_S: Final = 1.0  # a read() blocked this long counts as a stalled source
_WATCHDOG_PERIOD_S: Final = 0.5
_LIVE_RETRY_S: Final = 0.2  # wait between reads after a live source returned nothing
_FISH_ID_NUMBER = re.compile(r"^f(\d+)$")
_DISABLED = LivenessStatus(LivenessState.DISABLED)


class EngineState(enum.StrEnum):
    """Lifecycle of the engine (``snapshot().run_state``)."""

    IDLE = "idle"  # constructed; stepped synchronously if at all
    RUNNING = "running"  # background thread active
    STOPPED = "stopped"  # thread stopped by stop()
    ENDED = "ended"  # a finite source (video file) is exhausted
    ERROR = "error"  # stopped after repeated frame failures (see last_error)
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ChannelStats:
    """Live statistics of one channel (DESIGN.md 9.2 and the ``entropy`` message).

    ``bits_per_s`` is credited bits per second of frame time (EMA, time
    constant :data:`RATE_TAU_S`), ``h_per_sample`` the rolling min-entropy
    estimate (``None`` for ``system``, which has no samples), and
    ``credited_total`` the credit earned since start. For ``system`` both
    count 256 bits per pool draw.
    """

    name: str
    bits_per_s: float
    h_per_sample: float | None
    credited_total: float
    health: HealthState


@dataclass(frozen=True, slots=True)
class FishLabel:
    """Display name and colour of a fish identity (for track overlays)."""

    name: str
    color: str


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Immutable state after one frame (DESIGN.md 9.2), safe to share across threads.

    Fields after ``liveness`` extend the contract: ``source_id``,
    ``processing_fps`` (wall-clock throughput), ``pool_capacity_bits``,
    ``fish`` (labels for every known fish id), ``liveness_policy``, and the
    engine's ``run_state`` / ``errors`` / ``last_error``. ``tracks`` are in
    frame pixels. ``frame_index`` is -1 in a snapshot published before any
    frame was processed (an engine that failed at start-up).
    """

    frame_index: int
    t: float
    wall_time: float
    fps: float
    mode: LightMode
    width: int
    height: int
    tracks: tuple[TrackSnapshot, ...]
    channels: dict[str, ChannelStats]
    pool: dict[str, float]
    health: dict[str, HealthReport]
    overall: HealthState
    liveness: LivenessStatus
    source_id: str = "unknown"
    processing_fps: float = 0.0
    pool_capacity_bits: float = 4096.0
    fish: dict[str, FishLabel] = field(default_factory=dict)
    liveness_policy: ResolvedLiveness = "off"
    run_state: EngineState = EngineState.IDLE
    errors: int = 0
    last_error: str | None = None


def chaos_fraction(score: float) -> float:
    """Map the vision chaos score (dimensionless, ``0..inf``) to a ``0..1`` fraction.

    ``score / (score + CHAOS_HALF_SCORE)``: monotone (rankings are kept), 0 for
    a perfectly predictable swimmer, 0.5 at a score of 1 (the vision module's
    "calm cruiser well below 1, erratic darter several"), approaching 1 for
    ever more chaotic fish. A score that is already a fraction stays a
    fraction. Non-finite or negative input maps to 0.
    """
    if not (math.isfinite(score) and score > 0):
        return 0.0
    return score / (score + CHAOS_HALF_SCORE)


class _RateEma:
    """Exponential moving average of a rate on an arbitrary clock.

    Amounts that arrive while the clock does not advance are carried into the
    next interval, and ``alpha = 1 - exp(-dt / tau)`` makes the average
    independent of the frame rate.
    """

    __slots__ = ("_carry", "_last", "tau", "value")

    def __init__(self, tau: float) -> None:
        self.tau = tau
        self.value = 0.0
        self._last: float | None = None
        self._carry = 0.0

    def update(self, amount: float, t: float) -> float:
        if self._last is None:
            self._last = t
            return self.value
        dt = t - self._last
        if not dt > 0:
            self._carry += amount
            return self.value
        rate = (self._carry + amount) / dt
        self.value += (1.0 - math.exp(-dt / self.tau)) * (rate - self.value)
        self._last = t
        self._carry = 0.0
        return self.value


@dataclass(frozen=True, slots=True)
class _FishView:
    """What fish cards need about one fish, captured on the engine thread."""

    fish_id: str
    name: str
    color: str
    present: bool
    track_id: int | None
    speed_avg: float
    speed_max: float
    activity: float
    chaos_score: float
    distance_px: float
    seen_s: float
    bits_session: float


class EntropyEngine:
    """Turn a frame source into credited entropy, receipts and live statistics.

    Parameters
    ----------
    config:
        Runtime settings (sensor sites, safety factor, pool capacity, H values,
        ``min_fish``, ``vision_max_width``, ``fish_names``).
    source:
        The frame source; the engine owns it and closes it in :meth:`close`.
    tracker:
        Fish tracker (default :class:`~puffer.vision.FishTracker` with the
        motion detector).
    liveness, liveness_policy:
        The LED verifier and the resolved policy: ``required`` gates camera
        credit on VERIFIED, ``advisory`` only reports, ``off`` needs
        ``liveness=None`` and reports DISABLED.
    led:
        LED controller for a physical LED (started on the first frame,
        switched off and its driver closed in :meth:`close`).
    led_roi:
        LED region known up front (config or simulator), excluded from
        sensor sampling until the verifier reports one.
    stats, gallery_path:
        Persistent fish statistics and re-ID gallery file (both optional;
        the engine flushes/saves them periodically and on :meth:`close`).
    live_source:
        True for cameras: a ``None`` from ``read()`` means "no frame right now"
        (retry, let ``source_timeout`` fail the channels), not "end of file".
    health_config:
        Overrides the health thresholds derived from ``config``.
    sensor_site_seed:
        Reproducible sensor sites for tests; ``None`` (production) draws them
        from ``os.urandom``.
    gallery_save_s:
        Frame-clock interval between gallery saves.
    max_consecutive_errors:
        Frame failures in a row after which the background thread gives up.
    clock:
        Monotonic wall clock (watchdog, timeouts, processing rate).
    """

    def __init__(
        self,
        config: PufferConfig,
        source: FrameSource,
        *,
        tracker: FishTracker | None = None,
        liveness: LivenessVerifier | None = None,
        liveness_policy: ResolvedLiveness = "off",
        led: LedController | None = None,
        led_roi: Roi | None = None,
        stats: FishStats | None = None,
        gallery_path: Path | None = None,
        live_source: bool = False,
        health_config: HealthConfig | None = None,
        sensor_site_seed: bytes | None = None,
        gallery_save_s: float = 60.0,
        max_consecutive_errors: int = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if liveness_policy not in ("required", "advisory", "off"):
            raise ValueError(
                f"liveness_policy must be required, advisory or off: {liveness_policy}"
            )
        if liveness_policy == "off" and liveness is not None:
            raise ValueError("liveness_policy 'off' takes no verifier")
        if liveness_policy != "off" and liveness is None:
            raise ValueError(f"liveness_policy {liveness_policy!r} needs a LivenessVerifier")
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")
        if not gallery_save_s > 0:
            raise ValueError("gallery_save_s must be positive")
        self.config = config
        self.source = source
        self.tracker = tracker if tracker is not None else FishTracker()
        self.liveness_verifier = liveness
        self.liveness_policy: ResolvedLiveness = liveness_policy
        self.led = led
        self.stats = stats
        self.gallery_path = gallery_path
        self.live_source = live_source
        self.max_consecutive_errors = max_consecutive_errors
        self._clock = clock
        self._site_seed = sensor_site_seed
        self._gallery_save_s = float(gallery_save_s)

        self.health = HealthMonitor(
            health_config
            or HealthConfig(
                sensor_h=config.sensor_h_assessed,
                fish_h=config.fish_h_assessed,
                min_fish=config.min_fish,
            )
        )
        self.pool = EntropyPool(config.pool_capacity_bits)
        self.rng = PufferRNG(self.pool, context=self.receipt_context, pump=self._pump)
        """The only way to obtain secret bytes (thread-safe; keyed by ``os.urandom``)."""

        self._excluded_roi: Roi | None = None
        self._sensor = self._make_sensor(led_roi)
        self._fish_channel = FishChannel()
        self._estimates = {
            name: RollingEstimate(h_cap=DEFAULT_H_CAP[name]) for name in CAMERA_CHANNELS
        }
        self._totals = dict.fromkeys((*CAMERA_CHANNELS, CHANNEL_SYSTEM), 0.0)
        self._rates = {name: _RateEma(RATE_TAU_S) for name in self._totals}
        self._fps = _RateEma(FPS_TAU_S)
        self._proc_fps = _RateEma(FPS_TAU_S)
        self._draws_seen = self.pool.draw_index

        # Engine-thread state (only touched inside _process).
        self._fish_prev: dict[str, tuple[float, float]] = {}  # fish_id -> (seen_s, distance_px)
        self._fish_bits: dict[str, float] = {}
        self._known_fish: frozenset[str] = frozenset()
        self._next_gallery_save: float | None = None
        self._next_history: float | None = None
        self._vision_scale = 1.0

        # Published state (immutable values, swapped atomically).
        self._snapshot: EngineSnapshot | None = None
        self._frame: Frame | None = None
        self._liveness_status = liveness.status if liveness is not None else _DISABLED
        self._fish_view: tuple[_FishView, ...] = ()
        self._labels: dict[str, FishLabel] = {}
        self._names: dict[str, str] = dict(config.fish_names)
        self._last_frame_t: float | None = None
        self._last_frame_mono: float | None = None

        self._history: deque[dict[str, Any]] = deque(
            maxlen=int(HISTORY_MAX_S / HISTORY_PERIOD_S) + 1
        )
        self._history_lock = threading.Lock()
        self._jpeg_cache: tuple[tuple[int, int, int], bytes] | None = None
        self._jpeg_lock = threading.Lock()

        self._subscribers: list[SnapshotCallback] = []
        self._listeners: list[OutputListener] = []
        self._callback_lock = threading.Lock()
        self._callback_failures: Counter[int] = Counter()
        self._component_failures: Counter[str] = Counter()

        self._step_lock = threading.RLock()  # one stepper at a time (held across read())
        self._process_lock = threading.Lock()  # frame processing vs. the watchdog
        self._thread_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._read_started: float | None = None
        self._led_started = False
        self._ended = False
        self._closed = False
        self._run_state = EngineState.IDLE
        self._errors = 0
        self._last_error: str | None = None

        self._load_gallery()

    # ================================================================ stepping

    def step(self) -> EngineSnapshot | None:
        """Read and process one frame; ``None`` when the source has no frame.

        For a finite source ``None`` means it is exhausted; for a live source
        it means the read failed (the ``source_timeout`` check is advanced).
        Exceptions propagate to the caller.

        Raises:
            RuntimeError: the background thread is running, or the engine is closed.
        """
        thread = self._thread
        if thread is not None and thread.is_alive():
            raise RuntimeError("the engine thread is running; use snapshot() instead of step()")
        return self._step()

    def collect(
        self, min_credit_bits: float, *, timeout_s: float, max_frames: int | None = None
    ) -> bool:
        """Process frames until the camera channels hold ``min_credit_bits`` of credit.

        Synchronous (for the CLI): steps the source on the calling thread, or
        simply waits when the background thread is running. Returns ``False``
        when ``timeout_s`` (wall clock) or ``max_frames`` is reached, or a
        finite source ends, first. A single failing frame is skipped; the
        error of ``max_consecutive_errors`` failures in a row is raised.

        Raises:
            ValueError: the target can never be reached (above the pool's
                capacity for both camera channels) or the limits are invalid.
        """
        reachable = self.pool.capacity_bits * len(CAMERA_CHANNELS)
        if not (math.isfinite(min_credit_bits) and 0 <= min_credit_bits <= reachable):
            raise ValueError(
                f"min_credit_bits must be in [0, {reachable:g}] (both camera channels "
                f"saturate at {self.pool.capacity_bits:g} bits), got {min_credit_bits!r}"
            )
        if not (math.isfinite(timeout_s) and timeout_s >= 0):
            raise ValueError("timeout_s must be finite and non-negative")
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        if self._closed:
            raise RuntimeError("engine is closed")
        if self.running:
            return self.pool.wait_for_credit(min_credit_bits, timeout_s)
        deadline = self._clock() + timeout_s
        frames = 0
        consecutive = 0
        while self.pool.camera_credit() < min_credit_bits:
            if (max_frames is not None and frames >= max_frames) or self._clock() >= deadline:
                return False
            try:
                snap = self._step()
            except Exception as exc:
                consecutive += 1
                self._record_error(exc, consecutive)
                if consecutive >= self.max_consecutive_errors:
                    raise
                continue
            consecutive = 0
            if snap is None:
                if not self.live_source:
                    return False
                time.sleep(min(_LIVE_RETRY_S, max(0.0, deadline - self._clock())))
                continue
            frames += 1
        return True

    # ================================================================ thread

    def start(self) -> None:
        """Run the frame loop in a background thread (no-op if already running).

        Raises:
            RuntimeError: the engine is closed.
        """
        with self._thread_lock:
            if self._closed:
                raise RuntimeError("engine is closed")
            if self._thread is not None and self._thread.is_alive():
                if self._stop_event.is_set():
                    # stop() timed out (a camera stuck in read()): a second loop
                    # must not start next to the one that is still winding down.
                    raise RuntimeError("the previous engine thread has not stopped yet")
                return
            self._stop_event.clear()
            self._run_state = EngineState.RUNNING
            self._thread = threading.Thread(target=self._run, name="puffer-engine", daemon=True)
            self._thread.start()
            if self.live_source:
                self._watchdog = threading.Thread(
                    target=self._watch, name="puffer-engine-watchdog", daemon=True
                )
                self._watchdog.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread (idempotent); the engine can be started again.

        Waits up to ``timeout`` seconds for the current frame to finish. A
        thread stuck in a blocking ``read()`` is left to exit on its own (it
        will, once the read returns or :meth:`close` closes the source).
        """
        self._stop_event.set()
        with self._thread_lock:
            for thread in (self._thread, self._watchdog):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout)
                    if thread.is_alive():
                        logger.warning("%s did not stop within %.1f s", thread.name, timeout)
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._watchdog is not None and not self._watchdog.is_alive():
                self._watchdog = None
            if self._run_state is EngineState.RUNNING:
                self._run_state = EngineState.STOPPED

    @property
    def sensor_channel(self) -> SensorChannel:
        """The sensor channel in use (rebuilt when the LED region changes); diagnostics only."""
        return self._sensor

    @property
    def led_roi(self) -> Roi | None:
        """LED region currently excluded from sensor sampling and masked from the tracker."""
        return self._excluded_roi

    @property
    def running(self) -> bool:
        """Whether the background thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def run_state(self) -> EngineState:
        return self._run_state

    @property
    def errors(self) -> int:
        """Frames that raised since the engine was created."""
        return self._errors

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def close(self) -> None:
        """Stop the thread, switch the LED off, close the source, save gallery and stats.

        Idempotent; every step is attempted even if an earlier one fails.
        """
        if self._closed:
            return
        self.stop()
        # A camera stuck inside read() holds the step lock; do not wait for it
        # forever - closing the source below is what unblocks such a read.
        acquired = self._step_lock.acquire(timeout=5.0)
        try:
            if self._closed:
                return
            self._closed = True
            self._run_state = EngineState.CLOSED
        finally:
            if acquired:
                self._step_lock.release()
        steps: list[tuple[str, Callable[[], None]]] = [
            ("LED", self._close_led),
            ("source", self.source.close),
            ("fish gallery", self._save_gallery),
        ]
        if self.stats is not None:
            steps.append(("fish statistics", self.stats.close))
        for what, action in steps:
            try:
                action()
            except Exception:
                logger.exception("closing %s failed", what)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ================================================================ readers

    def snapshot(self) -> EngineSnapshot | None:
        """The state after the latest frame (``None`` before the first frame).

        Always carries the current ``run_state``/``errors``/``last_error``, so
        a thread that stopped on errors is visible even if no frame ever
        succeeded (then ``frame_index`` is -1).
        """
        snap = self._snapshot
        if snap is None:
            if self._errors == 0 and self._run_state is not EngineState.ERROR:
                return None
            snap = self._blank_snapshot()
        if (snap.run_state, snap.errors, snap.last_error) != (
            self._run_state,
            self._errors,
            self._last_error,
        ):
            snap = replace(
                snap, run_state=self._run_state, errors=self._errors, last_error=self._last_error
            )
        return snap

    def latest_frame(self) -> Frame | None:
        """The latest processed frame (treat its image as read-only)."""
        return self._frame

    def jpeg(self, *, max_width: int = 640, quality: int = 70) -> bytes | None:
        """JPEG of the latest frame, downscaled to ``max_width``; cached per frame.

        Encoding happens on the caller's thread (API), never in the frame loop.
        """
        if max_width < 16 or not 1 <= quality <= 100:
            raise ValueError("need max_width >= 16 and quality in 1..100")
        frame = self._frame
        if frame is None:
            return None
        key = (frame.index, int(max_width), int(quality))
        with self._jpeg_lock:
            cached = self._jpeg_cache
            if cached is not None and cached[0] == key:
                return cached[1]
            image: npt.NDArray[Any] = frame.image
            if frame.width > max_width:
                height = max(1, round(frame.height * max_width / frame.width))
                image = cv2.resize(image, (int(max_width), height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            data = encoded.tobytes()
            self._jpeg_cache = (key, data)
            return data

    def receipt_context(self) -> ReceiptContext:
        """Mode, liveness, health and fish names for receipts (callable from any thread).

        Health always lists both camera channels, so receipts and beacon
        summaries never carry a null health state.
        """
        snap = self._snapshot
        return ReceiptContext(
            mode=snap.mode.value if snap is not None else "unknown",
            liveness=self._liveness_status,
            health={name: self.health.state(name) for name in CAMERA_CHANNELS},
            fish_names=dict(self._names),
        )

    def fish_cards(self) -> list[dict[str, Any]]:
        """One DESIGN.md 7.1 ``FishCard`` per known fish, in fish-id order.

        ``stats`` units: ``speed_avg``/``speed_max`` in frame pixels per
        second, ``activity`` (fraction of tracked time spent moving) and
        ``chaos_score`` (:func:`chaos_fraction` of the vision score) in
        ``0..1`` - all four for the current session; ``bits_total``,
        ``bits_today``, ``distance_px`` and ``seen_s`` are persisted totals
        from :class:`~puffer.fishstats.FishStats` (session values without it).
        """
        totals: dict[str, FishTotals] = {}
        if self.stats is not None and not self._closed:
            try:
                totals = self.stats.totals()
            except FishStatsError:  # closed concurrently (shutdown): fall back to the session
                totals = {}
        cards = []
        for fish in self._fish_view:
            total = totals.get(fish.fish_id)
            cards.append(
                {
                    "fish_id": fish.fish_id,
                    "name": fish.name,
                    "color": fish.color,
                    "present": fish.present,
                    "track_id": fish.track_id,
                    "stats": {
                        "speed_avg": round(fish.speed_avg, 2),
                        "speed_max": round(fish.speed_max, 2),
                        "activity": round(fish.activity, 4),
                        "chaos_score": round(fish.chaos_score, 4),
                        "bits_total": round(
                            total.bits_total if total is not None else fish.bits_session, 3
                        ),
                        "bits_today": round(
                            total.bits_today if total is not None else fish.bits_session, 3
                        ),
                        "distance_px": round(
                            total.distance_px if total is not None else fish.distance_px, 1
                        ),
                        "seen_s": round(total.seen_s if total is not None else fish.seen_s, 1),
                    },
                }
            )
        return cards

    def leaderboard(self, days: int = 7) -> list[dict[str, Any]]:
        """``[{rank, fish_id, name, chaos_score, bits}]`` (DESIGN.md 7.1), best first.

        Ranked by mean chaos fraction over the last ``days`` UTC days from the
        persisted statistics, with current display names; without statistics,
        over the current session.
        """
        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise ValueError("days must be a positive integer")
        if self.stats is not None and not self._closed:
            try:
                board = self.stats.leaderboard(days)
            except FishStatsError:  # closed concurrently (shutdown): fall back to the session
                pass
            else:
                names = self._names
                for entry in board:
                    entry["name"] = names.get(entry["fish_id"], entry["name"])
                return board
        ranked = sorted(self._fish_view, key=lambda f: (-f.chaos_score, -f.bits_session, f.fish_id))
        return [
            {
                "rank": rank,
                "fish_id": f.fish_id,
                "name": f.name,
                "chaos_score": round(f.chaos_score, 4),
                "bits": round(f.bits_session, 3),
            }
            for rank, f in enumerate(ranked, start=1)
        ]

    def entropy_history(self, seconds: float = 300.0) -> list[dict[str, Any]]:
        """``entropy`` message payloads (DESIGN.md 7.2) of the last ``seconds``, oldest first.

        Sampled every :data:`HISTORY_PERIOD_S` of frame time into a ring buffer
        of :data:`HISTORY_MAX_S` seconds (backs ``GET /api/entropy/history``).
        """
        if not (math.isfinite(seconds) and seconds >= 0):
            raise ValueError("seconds must be finite and non-negative")
        with self._history_lock:
            items = list(self._history)
        if not items:
            return []
        since = items[-1]["t"] - min(seconds, HISTORY_MAX_S)
        return [item for item in items if item["t"] >= since]

    # ================================================================ callbacks

    def subscribe(self, callback: SnapshotCallback) -> Callable[[], None]:
        """Call ``callback(snapshot)`` after every frame; returns an unsubscribe function.

        Callbacks run on the frame loop's thread and must be quick (hand the
        snapshot to an asyncio loop with ``call_soon_threadsafe``). Exceptions
        are logged and ignored.
        """
        return self._add_callback(self._subscribers, callback)

    def add_output_listener(self, listener: OutputListener) -> Callable[[], None]:
        """Call ``listener(output, credit_bits, health_state)`` for every channel output.

        For analysis tooling: every output is reported, including FAILED ones
        (with zero credit, not absorbed). Runs on the frame loop's thread;
        exceptions are logged and ignored. Returns an unsubscribe function.
        """
        return self._add_callback(self._listeners, listener)

    def _add_callback(self, registry: list[Any], callback: Any) -> Callable[[], None]:
        with self._callback_lock:
            registry.append(callback)

        def unsubscribe() -> None:
            with self._callback_lock:
                if callback in registry:
                    registry.remove(callback)

        return unsubscribe

    def _call(self, registry: list[Any], *args: Any) -> None:
        with self._callback_lock:
            callbacks = tuple(registry)
        for callback in callbacks:
            try:
                callback(*args)
            except Exception:
                key = id(callback)
                self._callback_failures[key] += 1
                n = self._callback_failures[key]
                if n == 1 or n % 1000 == 0:
                    logger.exception("engine callback %r failed (%d times)", callback, n)

    # ================================================================ internals

    def _step(self) -> EngineSnapshot | None:
        with self._step_lock:
            if self._closed:
                raise RuntimeError("engine is closed")
            self._start_led()
            self._read_started = self._clock()
            try:
                frame = self.source.read()
            finally:
                self._read_started = None
            if frame is None:
                if self.live_source:
                    self._check_stalled()
                else:
                    self._ended = True
                return None
            with self._process_lock:
                return self._process(frame)

    def _process(self, frame: Frame) -> EngineSnapshot:
        t = frame.timestamp
        mono = self._clock()
        health = self.health
        health.observe_frame(frame)

        verifier = self.liveness_verifier
        status = self._verify(verifier, frame) if verifier is not None else _DISABLED
        self._liveness_status = status
        health.observe_liveness(status, self.liveness_policy == "required", t)
        if verifier is not None and verifier.roi is not None and verifier.roi != self._excluded_roi:
            self._sensor = self._make_sensor(verifier.roi)

        tracking = self._track(frame)
        health.observe_tracking(tracking)

        sensor_out = self._sensor.process(frame)
        fish_out = self._fish_channel.process(tracking)
        credits = {out.channel: self._credit(out) for out in (sensor_out, fish_out)}

        draws = self.pool.draw_index
        system_bits = SYSTEM_BITS_PER_DRAW * max(0, draws - self._draws_seen)
        self._draws_seen = draws
        credits[CHANNEL_SYSTEM] = system_bits
        for name, bits in credits.items():
            self._totals[name] += bits
            self._rates[name].update(bits, t)
        fps = self._fps.update(1.0, t)
        proc_fps = self._proc_fps.update(1.0, mono)

        self._update_fish(frame, tracking, fish_out, credits[CHANNEL_FISH])
        self._last_frame_t, self._last_frame_mono = t, mono
        self._frame = frame
        snap = self._publish(frame, tracking, fps, proc_fps)
        self._call(self._subscribers, snap)
        return snap

    def _verify(self, verifier: LivenessVerifier, frame: Frame) -> LivenessStatus:
        """Liveness verdict for ``frame``; a verifier bug fails closed instead of the frame.

        The verifier leaves FAILED behind when it raises, so under ``required``
        the camera channels lose credit while the sensor and fish keep being
        absorbed and health-tested.
        """
        try:
            return verifier.process(frame)
        except Exception as exc:
            self._component_error("liveness", exc)
            return replace(verifier.status, state=LivenessState.FAILED)

    def _track(self, frame: Frame) -> TrackingResult:
        """Fish tracking for ``frame``; a vision bug costs the fish channel, not the sensor.

        An empty result means no fish samples this frame; if vision keeps
        failing, ``fish_presence`` fails the fish channel on its own.
        """
        try:
            return self.tracker.process(self._vision_frame(frame))
        except Exception as exc:
            self._component_error("vision", exc)
            return TrackingResult(frame.index, frame.timestamp, (), (), ())

    def _component_error(self, component: str, exc: Exception) -> None:
        """Count and log a failure of one component that the frame survived."""
        self._errors += 1
        self._last_error = f"{component}: {type(exc).__name__}: {exc}"
        self._component_failures[component] += 1
        n = self._component_failures[component]
        if n == 1 or n % 1000 == 0:
            logger.exception("%s failed on a frame (%d times); continuing", component, n)

    def _credit(self, out: ChannelOutput) -> float:
        """Health-test the samples, refresh the estimate, absorb and credit if usable."""
        name = out.channel
        health = self.health
        health.observe_samples(name, out.samples, out.t)
        estimate = self._estimates[name]
        if estimate.feed(out.samples):
            health.observe_entropy_estimate(name, estimate.h_est, out.t)
        state = health.state(name)
        credit = 0.0
        if state is not HealthState.FAILED:
            credit = compute_credit(
                int(out.samples.size),
                estimate.h_est,
                estimate.h_cap,
                self.config.safety_factor,
                health.credit_multiplier(name),
            )
            self.pool.absorb(out, credit)
        self._call(self._listeners, out, credit, state)
        return credit

    def _vision_frame(self, frame: Frame) -> Frame:
        """The frame the tracker sees: downscaled to ``vision_max_width``, LED masked out.

        A blinking LED is a perfect motion blob: unmasked, the tracker
        confirms it as a fish, re-ID names it, and the fish channel credits
        "innovations" driven by our own public challenge. The padded LED
        region is therefore painted with the median colour of its
        surroundings, which follows the lighting like any background and
        never moves.
        """
        max_width = self.config.vision_max_width
        image: npt.NDArray[Any] = frame.image
        scale = 1.0
        if frame.width > max_width:
            scale = max_width / frame.width
            height = max(1, round(frame.height * scale))
            image = cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)
        self._vision_scale = scale
        roi = self._excluded_roi
        if roi is not None:
            image = _mask_region(image, tuple(round(v * scale) for v in roi), VISION_LED_PADDING)
        if image is frame.image:
            return frame
        return replace(frame, image=np.ascontiguousarray(image))

    def _update_fish(
        self, frame: Frame, tracking: TrackingResult, fish_out: ChannelOutput, fish_credit: float
    ) -> None:
        """Attribute credit to fish, record statistics, refresh names, labels and cards."""
        self._sync_names()
        inv = 1.0 / self._vision_scale
        owners = fish_out.sample_owner or ()
        shares: Counter[str] = Counter(o for o in owners if o is not None)
        n_samples = len(owners)
        characters = self.tracker.characters
        observed = {o.fish_id for o in tracking.observations if o.fish_id is not None}
        for fish_id in observed:
            bits = fish_credit * shares.get(fish_id, 0) / n_samples if n_samples else 0.0
            self._fish_bits[fish_id] = self._fish_bits.get(fish_id, 0.0) + bits
            character = characters.get(fish_id)
            if character is None:
                continue
            prev_seen, prev_dist = self._fish_prev.get(fish_id, (0.0, 0.0))
            seen, dist = character.seen_s, character.distance_px
            if seen < prev_seen or dist < prev_dist:  # statistics restarted
                prev_seen, prev_dist = 0.0, 0.0
            self._fish_prev[fish_id] = (seen, dist)
            if self.stats is not None:
                try:
                    self.stats.record(
                        fish_id,
                        t=frame.timestamp,
                        wall_time=frame.wall_time,
                        name=self._names.get(fish_id, fish_id),
                        bits=bits,
                        seen_s=seen - prev_seen,
                        distance_px=(dist - prev_dist) * inv,
                        chaos_score=chaos_fraction(character.chaos_score),
                    )
                except Exception as exc:  # a full disk must not stop the entropy
                    self._component_error("fish statistics", exc)
        self._fish_view = self._build_fish_view(tracking, characters, inv)
        self._maybe_save_gallery(frame.timestamp)

    def _sync_names(self) -> None:
        """Apply ``fish_names`` overrides to new identities; refresh names and labels."""
        identities = self.tracker.identities
        if frozenset(identities) != self._known_fish:
            overrides = self.config.fish_names
            for fish_id in identities.keys() - self._known_fish:
                name = overrides.get(fish_id)
                if name is not None and identities[fish_id].name != name:
                    self.tracker.reid.rename(fish_id, name)
            self._known_fish = frozenset(identities)
        names = dict(self.config.fish_names)
        names.update({fid: ident.name for fid, ident in identities.items()})
        labels = {fid: FishLabel(ident.name, ident.color_hex) for fid, ident in identities.items()}
        if names != self._names:
            self._names = names
        if labels != self._labels:
            self._labels = labels

    def _build_fish_view(
        self, tracking: TrackingResult, characters: Mapping[str, FishCharacter], inv: float
    ) -> tuple[_FishView, ...]:
        present = {
            track.fish_id: track.track_id
            for track in tracking.tracks
            if track.confirmed and track.fish_id is not None
        }
        view = []
        for fish_id in sorted(self._labels, key=_fish_sort_key):
            label = self._labels[fish_id]
            character = characters.get(fish_id)
            view.append(
                _FishView(
                    fish_id=fish_id,
                    name=label.name,
                    color=label.color,
                    present=fish_id in present,
                    track_id=present.get(fish_id),
                    speed_avg=character.speed_avg * inv if character else 0.0,
                    speed_max=character.speed_max * inv if character else 0.0,
                    activity=min(1.0, max(0.0, character.activity)) if character else 0.0,
                    chaos_score=chaos_fraction(character.chaos_score) if character else 0.0,
                    distance_px=character.distance_px * inv if character else 0.0,
                    seen_s=character.seen_s if character else 0.0,
                    bits_session=self._fish_bits.get(fish_id, 0.0),
                )
            )
        return tuple(view)

    def _publish(
        self, frame: Frame, tracking: TrackingResult, fps: float, proc_fps: float
    ) -> EngineSnapshot:
        t = frame.timestamp
        reports = self.health.reports(t)
        channels = self._channel_stats(reports)
        credits = self.pool.credits()
        inv = 1.0 / self._vision_scale
        tracks = (
            tracking.tracks if inv == 1.0 else tuple(_scale_track(s, inv) for s in tracking.tracks)
        )
        snap = EngineSnapshot(
            frame_index=frame.index,
            t=t,
            wall_time=frame.wall_time,
            fps=fps,
            mode=frame.mode,
            width=frame.width,
            height=frame.height,
            tracks=tracks,
            channels=channels,
            pool={name: credits.get(name, 0.0) for name in CAMERA_CHANNELS},
            health=reports,
            overall=self.health.overall(),
            liveness=self._liveness_status,
            source_id=frame.source_id,
            processing_fps=proc_fps,
            pool_capacity_bits=self.pool.capacity_bits,
            fish=self._labels,
            liveness_policy=self.liveness_policy,
            run_state=self._run_state,
            errors=self._errors,
            last_error=self._last_error,
        )
        self._snapshot = snap
        self._record_history(snap)
        return snap

    def _channel_stats(self, reports: Mapping[str, HealthReport]) -> dict[str, ChannelStats]:
        out = {
            name: ChannelStats(
                name=name,
                bits_per_s=self._rates[name].value,
                h_per_sample=self._estimates[name].h_est,
                credited_total=self._totals[name],
                health=reports[name].state,
            )
            for name in CAMERA_CHANNELS
        }
        out[CHANNEL_SYSTEM] = ChannelStats(
            name=CHANNEL_SYSTEM,
            bits_per_s=self._rates[CHANNEL_SYSTEM].value,
            h_per_sample=None,
            credited_total=self._totals[CHANNEL_SYSTEM],
            health=reports[CHANNEL_SYSTEM].state if CHANNEL_SYSTEM in reports else HealthState.OK,
        )
        return out

    def _record_history(self, snap: EngineSnapshot) -> None:
        due = self._next_history
        if due is not None and snap.t < due and snap.t >= due - 2 * HISTORY_PERIOD_S:
            return
        self._next_history = snap.t + HISTORY_PERIOD_S
        item = {
            "t": snap.t,
            "mode": snap.mode.value,
            "channels": {
                name: {
                    "bits_per_s": round(c.bits_per_s, 3),
                    "h_per_sample": round(c.h_per_sample, 4)
                    if c.h_per_sample is not None
                    else None,
                    "credited_total": round(c.credited_total, 3),
                    "health": c.health.value,
                }
                for name, c in snap.channels.items()
            },
            "pool": {**snap.pool, "capacity_bits": snap.pool_capacity_bits},
        }
        with self._history_lock:
            if self._history and snap.t < self._history[-1]["t"]:
                self._history.clear()  # the frame clock restarted (new source)
            self._history.append(item)

    def _blank_snapshot(self) -> EngineSnapshot:
        reports = self.health.reports()
        return EngineSnapshot(
            frame_index=-1,
            t=0.0,
            wall_time=time.time(),
            fps=0.0,
            mode=LightMode.DAY,
            width=0,
            height=0,
            tracks=(),
            channels=self._channel_stats(reports),
            pool={name: 0.0 for name in CAMERA_CHANNELS},
            health=reports,
            overall=self.health.overall(),
            liveness=self._liveness_status,
            source_id=getattr(self.source, "source_id", "unknown"),
            pool_capacity_bits=self.pool.capacity_bits,
            liveness_policy=self.liveness_policy,
        )

    def _make_sensor(self, roi: Roi | None) -> SensorChannel:
        self._excluded_roi = roi
        excluded: tuple[Roi, ...] = ()
        if roi is not None:
            x, y, w, h = roi
            pad = LED_ROI_PADDING
            excluded = ((x - pad, y - pad, w + 2 * pad, h + 2 * pad),)
        return SensorChannel(
            SensorChannelConfig(
                n_sites=self.config.sensor_sites, exclude_rois=excluded, site_seed=self._site_seed
            )
        )

    # -- background loop ----------------------------------------------------------

    def _run(self) -> None:
        consecutive = 0
        try:
            while not self._stop_event.is_set():
                try:
                    snap = self._step()
                except Exception as exc:
                    if self._closed:
                        break
                    consecutive += 1
                    self._record_error(exc, consecutive)
                    if consecutive >= self.max_consecutive_errors:
                        self._run_state = EngineState.ERROR
                        logger.error(
                            "engine stopped after %d consecutive frame failures", consecutive
                        )
                        break
                    self._stop_event.wait(min(0.05 * consecutive, 1.0))
                    continue
                consecutive = 0
                if snap is None:
                    if not self.live_source:
                        self._run_state = EngineState.ENDED
                        logger.info("frame source %s ended", getattr(self.source, "source_id", "?"))
                        break
                    self._stop_event.wait(_LIVE_RETRY_S)
        finally:
            self._stop_event.set()  # also stops the watchdog

    def _record_error(self, exc: BaseException, consecutive: int) -> None:
        self._errors += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        if consecutive == 1:
            logger.exception("engine frame failed")
        elif consecutive % 10 == 0:
            logger.warning(
                "engine frame failed %d times in a row: %s", consecutive, self._last_error
            )

    def _watch(self) -> None:
        """Advance ``source_timeout`` while a live source is stuck inside ``read()``."""
        while not self._stop_event.wait(_WATCHDOG_PERIOD_S):
            started = self._read_started
            if started is not None and self._clock() - started >= _STALL_CHECK_S:
                self._check_stalled()

    def _check_stalled(self) -> None:
        """Run the timeout check on the extrapolated frame clock and republish health."""
        if not self._process_lock.acquire(blocking=False):
            return  # a frame is being processed right now: it carries fresh health
        try:
            mono = self._clock()
            if self._last_frame_t is None or self._last_frame_mono is None:
                now = mono  # live cameras stamp frames with this very clock
            else:
                now = self._last_frame_t + (mono - self._last_frame_mono)
            self.health.check_timeout(now)
            snap = self._snapshot
            if snap is not None:
                reports = self.health.reports(now)
                self._snapshot = replace(
                    snap,
                    health=reports,
                    overall=self.health.overall(),
                    channels=self._channel_stats(reports),
                )
        finally:
            self._process_lock.release()

    def _pump(self) -> None:
        """``rng.wait_for_credit`` hook: step synchronously when no thread is running."""
        if not (self.running or self._closed or self._ended):
            self._step()

    # -- LED, gallery ------------------------------------------------------------

    def _start_led(self) -> None:
        if self.led is not None and not self._led_started:
            self._led_started = True
            self.led.start()

    def _close_led(self) -> None:
        if self.led is None:
            return
        try:
            self.led.stop()  # switches the LED off
        finally:
            self.led.driver.close()

    def _load_gallery(self) -> None:
        path = self.gallery_path
        if path is None or not path.exists():
            return
        try:
            self.tracker.reid.load(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            backup = path.with_name(path.name + ".bad")
            logger.warning("cannot load fish gallery %s (%s); moved to %s", path, exc, backup)
            try:
                path.replace(backup)
            except OSError:
                logger.exception("cannot move %s aside", path)

    def _maybe_save_gallery(self, t: float) -> None:
        if self.gallery_path is None:
            return
        if (
            self._next_gallery_save is None
            or t < self._next_gallery_save - 2 * self._gallery_save_s
        ):
            self._next_gallery_save = t + self._gallery_save_s
            return
        if t >= self._next_gallery_save:
            self._next_gallery_save = t + self._gallery_save_s
            self._save_gallery()

    def _save_gallery(self) -> None:
        if self.gallery_path is None:
            return
        try:
            self.tracker.reid.save(self.gallery_path)
        except OSError:
            logger.exception("cannot save fish gallery to %s", self.gallery_path)


def _scale_track(track: TrackSnapshot, k: float) -> TrackSnapshot:
    x1, y1, x2, y2 = track.bbox
    cx, cy = track.center
    vx, vy = track.velocity
    return replace(
        track,
        bbox=(x1 * k, y1 * k, x2 * k, y2 * k),
        center=(cx * k, cy * k),
        velocity=(vx * k, vy * k),
        trail=tuple((x * k, y * k) for x, y in track.trail),
    )


def _mask_region(image: npt.NDArray[np.uint8], roi: tuple[int, ...], pad: int) -> Any:
    """A copy of ``image`` with ``roi`` (padded) filled by the median of a 2-px ring around it."""
    height, width = image.shape[:2]
    x, y, w, h = roi
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    if x0 >= x1 or y0 >= y1:
        return image
    ox0, oy0, ox1, oy1 = max(0, x0 - 2), max(0, y0 - 2), min(width, x1 + 2), min(height, y1 + 2)
    channels = image.shape[2] if image.ndim == 3 else 1
    ring = np.concatenate(
        [
            image[oy0:y0, ox0:ox1].reshape(-1, channels),
            image[y1:oy1, ox0:ox1].reshape(-1, channels),
            image[y0:y1, ox0:x0].reshape(-1, channels),
            image[y0:y1, x1:ox1].reshape(-1, channels),
        ]
    )
    out = image.copy()
    fill = np.median(ring, axis=0) if ring.size else np.zeros(channels)
    out[y0:y1, x0:x1] = np.round(fill).astype(np.uint8).reshape(-1)
    return out


def _fish_sort_key(fish_id: str) -> tuple[int, int, str]:
    """Natural order: f1, f2, ..., f10, then any other ids alphabetically."""
    match = _FISH_ID_NUMBER.match(fish_id)
    return (0, int(match.group(1)), fish_id) if match else (1, 0, fish_id)
