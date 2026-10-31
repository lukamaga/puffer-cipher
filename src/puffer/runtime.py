"""Wire concrete components into a running system (DESIGN.md section 9.2).

:mod:`puffer.engine` is deliberately ignorant of *which* camera, LED, detector
or database it talks to; this module makes those choices from a
:class:`~puffer.config.PufferConfig`, so the CLI, the API server and the tests
build the system the same way.

Decisions made here
-------------------
* **One challenge schedule, one clock.** A single
  :class:`~puffer.liveness.ChallengeSchedule` is shared by whatever *shows* the
  challenge and the verifier that *checks* it. The simulator renders it on its
  own (simulated) frame clock; a real LED is driven by
  :class:`~puffer.hardware.LedController` on ``time.monotonic()``, the clock
  live cameras stamp their frames with. A video file shows whatever pattern
  was recorded, so when liveness is forced on for one, the verifier compares
  it against fresh bits that did not exist at recording time - that is the
  replay attack the LED defeats.
* **Liveness policy** (``auto``): *required* for the simulator and live
  cameras, *off* for video files (a recording can never pass, so requiring it
  would only mean "never credit recorded fish"). ``advisory`` verifies and
  displays without gating credit; ``off`` builds no verifier
  (``LivenessState.DISABLED``). A camera without an LED keeps ``required``
  under ``auto``: without the challenge nothing distinguishes the tank from
  a looped recording, so its entropy stays uncredited (outputs remain as
  strong as ``os.urandom``) and a warning says how to change that.
* **LED** (``auto``): rendered by the simulator; ``gpio:17`` for a camera on a
  Raspberry Pi; none otherwise.
* **Persistent state** lives in ``data_dir`` (created with mode 0700): the fish
  statistics (``puffer.db``), the re-ID gallery (``fish_gallery.json``), the
  beacon chain (``beacon.db``) and the beacon signing key (``beacon.key``, mode
  0600). The beacon gets its own database file because both stores version
  their schema with ``PRAGMA user_version``, and because ``beacon.db`` holds a
  secret (the pending reveal) that the statistics database does not.
* **The beacon identity is fish-born.** A new beacon signing key is drawn from
  the engine's PufferRNG (OS-keyed, so never weaker than ``os.urandom``).
"""

from __future__ import annotations

import logging
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal

from puffer.beacon import BeaconService, BeaconStore, EntropySource
from puffer.beacon.service import wall_clock_ms
from puffer.config import LivenessPolicy, PufferConfig
from puffer.crypto import load_or_create_signing_key
from puffer.engine import EntropyEngine, ResolvedLiveness
from puffer.entropy import PufferRNG
from puffer.fishstats import FishStats
from puffer.hardware import GpioLed, LedController, NullLed, SimulatedLed
from puffer.liveness import ChallengeSchedule, LivenessVerifier
from puffer.sources import (
    OpenCVCamera,
    Picamera2Camera,
    SyntheticAquarium,
    VideoFileSource,
    open_source,
)
from puffer.types import Detector, FrameSource, LedDriver
from puffer.vision import FishReID, FishTracker, MotionDetector, YoloDetector

__all__ = [
    "BEACON_DB",
    "BEACON_KEY",
    "DEFAULT_GPIO_PIN",
    "FISH_GALLERY",
    "STATS_DB",
    "SourceKind",
    "beacon_entropy",
    "build_beacon",
    "build_engine",
    "ensure_data_dir",
    "resolve_led",
    "resolve_liveness",
    "source_kind",
]

logger = logging.getLogger(__name__)

STATS_DB: Final = "puffer.db"
FISH_GALLERY: Final = "fish_gallery.json"
BEACON_DB: Final = "beacon.db"
BEACON_KEY: Final = "beacon.key"
DEFAULT_GPIO_PIN: Final = 17
BEACON_KEY_PURPOSE: Final = "beacon-key"

SourceKind = Literal["synthetic", "video", "camera"]
_PI_MODEL_FILE: Final = Path("/proc/device-tree/model")


def source_kind(source: FrameSource) -> SourceKind:
    """Classify a source for the ``auto`` policies.

    Anything that is neither the simulator nor a video file is treated as a
    live camera: that is the fail-closed choice (liveness required, read
    failures are retried and time out instead of ending the run).
    """
    if isinstance(source, SyntheticAquarium):
        return "synthetic"
    if isinstance(source, VideoFileSource):
        return "video"
    if isinstance(source, OpenCVCamera | Picamera2Camera):
        return "camera"
    return "camera"


def resolve_liveness(policy: LivenessPolicy, kind: SourceKind) -> ResolvedLiveness:
    """``auto`` -> ``off`` for video files, ``required`` otherwise; explicit values pass through."""
    if policy == "auto":
        return "off" if kind == "video" else "required"
    return policy


def resolve_led(led: str, kind: SourceKind, *, on_pi: bool | None = None) -> str:
    """``auto`` -> ``simulated`` (simulator), ``gpio:17`` (camera on a Pi) or ``none``."""
    if led != "auto":
        return led
    if kind == "synthetic":
        return "simulated"
    pi = _on_raspberry_pi() if on_pi is None else on_pi
    return f"gpio:{DEFAULT_GPIO_PIN}" if kind == "camera" and pi else "none"


def ensure_data_dir(path: Path) -> Path:
    """Create ``path`` (mode 0700) if needed and return it.

    An existing directory is left as it is - it may be shared on purpose -
    but a warning is logged when others can write to it: everything secret
    inside is a private file of its own (0600).
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o022:
        logger.warning("data directory %s is writable by others (mode %o)", path, mode)
    return path


def build_engine(
    config: PufferConfig,
    *,
    source: FrameSource | None = None,
    schedule: ChallengeSchedule | None = None,
) -> EntropyEngine:
    """Build an :class:`EntropyEngine` for ``config`` (``source`` overrides ``config.source``).

    ``schedule`` is the LED challenge; by default a fresh one (``os.urandom``
    bits, ``symbol_s`` from the config). Pass the schedule an injected
    simulator renders, otherwise its LED shows a pattern the verifier has
    never seen. The engine owns everything built here (source, LED,
    statistics) and releases it in :meth:`EntropyEngine.close`.

    Raises:
        puffer.sources.SourceSpecError: malformed ``config.source``.
        puffer.hardware.LedHardwareError: a GPIO LED was requested but is unavailable.
        ImportError: ``detector = "yolo:..."`` without the ``ultralytics`` package.
    """
    data_dir = ensure_data_dir(config.data_dir)
    if schedule is None:
        schedule = ChallengeSchedule(symbol_s=config.symbol_s)
    owned: list[Callable[[], None]] = []
    try:
        if source is None:
            # The simulator renders the schedule itself; other sources ignore it.
            source = open_source(
                config.source, led_schedule=None if config.led == "none" else schedule
            )
            owned.append(source.close)
        kind = source_kind(source)
        policy = resolve_liveness(config.liveness, kind)
        led_mode = resolve_led(config.led, kind)
        led = _led_controller(led_mode, kind, schedule)
        if led is not None:
            owned.append(led.driver.close)

        led_roi = config.led_roi
        if led_roi is None and isinstance(source, SyntheticAquarium):
            led_roi = source.led_roi  # the simulator knows exactly where it draws the LED
        verifier = None
        if policy != "off":
            verifier = LivenessVerifier(schedule, led_roi, window_bits=config.liveness_window_bits)
            if led_mode == "none" and kind != "video":
                logger.warning(
                    "liveness is %s but no LED is configured: camera entropy can never be "
                    "verified; set led=gpio:<pin>, or liveness=advisory/off to accept the risk",
                    policy,
                )

        stats = FishStats(data_dir / STATS_DB)
        owned.append(stats.close)
        engine = EntropyEngine(
            config,
            source,
            tracker=_tracker(config),
            liveness=verifier,
            liveness_policy=policy,
            led=led,
            led_roi=led_roi,
            stats=stats,
            gallery_path=data_dir / FISH_GALLERY,
            live_source=kind == "camera",
        )
    except BaseException:
        for release in reversed(owned):
            try:
                release()
            except Exception:
                logger.exception("cleanup after a failed build_engine")
        raise
    logger.info(
        "engine: source=%s liveness=%s led=%s detector=%s",
        getattr(source, "source_id", "?"),
        policy,
        led_mode,
        config.detector,
    )
    return engine


def beacon_entropy(rng: PufferRNG) -> EntropySource:
    """The beacon's entropy source: fresh PufferRNG bytes plus their integer-only summary.

    Each ``local_random`` consumes pool credit like any secret output (it is
    secret until the next pulse reveals it). The receipt context always
    reports sensor and fish health, so the summary has no null fields.
    """

    def entropy(n: int, purpose: str) -> tuple[bytes, Mapping[str, Any]]:
        data, receipt = rng.generate(n, purpose)
        return data, receipt.summary_for_beacon()

    return entropy


def build_beacon(
    config: PufferConfig,
    engine: EntropyEngine,
    *,
    clock_ms: Callable[[], int] | None = None,
) -> BeaconService:
    """A :class:`BeaconService` on ``data_dir/beacon.db`` signed with ``data_dir/beacon.key``.

    The caller decides whether to run it (``config.beacon_enabled``) and
    closes ``service.store`` when done. ``clock_ms`` (Unix milliseconds) is
    injectable for tests.

    Raises:
        puffer.beacon.BeaconStateError: the stored chain was signed by another key.
        puffer.crypto.FormatError: ``beacon.key`` is malformed.
    """
    data_dir = ensure_data_dir(config.data_dir)
    store = BeaconStore(data_dir / BEACON_DB)
    try:
        signing_key = load_or_create_signing_key(
            data_dir / BEACON_KEY,
            random_bytes=lambda n: engine.rng.generate(n, BEACON_KEY_PURPOSE)[0],
        )
        return BeaconService(
            store,
            signing_key,
            beacon_entropy(engine.rng),
            period_ms=config.beacon_period_ms,
            clock_ms=clock_ms if clock_ms is not None else wall_clock_ms,
        )
    except BaseException:
        store.close()
        raise


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _tracker(config: PufferConfig) -> FishTracker:
    weights = config.yolo_weights
    detector: Detector = YoloDetector(weights) if weights is not None else MotionDetector()
    # A YOLO model trained with one class per fish names the fish itself;
    # generic "fish" labels fall back to appearance matching.
    return FishTracker(detector=detector, reid=FishReID(trust_labels=weights is not None))


def _led_controller(
    led_mode: str, kind: SourceKind, schedule: ChallengeSchedule
) -> LedController | None:
    """A controller for a physical (or virtual) LED; ``None`` when nothing is driven."""
    if led_mode == "none" or kind == "video":
        return None
    if led_mode == "simulated":
        if kind == "synthetic":
            return None  # the simulator renders the schedule on its own clock
        driver: LedDriver = SimulatedLed()  # dry run: records commands, nothing to film
    elif led_mode.startswith("gpio:"):
        driver = GpioLed(int(led_mode.removeprefix("gpio:")))
    else:  # pragma: no cover - PufferConfig validates led
        driver = NullLed()
    return LedController(schedule, driver)


def _on_raspberry_pi() -> bool:
    try:
        return "raspberry pi" in _PI_MODEL_FILE.read_text(errors="replace").lower()
    except OSError:
        return False
