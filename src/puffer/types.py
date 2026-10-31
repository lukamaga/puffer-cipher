"""Shared data types and protocols for every puffer subsystem.

This module is the contract between packages (sources, vision, liveness,
entropy, beacon, api). Keep it dependency-light: only numpy and stdlib.
Changing a field here is an API change for the whole project - see
docs/DESIGN.md.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

BBox = tuple[float, float, float, float]
"""Axis-aligned box in pixels: (x1, y1, x2, y2), x2 > x1, y2 > y1."""

Point = tuple[float, float]


# --------------------------------------------------------------------------
# Frames and sources
# --------------------------------------------------------------------------


class LightMode(enum.StrEnum):
    """Illumination regime of a frame."""

    DAY = "day"  # visible light, colour frames
    NIGHT = "night"  # lights off, infrared camera / IR illuminator


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured video frame.

    ``image`` is always HxWx3 uint8 in BGR order (OpenCV convention). IR /
    grayscale cameras replicate the single channel into three.
    ``timestamp`` is a monotonic clock in seconds that is shared with the
    LED challenge schedule (for live cameras: ``time.monotonic()`` at capture;
    for the simulator: simulated time; for files: presentation timestamp).
    """

    index: int
    timestamp: float
    wall_time: float
    image: npt.NDArray[np.uint8]
    mode: LightMode = LightMode.DAY
    source_id: str = "unknown"

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields frames: camera, video file, simulator."""

    source_id: str
    fps: float

    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted."""
        ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# LED (liveness challenge actuator)
# --------------------------------------------------------------------------


@runtime_checkable
class LedDriver(Protocol):
    """Physical or simulated LED placed inside the aquarium."""

    def set(self, on: bool, t: float) -> None:
        """Switch the LED. ``t`` is the shared monotonic timestamp of the command."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class LedSchedule(Protocol):
    """A server-controlled, unpredictable on/off pattern over time."""

    def state_at(self, t: float) -> bool:
        """Commanded LED state at monotonic time ``t``."""
        ...


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    score: float
    class_id: int = 0
    label: str = "fish"


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Serializable view of one tracked fish at one frame."""

    track_id: int
    fish_id: str | None  # stable identity from re-identification (None = unknown)
    bbox: BBox
    score: float
    center: Point
    velocity: Point  # pixels / second (Kalman estimate)
    trail: tuple[Point, ...]  # recent centres, oldest first
    confirmed: bool
    age_frames: int


@dataclass(frozen=True, slots=True)
class FishObservation:
    """Motion features of one confirmed track at one frame.

    ``innovation`` is the Kalman measurement residual (measured centre minus
    predicted centre, in pixels): the part of the motion the tracker could
    not predict. It is the primary raw material of the fish entropy channel.
    """

    track_id: int
    fish_id: str | None
    t: float
    cx: float
    cy: float
    vx: float
    vy: float
    speed: float  # px / s
    heading: float  # radians, atan2(vy, vx)
    accel: float  # px / s^2 (magnitude of velocity change)
    turn_rate: float  # rad / s
    innovation: Point


@dataclass(frozen=True, slots=True)
class TrackingResult:
    frame_index: int
    t: float
    detections: tuple[Detection, ...]
    tracks: tuple[TrackSnapshot, ...]
    observations: tuple[FishObservation, ...]


# --------------------------------------------------------------------------
# Entropy channels, health, liveness
# --------------------------------------------------------------------------


class HealthState(enum.StrEnum):
    OK = "ok"  # healthy, entropy is credited
    STARTUP = "startup"  # start-up tests still running, not credited yet
    DEGRADED = "degraded"  # usable but reduced credit (e.g. few fish, night)
    FAILED = "failed"  # excluded: not absorbed, not credited


@dataclass(frozen=True, slots=True)
class HealthReport:
    channel: str
    state: HealthState
    reasons: tuple[str, ...] = ()
    checks: dict[str, str] = field(default_factory=dict)  # check name -> HealthState.value


@dataclass(frozen=True, slots=True)
class ChannelOutput:
    """What an entropy channel extracted from one frame.

    ``absorb`` is everything that goes into the pool hash (cheap to hash, may
    be large: the whole raw frame). ``samples`` is a small, well-defined
    sample sequence (uint8 symbols) used for min-entropy estimation and the
    continuous health tests; credit is computed from it, never from ``absorb``.
    ``sample_owner`` optionally maps each sample to a fish id for attribution.
    """

    channel: str
    frame_index: int
    t: float
    absorb: bytes
    samples: npt.NDArray[np.uint8]
    sample_owner: tuple[str | None, ...] | None = None


class LivenessState(enum.StrEnum):
    DISABLED = "disabled"  # no LED configured (e.g. recorded-video MVP)
    CALIBRATING = "calibrating"  # locating LED / collecting first window
    VERIFIED = "verified"  # observed pattern matches the fresh challenge
    FAILED = "failed"  # mismatch: video is stale, replayed or LED hidden


@dataclass(frozen=True, slots=True)
class LivenessStatus:
    state: LivenessState
    match_ratio: float = 0.0  # fraction of symbols matching at best lag
    lag_s: float = 0.0  # estimated camera latency
    confidence_bits: float = 0.0  # -log2(P[chance match]) of current window
    expected: str = ""  # recent commanded symbols, e.g. "0110"
    observed: str = ""  # recent observed symbols at best lag
    window_start: float = 0.0
    window_end: float = 0.0


CHANNEL_SENSOR = "sensor"
CHANNEL_FISH = "fish"
CHANNEL_SYSTEM = "system"
CAMERA_CHANNELS = (CHANNEL_SENSOR, CHANNEL_FISH)
