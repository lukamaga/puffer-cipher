"""Entropy channels: turn camera frames and fish tracks into pool input.

Each channel produces a :class:`~puffer.types.ChannelOutput` with two very
different parts, and keeping them apart is the core design decision of the
entropy subsystem (DESIGN.md section 4.1):

* ``absorb`` - *everything* the channel saw, hashed into the pool. It is
  cheap to hash and can only help: unpredictable bytes add entropy, and
  predictable bytes cost nothing, because pool output is always keyed with
  fresh OS randomness.
* ``samples`` - a small, well-defined sequence of uint8 symbols whose
  statistics we can actually estimate (SP 800-90B style). Credit is computed
  from these samples only, never from ``absorb``: the raw frame is highly
  structured (the tank, the gravel) and its byte statistics say nothing
  about its unpredictability.

Channels
--------
``sensor``
    Temporal pixel noise: ``(I_t - I_{t-1}) mod 256`` of the green channel
    at ``K`` fixed sample sites. Shot and read noise make this difference
    genuinely random even in a static scene. Sites are chosen once, at a
    random (``os.urandom``-seeded) set of positions, avoiding the border
    (lens vignetting, compression edge artefacts) and the LED region (whose
    brightness is *our own* public challenge pattern, not noise). Randomised
    sites also deny an attacker who controls part of the scene a known set
    of pixels to target. The site order is random too, so spatial
    correlations between neighbouring pixels do not become sequential
    correlations that would bias the health tests.
``fish``
    Kalman innovations - the part of each fish's motion the tracker could
    not predict - quantised to 1/8 pixel; the low 4 bits of each axis form
    one uint8 symbol per confirmed fish per frame, attributed to that fish.

Credit (DESIGN.md section 4.3) is ``n_samples * min(H_est, H_cap) * 0.5``,
where ``H_est`` comes from :class:`RollingEstimate` (SP 800-90B Most Common
Value over a sliding window) and ``H_cap`` bounds what we are willing to
believe per sample no matter what the estimator says.
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from puffer.entropy.estimators import (
    InsufficientDataError,
    collision,
    compression,
    lag,
    most_common_value,
    multi_mmc,
    to_bitstring,
)
from puffer.types import (
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    ChannelOutput,
    Frame,
    TrackingResult,
)

__all__ = [
    "DEFAULT_H_CAP",
    "SAFETY_FACTOR",
    "FishChannel",
    "RollingEstimate",
    "SensorChannel",
    "SensorChannelConfig",
    "compute_credit",
]

SAFETY_FACTOR = 0.5
"""Credit only half of the (already conservative, 99% upper-bound) estimate."""

DEFAULT_H_CAP = {CHANNEL_SENSOR: 4.0, CHANNEL_FISH: 3.0}
"""Per-sample credit caps in bits (DESIGN.md section 4.3)."""

_GREEN = 1  # BGR order: green has the most photosites on a Bayer sensor
_FRAME_HEADER = struct.Struct(">Qd")  # u64 frame index, f64 timestamp
_FISH_FIELDS = 10
_FISH_QUANT = 8.0  # innovations are quantised to 1/8 pixel


@dataclass(frozen=True, slots=True)
class SensorChannelConfig:
    """Where the sensor channel samples.

    ``exclude_rois`` are ``(x, y, w, h)`` rectangles (typically the LED);
    ``site_seed`` makes site selection reproducible (tests, analysis runs) -
    leave it ``None`` in production so sites come from ``os.urandom``.
    """

    n_sites: int = 4096
    border: int = 8
    exclude_rois: Sequence[tuple[int, int, int, int]] = ()
    site_seed: bytes | None = None

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError("n_sites must be positive")
        if self.border < 0:
            raise ValueError("border must be non-negative")
        for roi in self.exclude_rois:
            if len(roi) != 4 or roi[2] < 0 or roi[3] < 0:
                raise ValueError(f"exclude_rois entries are (x, y, w, h) with w, h >= 0: {roi!r}")


class SensorChannel:
    """Temporal sensor noise at fixed random pixel sites (channel ``"sensor"``).

    Not thread-safe: one instance per frame stream.
    """

    name: ClassVar[str] = CHANNEL_SENSOR

    def __init__(self, config: SensorChannelConfig | None = None) -> None:
        self.config = config or SensorChannelConfig()
        self._shape: tuple[int, int] | None = None
        self._ys: npt.NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._xs: npt.NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._prev: npt.NDArray[np.uint8] | None = None

    @property
    def sites(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """``(ys, xs)`` of the sample sites, in sampling order (empty before the first frame)."""
        return self._ys, self._xs

    def process(self, frame: Frame) -> ChannelOutput:
        image = frame.image
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected an HxWx3 image, got shape {image.shape}")
        shape = (int(image.shape[0]), int(image.shape[1]))
        if shape != self._shape:
            self._choose_sites(*shape)
        values = image[self._ys, self._xs, _GREEN]
        # uint8 subtraction wraps around, which is exactly "mod 256".
        samples = np.empty(0, dtype=np.uint8) if self._prev is None else values - self._prev
        self._prev = values
        absorb = _FRAME_HEADER.pack(frame.index, frame.timestamp) + image.tobytes()
        return ChannelOutput(
            channel=self.name,
            frame_index=frame.index,
            t=frame.timestamp,
            absorb=absorb,
            samples=samples,
        )

    def _choose_sites(self, height: int, width: int) -> None:
        """Pick sample sites for a new frame size and drop the previous frame."""
        cfg = self.config
        valid = np.zeros((height, width), dtype=bool)
        b = cfg.border
        valid[b : height - b, b : width - b] = True
        for x, y, w, h in cfg.exclude_rois:
            valid[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)] = False
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            raise ValueError(f"no sample sites left in a {width}x{height} frame")
        seed = cfg.site_seed if cfg.site_seed is not None else os.urandom(32)
        picker = np.random.default_rng(int.from_bytes(seed, "big"))
        chosen = picker.choice(candidates, size=min(cfg.n_sites, candidates.size), replace=False)
        ys, xs = np.divmod(chosen, width)
        self._ys, self._xs = ys.astype(np.intp), xs.astype(np.intp)
        self._shape = (height, width)
        self._prev = None


class FishChannel:
    """Unpredictable fish motion: quantised Kalman innovations (channel ``"fish"``)."""

    name: ClassVar[str] = CHANNEL_FISH

    def process(self, result: TrackingResult) -> ChannelOutput:
        """Absorb every observation's motion features; sample its innovation.

        Observations with a non-finite innovation (a degenerate filter
        update) are still absorbed but produce no sample, so they can never
        be credited.
        """
        obs = result.observations
        rows = np.array(
            [
                (
                    o.track_id,
                    o.cx,
                    o.cy,
                    o.vx,
                    o.vy,
                    o.heading,
                    o.accel,
                    o.turn_rate,
                    o.innovation[0],
                    o.innovation[1],
                )
                for o in obs
            ],
            dtype=np.float64,
        ).reshape(-1, _FISH_FIELDS)
        absorb = rows.astype(">f8").tobytes()
        innovation = rows[:, 8:10]
        finite = np.isfinite(innovation).all(axis=1)
        # np.rint rounds half to even like round(); np.mod on floats keeps the
        # low 4 bits of the (two's-complement) integer without int overflow.
        q = np.mod(np.rint(innovation[finite] * _FISH_QUANT), 16).astype(np.uint8)
        samples = ((q[:, 0] << 4) | q[:, 1]).astype(np.uint8)
        owners = tuple(
            o.fish_id if o.fish_id is not None else f"track-{o.track_id}"
            for o, ok in zip(obs, finite, strict=True)
            if ok
        )
        return ChannelOutput(
            channel=self.name,
            frame_index=result.frame_index,
            t=result.t,
            absorb=absorb,
            samples=samples,
            sample_owner=owners,
        )


def compute_credit(
    n_samples: int,
    h_est: float,
    h_cap: float,
    safety_factor: float = SAFETY_FACTOR,
    multiplier: float = 1.0,
) -> float:
    """Credited bits for a batch: ``n * min(H_est, H_cap) * safety_factor * multiplier``.

    ``multiplier`` is the health multiplier (DEGRADED halves credit); it may
    only reduce credit, never inflate it.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if not all(math.isfinite(v) for v in (h_est, h_cap, safety_factor, multiplier)):
        raise ValueError("credit parameters must be finite")
    if h_cap < 0 or not 0.0 <= safety_factor <= 1.0 or not 0.0 <= multiplier <= 1.0:
        raise ValueError("need h_cap >= 0 and safety_factor, multiplier in [0, 1]")
    return n_samples * min(max(h_est, 0.0), h_cap) * safety_factor * multiplier


_BITS_PER_SAMPLE = 8


def _min_entropy_estimate(window: npt.NDArray[np.uint8]) -> float:
    """Minimum of five SP 800-90B estimators over ``window``; see :class:`RollingEstimate`."""
    estimates = [most_common_value(window).min_entropy]
    for estimator in (lag, multi_mmc):
        try:
            estimates.append(estimator(window).min_entropy)
        except InsufficientDataError:
            continue  # too few samples for this predictor: MCV still bounds it
    bits = to_bitstring(window, _BITS_PER_SAMPLE)
    for bit_estimator in (collision, compression):
        try:
            estimates.append(_BITS_PER_SAMPLE * bit_estimator(bits).min_entropy)
        except InsufficientDataError:
            continue
    return min(estimates)


class RollingEstimate:
    """Quick per-sample min-entropy estimate over the last ``window`` samples.

    The estimate is the minimum of five SP 800-90B estimators:

    * Most Common Value (6.3.1) bounds the probability of the likeliest
      symbol. It is cheap but blind to *order*: a deterministic cycle
      ``0,1,2,0,1,2,...`` looks like 1.5 bits per sample to it.
    * Lag prediction (6.3.8) and MultiMMC prediction (6.3.9) try to *predict*
      the next sample from the recent past. Periodic, repeated or
      Markov-dependent data is predictable, so they drive the estimate to
      ~0 bits for exactly the sources MCV overrates (a stuck cycle, a looped
      pattern, a decoy track replaying the same motion).
    * Collision (6.3.2) and Compression (6.3.4) on the *bitstring* of the
      window (section 3.1.3), scaled by 8 bits per sample. Structure inside
      the bits of a byte, such as the small, symmetric pixel differences of
      sensor noise, is invisible to byte-level estimators: on simulated sensor
      data MCV says ~3.3 bits per sample while the bitstring Collision
      estimate, like the full offline assessment, says ~1.7.

    Taking the minimum keeps credit honest: a channel is credited only for
    what neither its symbol frequencies, its sequence structure nor its bit
    patterns explain. The estimate is refreshed every ``refresh_every``
    :meth:`feed` calls (one call per frame), costing ~30 ms per refresh for a
    4096-sample window, so
    the per-frame cost stays well under a millisecond. Until ``min_samples``
    samples exist ``h_est`` is 0: the bounds are valid for small samples but
    so loose that they would trip the low-entropy check during start-up.
    """

    def __init__(
        self,
        *,
        h_cap: float,
        window: int = 4096,
        refresh_every: int = 30,
        min_samples: int = 1024,
    ) -> None:
        if window < 2 or refresh_every < 1 or not 2 <= min_samples <= window or h_cap < 0:
            raise ValueError(
                "need h_cap >= 0, window >= 2, refresh_every >= 1 and 2 <= min_samples <= window"
            )
        self.h_cap = h_cap
        self.window = window
        self.refresh_every = refresh_every
        self.min_samples = min_samples
        self.samples_seen = 0
        self._buf = np.zeros(window, dtype=np.uint8)
        self._pos = 0
        self._count = 0
        self._since_refresh = 0
        self._h: float | None = None

    @property
    def ready(self) -> bool:
        """Whether at least one estimate has been computed."""
        return self._h is not None

    @property
    def h_est(self) -> float:
        """Latest estimate in bits per sample (uncapped; 0.0 until ready)."""
        return self._h if self._h is not None else 0.0

    def feed(self, samples: npt.NDArray[np.uint8]) -> bool:
        """Add one frame's samples; return True if the estimate was refreshed."""
        s = np.asarray(samples, dtype=np.uint8).ravel()
        self.samples_seen += int(s.size)
        s = s[-self.window :]  # older samples would be overwritten anyway
        k = int(s.size)
        if k:
            end = self._pos + k
            if end <= self.window:
                self._buf[self._pos : end] = s
            else:
                split = self.window - self._pos
                self._buf[self._pos :] = s[:split]
                self._buf[: k - split] = s[split:]
            self._pos = end % self.window
            self._count = min(self._count + k, self.window)
        self._since_refresh += 1
        due = self._h is None or self._since_refresh >= self.refresh_every
        if not due or self._count < self.min_samples:
            return False
        if self._count < self.window:
            ordered = self._buf[: self._count]
        else:  # full ring: oldest sample sits at the write position
            ordered = np.concatenate((self._buf[self._pos :], self._buf[: self._pos]))
        # Predictors and bitstring estimators depend on sample order, so the
        # window must be chronological, not in ring-buffer storage order.
        self._h = _min_entropy_estimate(ordered)
        self._since_refresh = 0
        return True

    def credit(self, n_samples: int, multiplier: float = 1.0) -> float:
        """Credit for ``n_samples`` new samples at the current estimate and cap."""
        return compute_credit(n_samples, self.h_est, self.h_cap, SAFETY_FACTOR, multiplier)
