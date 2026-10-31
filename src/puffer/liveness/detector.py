"""Per-frame LED observation: ROI brightness and an adaptive on/off decision.

Brightness
----------
The ROI brightness is the mean over the region of the per-pixel *maximum* over
the colour channels (the V of HSV). A coloured LED (red, blue, ...) lights up
one channel strongly while a luminance average would dilute it by the other
two; an infrared camera replicates one grey channel into three, for which the
maximum is simply the grey level. One definition therefore works for day and
night frames without knowing the LED colour.

Threshold
---------
Lighting in an aquarium is anything but constant (room lights, sunlight,
day/night lamps, auto-exposure), so a fixed threshold would fail. The detector
keeps an online two-cluster model - running estimates of the "off" and "on"
brightness levels, i.e. 1-D k-means with a forgetting factor - and classifies
against their midpoint with a hysteresis band, so a frame captured during a
transition does not make the state chatter.

* A cluster's level is the exact running mean of its first samples and an
  exponential moving average afterwards (fast start, then slow tracking).
* A cluster that has not been seen for ``stale_frames`` frames relaxes toward
  the other one. Manchester coding guarantees both levels appear every
  symbol, so a stale cluster means the LED is stuck, covered, or the lighting
  jumped (e.g. the on-level dropped below the old midpoint). Relaxing shrinks
  the contrast, which drops confidence (LED covered) or lets the levels
  re-form around the new brightness (lighting change) instead of stranding.
* ``confident`` requires real contrast (``min_contrast`` grey levels and
  ``noise_factor`` times the observed in-cluster noise) and a sample clearly
  away from the midpoint. The verifier only votes with confident frames.

The detector is pure signal processing; deciding *liveness* from the on/off
stream is the job of :mod:`puffer.liveness.verifier`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from puffer.types import Frame

Roi = tuple[int, int, int, int]
"""Region of interest in pixels: ``(x, y, w, h)``."""


@dataclass(frozen=True, slots=True)
class LedObservation:
    """The detector's verdict on one frame."""

    t: float
    brightness: float  # mean max-channel value in the ROI, 0..255
    on: bool
    confident: bool  # clear contrast and far enough from the threshold to vote


def validate_roi(roi: Roi, shape: tuple[int, ...] | None = None) -> Roi:
    """Normalise ``roi`` to ints and check it lies inside an image of ``shape``."""
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError(f"invalid ROI {roi!r}: need x, y >= 0 and w, h > 0")
    if shape is not None and (y + h > shape[0] or x + w > shape[1]):
        raise ValueError(f"ROI {roi!r} exceeds image of size {shape[1]}x{shape[0]}")
    return (x, y, w, h)


def max_channel(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Per-pixel maximum over the colour channels (the V of HSV) of an HxW[xC] image.

    Chained ``np.maximum`` over channel views: an order of magnitude faster
    than ``image.max(axis=2)``, whose reduction runs along the tiny, strided
    channel axis.
    """
    if image.ndim == 2:
        return image
    value = image[..., 0]
    for c in range(1, image.shape[2]):
        value = np.maximum(value, image[..., c])
    return value


def roi_brightness(image: npt.NDArray[np.uint8], roi: Roi) -> float:
    """Mean over ``roi`` of the per-pixel maximum colour channel (HSV value)."""
    x, y, w, h = validate_roi(roi, image.shape)
    return float(max_channel(image[y : y + h, x : x + w]).mean())


class LedDetector:
    """Turn ROI brightness into one on/off observation per frame.

    Parameters
    ----------
    roi:
        ``(x, y, w, h)`` of the LED, or ``None`` until calibration sets it.
    adapt_rate:
        Forgetting factor of the level estimates once warmed up (per sample of
        that cluster); sets how fast slow illumination drift is followed.
    hysteresis:
        Width of the dead band around the midpoint, as a fraction of the
        half-contrast. Inside it the previous state is kept.
    confidence_margin:
        A confident sample must lie at least this fraction of the
        half-contrast away from the midpoint.
    min_contrast:
        Minimal on/off level difference (grey levels) to trust any decision.
    noise_factor:
        The contrast must also exceed this multiple of the mean absolute
        in-cluster deviation (an SNR requirement).
    stale_frames:
        Frames without a sample before a cluster starts relaxing.
    """

    def __init__(
        self,
        roi: Roi | None = None,
        *,
        adapt_rate: float = 0.05,
        hysteresis: float = 0.2,
        confidence_margin: float = 0.4,
        min_contrast: float = 10.0,
        noise_factor: float = 4.0,
        stale_frames: int = 30,
    ) -> None:
        if not 0.0 < adapt_rate <= 1.0:
            raise ValueError("adapt_rate must be in (0, 1]")
        if not 0.0 <= hysteresis < 1.0:
            raise ValueError("hysteresis must be in [0, 1)")
        if not 0.0 <= confidence_margin < 1.0:
            raise ValueError("confidence_margin must be in [0, 1)")
        if min_contrast < 0 or noise_factor < 0:
            raise ValueError("min_contrast and noise_factor must be non-negative")
        if stale_frames < 1:
            raise ValueError("stale_frames must be >= 1")
        self._roi = None if roi is None else validate_roi(roi)
        self.adapt_rate = adapt_rate
        self.hysteresis = hysteresis
        self.confidence_margin = confidence_margin
        self.min_contrast = min_contrast
        self.noise_factor = noise_factor
        self.stale_frames = stale_frames
        self.reset()

    # ------------------------------------------------------------- properties

    @property
    def roi(self) -> Roi | None:
        return self._roi

    @roi.setter
    def roi(self, roi: Roi | None) -> None:
        """Move the ROI; the learnt levels belong to the old region and are reset."""
        self._roi = None if roi is None else validate_roi(roi)
        self.reset()

    @property
    def levels(self) -> tuple[float, float] | None:
        """``(off_level, on_level)`` once at least one sample was seen."""
        if self._n == 0:
            return None
        return (self._lo, self._hi)

    @property
    def threshold(self) -> float | None:
        """Midpoint between the two levels (``None`` before the first sample)."""
        return None if self._n == 0 else 0.5 * (self._lo + self._hi)

    @property
    def contrast(self) -> float:
        return 0.0 if self._n == 0 else self._hi - self._lo

    # ---------------------------------------------------------------- updates

    def reset(self) -> None:
        """Forget the learnt levels (keeps the ROI and parameters)."""
        self._lo = 0.0
        self._hi = 0.0
        self._n_lo = 0
        self._n_hi = 0
        self._since_lo = 0
        self._since_hi = 0
        self._noise = 0.0
        self._n = 0
        self._on = False

    def observe(self, frame: Frame) -> LedObservation:
        """Measure the ROI in ``frame`` and classify it."""
        if self._roi is None:
            raise RuntimeError("LedDetector has no ROI yet (configure one or calibrate)")
        return self.update(frame.timestamp, roi_brightness(frame.image, self._roi))

    def update(self, t: float, brightness: float) -> LedObservation:
        """Classify one brightness sample and adapt the two-cluster model."""
        b = float(brightness)
        if not math.isfinite(b):
            raise ValueError("brightness must be finite")
        self._n += 1
        if self._n == 1:
            # No contrast yet: seed both clusters; the first real assignment to
            # each one overwrites the seed (running-mean weight 1/1).
            self._lo = self._hi = b
            self._on = False
            return LedObservation(t, b, on=False, confident=False)

        contrast = self._hi - self._lo
        mid = 0.5 * (self._hi + self._lo)
        band = self.hysteresis * 0.5 * contrast
        on = b >= mid - band if self._on else b > mid + band
        confident = (
            contrast >= self.min_contrast
            and contrast >= self.noise_factor * self._noise
            and abs(b - mid) >= self.confidence_margin * 0.5 * contrast
        )
        self._on = on
        self._adapt(b, on)
        return LedObservation(t, b, on=on, confident=confident)

    def _adapt(self, b: float, on: bool) -> None:
        # Invariant lo <= hi: an "off" sample never exceeds hi and an "on" sample
        # never falls below lo (the hysteresis band is narrower than the contrast),
        # and relaxation moves a level toward the other one without passing it.
        if on:
            self._n_hi += 1
            rate = max(self.adapt_rate, 1.0 / self._n_hi)
            if self._n_hi > 1:  # the first sample's distance to the seed is not noise
                self._noise += rate * (abs(b - self._hi) - self._noise)
            self._hi += rate * (b - self._hi)
            self._since_hi, self._since_lo = 0, self._since_lo + 1
        else:
            self._n_lo += 1
            rate = max(self.adapt_rate, 1.0 / self._n_lo)
            if self._n_lo > 1:
                self._noise += rate * (abs(b - self._lo) - self._noise)
            self._lo += rate * (b - self._lo)
            self._since_lo, self._since_hi = 0, self._since_hi + 1

        if self._since_hi > self.stale_frames:
            self._hi += self.adapt_rate * (self._lo - self._hi)
        if self._since_lo > self.stale_frames:
            self._lo += self.adapt_rate * (self._hi - self._lo)
