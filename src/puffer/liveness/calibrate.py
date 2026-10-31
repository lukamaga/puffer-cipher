"""Find the LED in the picture by correlating image regions with the challenge.

Nobody should have to click on the LED in a dashboard, and the camera may be
bumped or re-mounted at any time, so the region of interest is learnt from the
video itself: the LED is *the region whose brightness follows the commanded
pattern*. That definition is also what makes calibration robust to things that
merely blink - a filter's status light, a flickering lamp, reflections, or a
deliberate decoy driven by an independent random pattern. Such regions vary
just as much as the LED but are uncorrelated with *our* fresh challenge.

Algorithm
---------
1. Every frame is reduced to a grid of small *fine cells* (``resolution``
   pixels square) holding the exact integer sum of the per-pixel max colour
   channel (the brightness definition of :mod:`puffer.liveness.detector`).
   Only this uint16 grid is kept: two bytes per ``resolution**2`` pixels
   instead of three bytes per pixel (115 kB instead of 2.7 MB per 720p frame).
2. An integer 2-D prefix sum over the fine grid gives the exact mean brightness
   time series of any cell-aligned rectangle with four lookups per frame.
3. **Coarse search:** square ``grid``-pixel cells at half-cell stride, scored by
   the Pearson correlation between their brightness series and the commanded
   on/off pattern at every candidate camera lag in ``[0, max_lag_s]`` - one
   matrix product for all cells and lags at once.
4. **Refinement:** around the best cell, every rectangle from ``resolution`` up
   to ``grid`` pixels per side and every offset within half a cell is ranked
   by LED *contrast* (mean brightness when commanded on minus when off); the
   largest box within 90% of the best contrast is the ROI. It hugs the LED
   blob, which gives the detector the full contrast and keeps fish swimming
   nearby out of the measurement.

The score is a correlation coefficient in ``[-1, 1]``: a visible LED scores
close to 1, an independent blinker scores ``O(1/sqrt(transitions))``.
Calibration only *proposes* a region; the cryptographically meaningful test is
the verifier's bit-level match on data recorded after the region was chosen.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from puffer.liveness.detector import Roi, max_channel, validate_roi
from puffer.liveness.schedule import ChallengeSchedule
from puffer.types import Frame, LedSchedule

_MAX_RESOLUTION = 16  # fine-cell sums must fit in uint16


class CalibrationResult(NamedTuple):
    roi: Roi
    score: float  # Pearson correlation of the ROI brightness with the pattern at ``lag_s``
    lag_s: float  # camera latency at which the correlation peaks (coarse estimate)


class CalibrationAccumulator:
    """Collect per-frame fine-cell brightness grids and solve for the LED ROI.

    Parameters
    ----------
    cell:
        Coarse search cell size in pixels (the ``grid`` of :func:`calibrate_roi`).
    resolution:
        Fine cell size in pixels; ROI coordinates are multiples of it. Must
        divide ``cell``.
    keep_s:
        If set, frames older than ``keep_s`` before the newest one are dropped,
        turning the accumulator into a sliding window (used for re-tries).
    """

    def __init__(self, *, cell: int = 16, resolution: int = 4, keep_s: float | None = None) -> None:
        if not 1 <= resolution <= _MAX_RESOLUTION or cell < resolution or cell % resolution:
            raise ValueError(f"resolution must be in 1..{_MAX_RESOLUTION} and divide cell")
        if keep_s is not None and not keep_s > 0:
            raise ValueError("keep_s must be positive")
        self.cell = cell
        self.resolution = resolution
        self.keep_s = keep_s
        self._frames: deque[tuple[float, npt.NDArray[np.uint16]]] = deque()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def times(self) -> npt.NDArray[np.float64]:
        return np.fromiter((t for t, _ in self._frames), dtype=np.float64, count=len(self))

    @property
    def span(self) -> float:
        """Time between the oldest and the newest stored frame."""
        return self._frames[-1][0] - self._frames[0][0] if self._frames else 0.0

    def clear(self) -> None:
        self._frames.clear()

    def add(self, frame: Frame) -> None:
        """Reduce ``frame`` to its fine-cell brightness grid and store it (< 1 ms for VGA)."""
        value = max_channel(frame.image)
        r = self.resolution
        hf, wf = value.shape[0] // r, value.shape[1] // r
        k = self.cell // r
        if hf < k or wf < k:
            raise ValueError(f"frame {value.shape[1]}x{value.shape[0]} is smaller than one cell")
        # Exact integer box sums, rows then columns (uint16 holds 16 * 16 * 255).
        rows = value[: hf * r, : wf * r].reshape(hf, r, wf * r).sum(axis=1, dtype=np.uint16)
        sums = rows.reshape(hf, wf, r).sum(axis=2, dtype=np.uint16)
        if self._frames and self._frames[-1][1].shape != sums.shape:
            raise ValueError("frame size changed during calibration")
        self._frames.append((float(frame.timestamp), sums))
        if self.keep_s is not None:
            newest = self._frames[-1][0]
            while self._frames and self._frames[0][0] < newest - self.keep_s:
                self._frames.popleft()

    def roi_series(self, roi: Roi) -> npt.NDArray[np.float64]:
        """Brightness series of a fine-grid-aligned ``roi`` over the stored frames.

        Identical to :func:`~puffer.liveness.detector.roi_brightness` on the
        original frames, which lets the verifier warm up the detector's levels
        from calibration data without having kept the frames.
        """
        x, y, w, h = validate_roi(roi)
        r = self.resolution
        if x % r or y % r or w % r or h % r:
            raise ValueError(f"ROI {roi!r} is not aligned to the {r}-pixel calibration grid")
        if not self._frames:
            raise ValueError("no frames collected")
        hf, wf = self._frames[0][1].shape
        if (y + h) // r > hf or (x + w) // r > wf:
            raise ValueError(f"ROI {roi!r} exceeds the calibrated image")
        rows, cols = slice(y // r, (y + h) // r), slice(x // r, (x + w) // r)
        sums = np.fromiter(
            (int(fine[rows, cols].sum(dtype=np.int64)) for _, fine in self._frames),
            dtype=np.float64,
            count=len(self),
        )
        return sums / float(w * h)

    def solve(
        self,
        schedule: LedSchedule,
        *,
        max_lag_s: float = 0.5,
        lag_step_s: float | None = None,
        min_frames: int = 8,
    ) -> CalibrationResult:
        """Return the region that best follows ``schedule`` over the stored frames.

        ``lag_step_s`` defaults to a quarter of the median frame interval: at
        30 fps every third frame sits on a 100 ms chip edge, so a lag grid of
        whole frames would misjudge those frames and blunt the correlation.
        """
        if len(self) < max(min_frames, 2):
            raise ValueError(f"need at least {max(min_frames, 2)} frames to calibrate")
        times = self.times
        lags = lag_grid(max_lag_s, lag_step_s or _frame_interval(times) / 4)
        commanded = commanded_states(schedule, times[:, None] - lags[None, :])
        pattern = _unit_columns(commanded.astype(np.float64))

        stack = self._stack()
        n_frames, hf, wf = stack.shape
        pixels = self.resolution**2
        # int32 halves the transient memory and is exact while a whole frame's sum fits.
        fits32 = hf * wf * pixels * 255 <= np.iinfo(np.int32).max
        prefix = np.zeros((n_frames, hf + 1, wf + 1), dtype=np.int32 if fits32 else np.int64)
        prefix[:, 1:, 1:] = stack.cumsum(axis=1, dtype=prefix.dtype).cumsum(axis=2)

        # Coarse detection: correlation is scale-free, so it finds the LED
        # neighbourhood whatever the LED's brightness or the lighting.
        k = self.cell // self.resolution
        stride = max(1, k // 2)
        cy, cx = (
            a.ravel()
            for a in np.meshgrid(
                np.arange(0, hf - k + 1, stride), np.arange(0, wf - k + 1, stride), indexing="ij"
            )
        )
        side = np.full(cy.size, k)
        coarse = _unit_columns(_box_series(prefix, cy, cx, side, side, pixels)).T @ pattern
        cell, lag = _argmax2(coarse)
        on = commanded[:, lag]
        r = self.resolution
        if coarse[cell, lag] <= 0 or on.all() or not on.any():
            roi = (int(cx[cell]) * r, int(cy[cell]) * r, k * r, k * r)
            return CalibrationResult(roi, float(coarse[cell, lag]), float(lags[lag]))

        # Refinement: correlation saturates for any box that holds a few LED
        # pixels (averaging more background only lowers the noise), so boxes
        # are ranked by *contrast* instead - how much brighter the region is
        # when the LED is commanded on. Among boxes within 90% of the best
        # contrast, the largest wins: it is still almost all LED, and more
        # pixels average away more sensor noise.
        half = k // 2
        y, x, h, w = (
            a.ravel()
            for a in np.meshgrid(
                np.arange(cy[cell] - half, cy[cell] + half + 1),
                np.arange(cx[cell] - half, cx[cell] + half + 1),
                np.arange(1, k + 1),
                np.arange(1, k + 1),
                indexing="ij",
            )
        )
        fits = (y >= 0) & (x >= 0) & (y + h <= hf) & (x + w <= wf)
        y, x, h, w = y[fits], x[fits], h[fits], w[fits]
        series = _box_series(prefix, y, x, h, w, pixels)
        contrast = series[on].mean(axis=0) - series[~on].mean(axis=0)
        good = contrast >= 0.9 * float(np.max(contrast))
        area = np.where(good, h * w, -1)
        box = int(np.lexsort((contrast, area))[-1])  # largest good box, then highest contrast
        corr = _unit_columns(series[:, box : box + 1]).T @ pattern
        best_lag = int(np.argmax(corr[0]))
        roi = (int(x[box]) * r, int(y[box]) * r, int(w[box]) * r, int(h[box]) * r)
        return CalibrationResult(roi, float(corr[0, best_lag]), float(lags[best_lag]))

    def _stack(self) -> npt.NDArray[np.uint16]:
        return np.stack([fine for _, fine in self._frames])


def calibrate_roi(
    frames: Iterable[Frame],
    schedule: LedSchedule,
    grid: int = 16,
    max_lag_s: float = 0.5,
    *,
    lag_step_s: float | None = None,
    resolution: int = 4,
) -> CalibrationResult:
    """Find the LED region in ``frames`` (see the module docstring).

    ``grid`` is the coarse cell size in pixels, ``max_lag_s`` the largest
    camera latency considered, ``lag_step_s`` the lag spacing (default: a
    quarter of the median frame interval). Returns the ROI, its correlation score and the
    lag at which it peaks; callers decide whether the score is good enough.
    """
    acc = CalibrationAccumulator(cell=grid, resolution=resolution)
    for frame in frames:
        acc.add(frame)
    return acc.solve(schedule, max_lag_s=max_lag_s, lag_step_s=lag_step_s)


def lag_grid(max_lag_s: float, step_s: float) -> npt.NDArray[np.float64]:
    """Candidate camera lags ``0, step, 2*step, ... <= max_lag_s``."""
    if not (math.isfinite(max_lag_s) and max_lag_s >= 0):
        raise ValueError("max_lag_s must be a non-negative finite number")
    if not (math.isfinite(step_s) and step_s > 0):
        raise ValueError("lag step must be a positive finite number")
    n = math.floor(max_lag_s / step_s + 1e-9) + 1
    return np.arange(n, dtype=np.float64) * step_s


def commanded_states(schedule: LedSchedule, times: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Commanded LED states at ``times`` (vectorised for :class:`ChallengeSchedule`)."""
    t = np.asarray(times, dtype=np.float64)
    if isinstance(schedule, ChallengeSchedule):
        return schedule.states_at(t)
    flat = np.fromiter((schedule.state_at(float(v)) for v in t.ravel()), dtype=np.bool_)
    return flat.reshape(t.shape)


def _frame_interval(times: npt.NDArray[np.float64]) -> float:
    steps = np.diff(np.sort(times))
    steps = steps[steps > 0]
    if steps.size == 0:
        raise ValueError("frames need distinct timestamps")
    return float(np.median(steps))


def _unit_columns(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Centre each column and scale it to unit norm; constant columns become 0.

    With unit columns, a dot product *is* the Pearson correlation, so scoring
    every (region, lag) pair is a single matrix product.
    """
    centred = a - a.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->j", centred, centred))
    scale = np.zeros_like(norm)
    np.divide(1.0, norm, out=scale, where=norm > 1e-9 * math.sqrt(a.shape[0]))
    return centred * scale


def _box_series(
    prefix: npt.NDArray[np.signedinteger],
    y: npt.NDArray[np.int64],
    x: npt.NDArray[np.int64],
    h: npt.NDArray[np.int64],
    w: npt.NDArray[np.int64],
    pixels_per_cell: int,
) -> npt.NDArray[np.float64]:
    """Mean brightness of every box (fine-grid units) in every frame: ``(frames, boxes)``."""
    y1, x1 = y + h, x + w
    sums = prefix[:, y1, x1] - prefix[:, y, x1] - prefix[:, y1, x] + prefix[:, y, x]
    means: npt.NDArray[np.float64] = sums / (h * w * pixels_per_cell).astype(np.float64)
    return means


def _argmax2(a: npt.NDArray[np.float64]) -> tuple[int, int]:
    i, j = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(i), int(j)
