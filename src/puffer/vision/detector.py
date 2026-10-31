"""Weight-free fish detector: background subtraction tuned for aquariums.

Why not OpenCV's MOG2/KNN subtractors? Their background absorbs anything
that stops moving within a few seconds, and fish *rest*: a sleeping fish
would silently vanish from the tracker and its identity would be lost.
:class:`MotionDetector` instead keeps an explicit background image:

* **Initialisation** - the per-pixel median of the first ``history_frames``
  frames. Moving fish occupy any given pixel for only a fraction of those
  frames, so the median shows the empty tank.
* **Selective update** - afterwards the background follows the scene with a
  running average (``update_rate``) *only where no foreground was found*
  (with a safety margin around blobs). Foreground pixels update at the
  much slower ``foreground_update_rate``: a resting fish stays detected for
  minutes, while a decoration that was moved for good is eventually
  absorbed instead of producing a phantom fish forever. As a resting fish
  slowly fades its detection score drops, which ByteTrack's low-score
  second association is designed to exploit.
* **Lighting** - every frame is normalised per channel by the ratio of the
  background median to the frame median (estimated on a sparse pixel
  grid). Slow drift, flicker and white-balance wander are global
  multiplicative changes and cancel out; fish cover too little of the frame
  to move the median.
* **Noise-adaptive threshold** - the decision threshold is
  ``max(threshold, noise_k * sigma)`` where ``sigma`` is a robust (MAD)
  estimate of the per-frame sensor noise, so a noisy night camera does not
  turn into a carpet of false positives.
* **Colour distance** - the maximum absolute difference over the B, G, R
  channels: a fish that differs from the water in a single channel (an
  orange fish on green plants) is still found. Grayscale/IR frames (three
  identical channels) are processed as one channel; switching between
  colour and IR restarts background learning automatically.
* **Blobs** - morphological opening (speckle) and closing (fills fins and
  stripes), connected components with an area filter, and merging of
  boxes that overlap heavily (a fish split by a dark stripe).
* **Ghosts** - something that sat still during learning and then left
  (a fish resting at start-up) leaves a "ghost" blob that the protected
  update would preserve for minutes. Ghosts are recognised because their
  contour is in the background image, not in the current frame, and are
  absorbed immediately. Limitation: while a slow fish is still touching
  its own ghost the two form one blob (an oversized box) until they part.
* **Scene changes** - when most of the frame is foreground, or the noise
  estimate jumps far above its running level, for several consecutive
  frames (lights switched, camera bumped), the background is re-learned.

Detection scores combine blob contrast (relative to the threshold) and
area (relative to ``min_area``) into ``[0, 1]`` so they plug directly into
ByteTrack's high/low score split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from puffer.types import Detection, Frame

UInt8Array = npt.NDArray[np.uint8]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_MAD_TO_SIGMA = 1.4826  # sigma = 1.4826 * MAD for Gaussian noise
_GAIN_LIMITS = (0.2, 5.0)  # beyond this the scene changed; let re-learning handle it
_GHOST_PAD = 3  # pixels of context around a blob for the ghost edge test
_NOISE_FLOOR = 0.5  # DN; keeps the overload test sane on (synthetic) noise-free video
_NOISE_EMA = 0.05  # adaptation rate of the running noise level


@dataclass(frozen=True, slots=True)
class _Blob:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def is_grayscale(image: npt.NDArray[np.uint8], step: int = 8) -> bool:
    """Whether a BGR image is really single-channel (IR camera, channels replicated).

    Checks a sparse grid only: replicated frames are identical everywhere,
    real colour frames differ almost everywhere.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return True
    sample = image[::step, ::step]
    return bool(
        np.array_equal(sample[..., 0], sample[..., 1])
        and np.array_equal(sample[..., 1], sample[..., 2])
    )


def _edge_energy(image: Float32Array, band: BoolArray) -> float:
    """Mean Sobel gradient magnitude (L1, max over channels) on ``band`` pixels."""
    gx = np.abs(np.asarray(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3), np.float32))
    gy = np.abs(np.asarray(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3), np.float32))
    magnitude = gx + gy
    if magnitude.ndim == 3:
        magnitude = magnitude.max(axis=2)
    return float(magnitude[band].mean())


def _kernel(size: int) -> UInt8Array | None:
    if size <= 1:
        return None
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)), np.uint8)


class MotionDetector:
    """Median-background motion detector implementing :class:`puffer.types.Detector`.

    Parameters
    ----------
    history_frames:
        Frames collected for the initial median background. No detections
        are produced until it is complete (:attr:`ready`).
    threshold:
        Minimum colour distance (0..255, after normalisation) for a pixel to
        be foreground; raised automatically on noisy frames.
    min_area, max_area_frac:
        Blob area limits in pixels and as a fraction of the frame.
    update_rate:
        Running-average rate of the background where there is no foreground.
    foreground_update_rate:
        Rate under (dilated) foreground; keeps resting fish visible.
    noise_k:
        Threshold in units of the estimated noise sigma.
    blur_ksize, open_ksize, close_ksize, mask_margin:
        Pre-blur, morphology and background-protection kernel sizes (pixels).
    merge_overlap:
        Merge boxes whose intersection exceeds this fraction of the smaller box.
    ghost_edge_ratio:
        A blob whose boundary edges are this much stronger in the background
        than in the frame is a ghost and is absorbed at once (0 disables).
    relearn_fraction, relearn_noise_factor, relearn_frames:
        A frame is *overloaded* when more than ``relearn_fraction`` of it is
        foreground, or its noise estimate exceeds ``relearn_noise_factor``
        times the running noise level (an unrelated image looks like
        enormous "noise" to the robust estimator, which would otherwise raise
        the threshold until nothing is foreground). After
        ``relearn_frames`` consecutive overloaded frames the background is
        re-learned.
    sample_step:
        Pixel stride of the grid used for medians (lighting and noise).
    label:
        Label attached to every detection.

    Attributes
    ----------
    noise_sigma, last_threshold:
        Diagnostics of the latest frame: robust noise estimate and the
        decision threshold actually used.
    """

    def __init__(
        self,
        history_frames: int = 30,
        threshold: float = 20.0,
        min_area: int = 60,
        max_area_frac: float = 0.2,
        update_rate: float = 0.01,
        *,
        foreground_update_rate: float = 0.0005,
        noise_k: float = 4.5,
        blur_ksize: int = 5,
        open_ksize: int = 3,
        close_ksize: int = 7,
        mask_margin: int = 7,
        merge_overlap: float = 0.6,
        ghost_edge_ratio: float = 1.5,
        relearn_fraction: float = 0.5,
        relearn_noise_factor: float = 4.0,
        relearn_frames: int = 5,
        sample_step: int = 4,
        label: str = "fish",
    ) -> None:
        if history_frames < 1:
            raise ValueError("history_frames must be >= 1")
        if min_area < 1:
            raise ValueError("min_area must be >= 1")
        if not 0.0 < max_area_frac <= 1.0:
            raise ValueError("max_area_frac must be in (0, 1]")
        if not (0.0 <= foreground_update_rate <= update_rate <= 1.0):
            raise ValueError("require 0 <= foreground_update_rate <= update_rate <= 1")
        if blur_ksize > 1 and blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd")
        self.history_frames = int(history_frames)
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.max_area_frac = float(max_area_frac)
        self.update_rate = float(update_rate)
        self.foreground_update_rate = float(foreground_update_rate)
        self.noise_k = float(noise_k)
        self.blur_ksize = int(blur_ksize)
        self.merge_overlap = float(merge_overlap)
        self.ghost_edge_ratio = float(ghost_edge_ratio)
        self.relearn_fraction = float(relearn_fraction)
        self.relearn_noise_factor = float(relearn_noise_factor)
        self.relearn_frames = int(relearn_frames)
        self.sample_step = max(1, int(sample_step))
        self.label = label
        self._open_kernel = _kernel(open_ksize)
        self._close_kernel = _kernel(close_ksize)
        self._margin_kernel = _kernel(mask_margin)
        self._ghost_kernel = np.asarray(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), np.uint8
        )
        self._warmup: list[UInt8Array] = []
        self._bg: Float32Array | None = None
        self._shape: tuple[int, ...] | None = None
        self._fg: BoolArray | None = None
        self._overload = 0
        self._noise_ref: float | None = None
        self.last_threshold = self.threshold
        self.noise_sigma = 0.0

    # ------------------------------------------------------------- properties

    @property
    def ready(self) -> bool:
        """Whether the initial background has been learned."""
        return self._bg is not None

    @property
    def foreground_mask(self) -> BoolArray | None:
        """Cleaned foreground mask of the latest frame (``H x W``), or ``None``.

        Exposed so re-identification can restrict colour histograms to fish
        pixels instead of the whole bounding box.
        """
        return self._fg

    @property
    def background(self) -> UInt8Array | None:
        """Current background estimate as a ``uint8`` BGR image (for debugging/UI)."""
        if self._bg is None:
            return None
        bg = np.clip(np.rint(self._bg), 0, 255).astype(np.uint8)
        if bg.ndim == 2:
            return np.asarray(cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), np.uint8)
        return bg

    def reset(self) -> None:
        """Discard the background and start learning again."""
        self._warmup.clear()
        self._bg = None
        self._shape = None
        self._fg = None
        self._overload = 0
        self._noise_ref = None

    # ---------------------------------------------------------------- detect

    def detect(self, frame: Frame) -> list[Detection]:
        """Detect moving or resting fish in ``frame``."""
        work = self._prepare(frame.image)
        if self._shape is not None and work.shape != self._shape:
            self.reset()  # resolution or colour/IR change
        if self._bg is None:
            self._learn(work)
            return []
        bg = self._bg
        norm = work.astype(np.float32)
        norm *= self._gain(norm, bg)
        absdiff = np.asarray(cv2.absdiff(norm, bg), np.float32)
        if absdiff.ndim == 2:
            dist = absdiff
        else:  # max over channels; cv2.reduce is ~20x faster than ndarray.max(axis=2)
            flat = cv2.reduce(absdiff.reshape(-1, absdiff.shape[2]), 1, cv2.REDUCE_MAX)
            dist = np.asarray(flat, np.float32).reshape(absdiff.shape[:2])

        step = self.sample_step
        self.noise_sigma = _MAD_TO_SIGMA * float(np.median(absdiff[::step, ::step]))
        thr = max(self.threshold, self.noise_k * self.noise_sigma)
        self.last_threshold = thr

        mask = np.greater(dist, thr).view(np.uint8)
        if self._open_kernel is not None:
            mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel), np.uint8)
        if self._close_kernel is not None:
            mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel), np.uint8)

        if self._overloaded(mask):
            self._overload += 1
            self._fg = None
            if self._overload >= self.relearn_frames:
                self.reset()
            return []
        self._overload = 0

        detections = self._blobs(mask, dist, thr, norm, bg)
        self._update_background(norm, bg, mask)
        self._fg = mask.astype(bool)
        return detections

    # ---------------------------------------------------------------- stages

    def _overloaded(self, mask: UInt8Array) -> bool:
        """Scene-change test; updates the running noise level on normal frames."""
        noise_ref = self.noise_sigma if self._noise_ref is None else self._noise_ref
        too_much_foreground = cv2.countNonZero(mask) > self.relearn_fraction * mask.size
        noise_jump = self.noise_sigma > self.relearn_noise_factor * max(noise_ref, _NOISE_FLOOR)
        if too_much_foreground or noise_jump:
            return True
        self._noise_ref = noise_ref + _NOISE_EMA * (self.noise_sigma - noise_ref)
        return False

    def _prepare(self, image: npt.NDArray[np.uint8]) -> UInt8Array:
        img = np.asarray(image, dtype=np.uint8)
        if is_grayscale(img):
            img = img if img.ndim == 2 else np.ascontiguousarray(img[..., 0])
        if self.blur_ksize > 1:
            k = self.blur_ksize
            img = np.asarray(cv2.GaussianBlur(img, (k, k), 0), np.uint8)
        return img

    def _learn(self, work: UInt8Array) -> None:
        self._shape = work.shape
        self._warmup.append(work)
        if len(self._warmup) >= self.history_frames:
            stack = np.stack(self._warmup)
            self._bg = np.median(stack, axis=0).astype(np.float32)
            self._warmup.clear()

    def _gain(self, frame: Float32Array, bg: Float32Array) -> Float32Array:
        step = self.sample_step
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        f_med = np.median(frame[::step, ::step].reshape(-1, channels), axis=0)
        b_med = np.median(bg[::step, ::step].reshape(-1, channels), axis=0)
        gain = np.clip(b_med / np.maximum(f_med, 1.0), *_GAIN_LIMITS).astype(np.float32)
        return gain if channels > 1 else gain.reshape(())

    def _blobs(
        self, mask: UInt8Array, dist: Float32Array, thr: float, norm: Float32Array, bg: Float32Array
    ) -> list[Detection]:
        """Score connected components; absorb ghosts. Mutates ``mask`` and ``bg`` for ghosts."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = mask.shape
        max_area = self.max_area_frac * height * width
        blobs: list[_Blob] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.min_area or area > max_area:
                continue
            x1, y1 = max(0, x - _GHOST_PAD), max(0, y - _GHOST_PAD)
            x2, y2 = min(width, x + w + _GHOST_PAD), min(height, y + h + _GHOST_PAD)
            inside = labels[y1:y2, x1:x2] == i
            if self._is_ghost(inside, norm[y1:y2, x1:x2], bg[y1:y2, x1:x2]):
                region = cv2.dilate(inside.view(np.uint8), self._ghost_kernel).astype(bool)
                bg[y1:y2, x1:x2][region] = norm[y1:y2, x1:x2][region]
                mask[y1:y2, x1:x2][inside] = 0
                continue
            contrast = float(dist[y1:y2, x1:x2][inside].mean()) / thr
            blobs.append(_Blob(x, y, x + w, y + h, self._score(contrast, area)))
        return [
            Detection((b.x1, b.y1, b.x2, b.y2), b.score, 0, self.label)
            for b in _merge_overlapping(blobs, self.merge_overlap)
        ]

    def _is_ghost(self, inside: BoolArray, frame_roi: Float32Array, bg_roi: Float32Array) -> bool:
        """Whether a blob is a ghost: an object that is in the background, not in the frame.

        A ghost appears where something sat still while the background was
        learned and has since left. Real objects have their contour in the
        *current frame*; a ghost has it in the *background*. Comparing edge
        energy on a band around the blob boundary tells the two apart
        without confusing a resting fish (contour in the frame) for a ghost.
        """
        if self.ghost_edge_ratio <= 0.0:
            return False
        blob = inside.view(np.uint8)
        # Non-empty for any blob that passed the area filter.
        band = cv2.morphologyEx(blob, cv2.MORPH_GRADIENT, self._ghost_kernel).astype(bool)
        frame_edges = _edge_energy(frame_roi, band)
        bg_edges = _edge_energy(bg_roi, band)
        return bg_edges > self.ghost_edge_ratio * max(frame_edges, 1e-6)

    def _score(self, contrast: float, area: float) -> float:
        """Map blob contrast (multiples of the threshold) and area to ``[0, 1]``.

        Geometric mean of two saturating terms: a blob needs *both* a clear
        contrast and a plausible size to score high. A strong fish scores
        ~0.8-1.0 (high-confidence for ByteTrack); a faint, fading or tiny
        blob lands in the low-confidence band used only to keep existing
        tracks alive.
        """
        contrast_term = 1.0 - math.exp(-max(contrast - 1.0, 0.0))
        area_term = 1.0 - math.exp(-area / self.min_area)
        return float(min(1.0, math.sqrt(contrast_term * area_term)))

    def _update_background(self, norm: Float32Array, bg: Float32Array, mask: UInt8Array) -> None:
        protect = (
            mask
            if self._margin_kernel is None
            else np.asarray(cv2.dilate(mask, self._margin_kernel), np.uint8)
        )
        free = np.ascontiguousarray(1 - protect)
        cv2.accumulateWeighted(norm, bg, self.update_rate, mask=free)
        if self.foreground_update_rate > 0.0:
            cv2.accumulateWeighted(norm, bg, self.foreground_update_rate, mask=protect)


def _merge_overlapping(blobs: list[_Blob], overlap: float) -> list[_Blob]:
    """Union boxes whose intersection exceeds ``overlap`` of the smaller box.

    Repeats until stable (merging can create new overlaps). Blob counts are
    tiny (a handful of fish), so the quadratic scan is the simple choice.
    """
    merged = list(blobs)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                iw = min(a.x2, b.x2) - max(a.x1, b.x1)
                ih = min(a.y2, b.y2) - max(a.y1, b.y1)
                if iw <= 0 or ih <= 0:
                    continue
                if iw * ih > overlap * min(a.area, b.area):
                    merged[i] = _Blob(
                        min(a.x1, b.x1),
                        min(a.y1, b.y1),
                        max(a.x2, b.x2),
                        max(a.y2, b.y2),
                        max(a.score, b.score),
                    )
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged
