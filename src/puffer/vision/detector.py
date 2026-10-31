"""Weight-free fish detector: background subtraction that knows what a fish is not.

Every detection becomes a candidate source of *fish entropy*, so a swaying
plant, a caustic on the sand or a column of bubbles that is reported as a
blob is not a cosmetic glitch. A real aquarium is full of such motion, and
no single background model handles all of it:

* a **median / running-average** background (one mode per pixel) keeps a
  resting fish visible for minutes, but flags every swaying leaf;
* a **mixture model** (OpenCV's MOG2) learns multimodal, periodic
  background - a leaf that alternates with water, a blinking light, the
  bubbler column - within seconds, but it learns a fish that stops just as
  fast, and fish *rest*;
* neither learns light that varies continuously (caustics, surface
  shimmer): that is not a handful of modes, it is a wide distribution.

:class:`MotionDetector` therefore asks several independent questions per
pixel and then per blob.

Pixel level - a pixel is foreground when it is

1. **present**: it differs from a protected background image ``B`` by more
   than a *per-pixel* threshold (below). ``B`` is the per-pixel median of
   the first ``history_frames`` frames and afterwards follows the scene with
   a running average *only where no fish was accepted* (fish pixels update
   at the much slower ``foreground_update_rate``), so a resting fish stays
   present; a decoration moved for good is eventually absorbed;
2. **and** either **moving** - a MOG2 mixture model, run at reduced
   resolution on the lighting-normalised frame, does not explain it - or
   **remembered**: it belonged to an *accepted* detection within the last
   ``hysteresis_frames`` frames. The memory is what keeps a fish that swam
   in and stopped: MOG2 absorbs it within seconds, but the protected
   background does not, and the pixel was a fish a moment ago. A swaying
   leaf never gets that memory - by the time it is present again MOG2 has
   learned it - and memory is only granted to blobs that passed the blob
   tests below.

**Dynamic threshold.** The presence threshold of each pixel is
``max(global, dynamic_k * rms(|I - B|))`` where the rms is a running mean
over roughly ``1 / dynamic_rate`` frames (a plain running mean over the
first seconds, so it is useful early), learned everywhere *except* on what
may be a fish. Where light flickers or ripples, the background itself
strays far from ``B`` and the threshold rises; along a fish's path it does
not, because the fish is excluded from what is learned (otherwise every
fish would raise the threshold of the water it swims through and blind the
detector to the next fish).

**Learning with a tracker.** On its own the detector cannot tell a
resting fish from a leaf that keeps being reported, so it protects every
reported blob from both models - which keeps resting fish, but also keeps
whatever clutter it once reported. :class:`~puffer.vision.FishTracker`
therefore tells it every frame where the fish are (:meth:`protect`): then
only fish (and young blobs strong enough to be a fish not classified yet)
are protected, and flickering light, a revealed patch of swaying leaves or
an ornament that stopped is learned as background within seconds.

The global threshold is ``max(threshold, noise_k * sigma)`` with ``sigma`` a
robust (MAD) estimate of the per-frame sensor noise, so a noisy night camera
does not turn into a carpet of false positives. Lighting drift, flicker and
white-balance wander are global multiplicative changes: every frame is
normalised per channel by the ratio of the background median to the frame
median before anything is compared. Grayscale/IR frames (three identical
channels) are processed as one channel; switching between colour and IR
restarts background learning automatically.

Blob level - connected components (after a morphological opening against
speckle and a closing that fills fins and stripes) are scored by
principled, scale-aware features:

* **size**: area relative to the frame (``min_area_frac``,
  ``max_area_frac``), with ``min_area`` pixels as an absolute floor;
* **contrast**: mean distance to the background in multiples of the
  *local* threshold, so a caustic that barely clears the raised threshold
  of the sand it lies on is faint, however bright it looks;
* **thickness**: the diameter of the largest inscribed disc. Fish have a
  body; plant edges, caustic lines and bubble chains are thin. The
  reference is ``min_thickness_frac`` of the frame width;
* **solidity**: area / convex-hull area. A fish (fins included) is compact;
  caustic networks and fragments of several leaves are not;
* **illumination**: where the background is textured (sand, gravel,
  rocks), a change of *light* leaves that texture visible, an *object*
  hides it. The normalised cross-correlation between the frame and the
  background inside the blob is high for the former, near zero for a fish;
* **ghosts**: something that sat still while the background was learned and
  then left leaves a "ghost" whose contour is in the background image, not
  in the frame; ghosts are absorbed immediately. Limitation: while a slow
  fish still touches its own ghost the two form one blob.

The score is ``base(contrast, area) * thickness * solidity * illumination``
(each factor a ramp in ``[0, 1]``) and plugs straight into ByteTrack's
high/low split: a clear fish scores ~0.8-1.0, a doubtful blob lands in the
low-confidence band that may only *continue* an existing track, and blobs
below ``min_score`` (which ByteTrack would discard anyway) are not reported.
Merged boxes that overlap heavily (a fish split by a dark stripe) are
unioned.

**Scene changes** - when most of the frame is foreground, or the noise
estimate jumps far above its running level, for several consecutive frames
(lights switched, camera bumped), every model is re-learned.

What is left for the tracker (:mod:`puffer.vision.classify`): thick,
compact clutter that moves - a broad leaf swinging into view, a cluster of
bubbles - is recognised by its *trajectory*, which a single frame cannot
show.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from puffer.types import BBox, Detection, Frame

UInt8Array = npt.NDArray[np.uint8]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_MAD_TO_SIGMA = 1.4826  # sigma = 1.4826 * MAD for Gaussian noise
_GAIN_LIMITS = (0.2, 5.0)  # beyond this the scene changed; let re-learning handle it
_GHOST_PAD = 3  # pixels of context around a blob for the ghost edge test
_NOISE_FLOOR = 0.5  # DN; keeps the overload test sane on (synthetic) noise-free video
_NOISE_EMA = 0.05  # adaptation rate of the running noise level
_AGE_NEVER = 255  # hysteresis age of a pixel that was never (or long ago) accepted
_EXCLUDE_MARGIN = 0.25  # dynamic-map exclusion: fraction of the box added on each side
_EXCLUDE_PAD = 4  # ... plus pixels: the fish moves before the next frame
_SOLIDITY_RAMP = (0.35, 0.7)  # solidity factor: 0 below, 1 above
_ILLUMINATION_RAMP = (0.45, 0.85)  # NCC: factor 1 below, 0 above
_TEXTURE_SIGMAS = 2.0  # background must vary this many noise sigmas for the NCC test
_SPLIT_THICKNESS = 1.5  # a split part needs a body this many reference thicknesses thick
_CORE_LEVEL = 0.5  # split cores: distance-transform plateaus above this fraction of the peak
_MEDIAN_REFRESH = 15  # frames between recomputations of the background median (gain)
_REFERENCE_LEVEL = 128.0  # median background level at which `threshold` applies unscaled
_STILL_RATE = 0.1  # running-mean rate of the stillness reference (~1/3 s at 30 fps)
_STILL_SIGMAS = 4.0  # a pixel is still while it stays within this many noise sigmas of it
_STILL_FLOOR = 6.0  # DN; minimum stillness tolerance (noise-free video, compression)
_CANDIDATE_SCORE = 0.6  # ByteTrack's new_track_thresh: a blob this strong may start a fish track
_MSQ_PRIOR_FRAMES = 30  # dynamic map: running mean with this many frames of zero prior
_MEMORY_MOTION = 0.5  # blobs with less motion support than this live on memory ...
_MEMORY_STILL = 0.8  # ... and must be at least this still: resting, not swaying


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


@dataclass(frozen=True, slots=True)
class BlobFeatures:
    """Shape and appearance features of one blob (diagnostics, tuning, tests).

    ``thickness`` is the diameter (pixels) of the largest disc inside the
    blob, ``solidity`` the blob area over its convex-hull area, ``ncc`` the
    frame-vs-background correlation inside the blob (``None`` where the
    background has no texture to correlate), ``contrast`` the mean distance
    to the background in multiples of the global threshold.
    """

    bbox: BBox
    area: int
    thickness: float
    solidity: float
    ncc: float | None
    contrast: float
    score: float


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


def _channel_max(image: Float32Array) -> Float32Array:
    """Per-pixel maximum over channels (identity for one channel).

    ``cv2.reduce`` is ~20x faster than ``ndarray.max(axis=2)``.
    """
    if image.ndim == 2:
        return image
    flat = cv2.reduce(image.reshape(-1, image.shape[2]), 1, cv2.REDUCE_MAX)
    return np.asarray(flat, np.float32).reshape(image.shape[:2])


def _edge_energy(image: Float32Array, band: BoolArray) -> float:
    """Mean Sobel gradient magnitude (L1, max over channels) on ``band`` pixels."""
    gx = np.asarray(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3), np.float32)
    gy = np.asarray(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3), np.float32)
    magnitude = np.abs(gx)
    magnitude += np.abs(gy)
    return float(_channel_max(magnitude)[band].mean())


def _kernel(size: int) -> UInt8Array | None:
    if size <= 1:
        return None
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)), np.uint8)


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below ``lo``, 1 at or above ``hi``, linear in between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def blob_thickness(inside: BoolArray) -> float:
    """Diameter of the largest disc that fits inside a blob mask (pixels)."""
    padded = np.pad(inside.view(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 2.0 * float(np.max(dist))


def blob_solidity(inside: BoolArray) -> float:
    """Blob area over the area of its convex hull, in ``[0, 1]``."""
    contours, _ = cv2.findContours(
        inside.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(np.concatenate(contours))))
    area = float(np.count_nonzero(inside))
    # Pixel-centre contours under-measure tiny blobs; never report > 1.
    return min(1.0, area / hull_area) if hull_area > 0.0 else 1.0


def split_touching(
    blob: BoolArray,
    min_thickness: float,
    max_solidity: float = 0.9,
    *,
    max_parts: int = 3,
    min_part_frac: float = 0.15,
    solidity: float | None = None,
) -> list[tuple[BoolArray, Float32Array]]:
    """Split a blob of fish that touch at a narrow neck; return ``[(part, distance)]``.

    Two fish swimming side by side, blurred and closed by morphology, form
    one blob shaped like a dumbbell: two thick bodies joined by a thin neck.
    The distance transform (distance to the blob edge) has one high plateau
    per body and a low saddle at the neck, so thresholding it at half its
    peak leaves one *core* per body; every pixel then goes to its nearest
    core. A single fish has one core (fins are thin), and a convex blob
    (``solidity >= max_solidity``) is never split.

    Splitting claims "these are two fish", so it needs more evidence than
    detecting one: the blob's thickest body must be at least twice
    ``min_thickness`` thick, and it may break into at most ``max_parts``
    parts, each with at least ``min_part_frac`` of the area and a body at
    least ``min_thickness`` thick. A caustic network or a tangle of leaves
    breaks into many thin, unequal pieces and is left whole (and then
    rejected for its low solidity); on the simulator, relaxing the first
    condition to ``min_thickness`` let enough caustic pieces through to
    double the contaminated fish observations. Fish that *overlap* (one in
    front of the other) have no neck and stay merged - a limitation of any
    blob detector.

    ``distance`` is the distance transform of the whole blob (same shape as
    ``blob``); its maximum within a part is that part's half-thickness.
    ``solidity`` may pass the blob's already computed solidity.
    """
    padded = np.pad(blob.view(np.uint8), 1)
    dist = np.asarray(cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE), np.float32)
    inner = dist[1:-1, 1:-1]
    whole = [(blob, inner)]
    peak = float(np.max(inner))
    if solidity is None:
        solidity = blob_solidity(blob)
    if peak < min_thickness or solidity >= max_solidity:
        return whole
    cores = np.ascontiguousarray(dist >= _CORE_LEVEL * peak, dtype=np.uint8)
    n_cores, _ = cv2.connectedComponents(cores, connectivity=8)
    if not 3 <= n_cores <= max_parts + 1:  # background + 2..max_parts cores
        return whole
    _, nearest = cv2.distanceTransformWithLabels(
        np.ascontiguousarray(1 - cores), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP
    )
    nearest = nearest[1:-1, 1:-1]
    total = float(np.count_nonzero(blob))
    parts = []
    for k in np.unique(nearest[cores[1:-1, 1:-1] > 0]):
        part = blob & (nearest == k)
        if np.count_nonzero(part) < min_part_frac * total:
            return whole
        if 2.0 * float(np.max(inner[part])) < min_thickness:
            return whole
        parts.append((part, inner))
    return parts


def illumination_ncc(
    frame_roi: Float32Array, bg_roi: Float32Array, inside: BoolArray, min_std: float
) -> float | None:
    """Correlation of frame and background luminance inside a blob.

    A change of light (caustic, reflection, shimmer) multiplies or offsets
    the background but leaves its texture in place: correlation near 1. An
    object replaces the texture: correlation near 0. Returns ``None`` when the
    background inside the blob is too flat (std below ``min_std``) for the
    test to mean anything.
    """
    p = frame_roi[inside].astype(np.float64)
    q = bg_roi[inside].astype(np.float64)
    if p.ndim == 2:  # colour: luminance-like channel mean
        p = p.mean(axis=1)
        q = q.mean(axis=1)
    if q.size < 2:
        return None
    p -= p.mean()
    q -= q.mean()
    q_var = float(q @ q) / q.size
    if q_var < min_std * min_std:
        return None
    den = math.sqrt(float(p @ p) / p.size * q_var)
    return float(p @ q) / p.size / den if den > 0.0 else 0.0


class MotionDetector:
    """Dynamic-background-aware motion detector implementing :class:`puffer.types.Detector`.

    Parameters
    ----------
    history_frames:
        Frames collected for the initial median background. No detections
        are produced until it is complete (:attr:`ready`).
    threshold:
        Minimum colour distance (0..255, after normalisation) for a pixel to
        be foreground in a normally lit scene; scaled down in proportion in
        dim scenes (IR night), raised automatically on noisy frames and,
        per pixel, on dynamic background.
    min_area, min_area_frac, max_area_frac:
        Blob area limits: at least ``min_area`` pixels *and* ``min_area_frac``
        of the frame (fish size is relative to the tank, not to the
        sensor: 0.0006 is 184 px at 640x480, a third of the smallest fish
        we measured), at most ``max_area_frac`` of the frame.
    update_rate:
        Running-average rate of the background where no fish was accepted.
    foreground_update_rate:
        Rate under (dilated) accepted detections; keeps resting fish visible.
    noise_k:
        Global threshold in units of the estimated noise sigma.
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
    motion_gate:
        Require MOG2 motion evidence (or hysteresis memory) for foreground.
        ``False`` gives a plain single-mode background subtractor.
    motion_history, motion_var_threshold:
        MOG2 ``history`` (frames; about how fast a new mode is learned) and
        ``varThreshold`` (squared Mahalanobis distance to a mode).
    hysteresis_frames:
        How many frames an accepted pixel is remembered without motion
        evidence; tolerates a fish dropping out of one frame while resting.
    dynamic_k, dynamic_rate:
        Per-pixel threshold ``dynamic_k * rms(|I - B|)`` and the update rate
        of that running mean square (0 disables the dynamic threshold).
    work_scale:
        Resolution (fraction of the frame) of the MOG2 and dynamic-threshold
        maps. They describe regions, not edges, so half resolution loses
        nothing and costs a quarter.
    min_thickness_frac:
        Reference body thickness as a fraction of the frame width. A blob
        thinner than half of it is dropped, full thickness scores fully.
    min_score:
        Blobs scoring at or below this are not reported (ByteTrack's
        ``low_thresh`` default, below which it ignores detections anyway).
    split_solidity:
        Blobs less solid than this are candidates for splitting at narrow
        necks (fish touching side by side, see :func:`split_touching`);
        0 disables splitting.
    protect_frames:
        Once a tracker gives hints (:meth:`protect`), how long a blob outside
        the hinted fish boxes stays protected from background learning.

    Attributes
    ----------
    noise_sigma, last_threshold:
        Diagnostics of the latest frame: robust noise estimate and the
        global decision threshold actually used.
    last_features:
        :class:`BlobFeatures` of every blob scored in the latest frame,
        including rejected ones (score 0).
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
        sample_step: int = 8,
        label: str = "fish",
        motion_gate: bool = True,
        motion_history: int = 500,
        motion_var_threshold: float = 16.0,
        hysteresis_frames: int = 5,
        dynamic_k: float = 3.0,
        dynamic_rate: float = 1.0 / 300.0,
        work_scale: float = 0.5,
        min_thickness_frac: float = 0.015,
        min_area_frac: float = 0.0006,
        min_score: float = 0.1,
        split_solidity: float = 0.9,
        protect_frames: int = 90,
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
        if motion_history < 1:
            raise ValueError("motion_history must be >= 1")
        if not 0 <= hysteresis_frames < _AGE_NEVER:
            raise ValueError(f"hysteresis_frames must be in [0, {_AGE_NEVER})")
        if dynamic_k < 0.0 or not 0.0 < dynamic_rate <= 1.0:
            raise ValueError("require dynamic_k >= 0 and 0 < dynamic_rate <= 1")
        if not 0.0 < work_scale <= 1.0:
            raise ValueError("work_scale must be in (0, 1]")
        if not 0.0 <= min_thickness_frac < 0.5:
            raise ValueError("min_thickness_frac must be in [0, 0.5)")
        if not 0.0 <= min_area_frac < max_area_frac:
            raise ValueError("min_area_frac must be in [0, max_area_frac)")
        if not 1 <= protect_frames < _AGE_NEVER:
            raise ValueError(f"protect_frames must be in [1, {_AGE_NEVER})")
        if not 0.0 <= min_score < 1.0:
            raise ValueError("min_score must be in [0, 1)")
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
        self.motion_gate = bool(motion_gate)
        self.motion_history = int(motion_history)
        self.motion_var_threshold = float(motion_var_threshold)
        self.hysteresis_frames = int(hysteresis_frames)
        self.dynamic_k = float(dynamic_k)
        self.dynamic_rate = float(dynamic_rate)
        self.work_scale = float(work_scale)
        self.min_thickness_frac = float(min_thickness_frac)
        self.min_area_frac = float(min_area_frac)
        self.min_score = float(min_score)
        self.split_solidity = float(split_solidity)
        self.protect_frames = int(protect_frames)
        self._open_kernel = _kernel(open_ksize)
        self._close_kernel = _kernel(close_ksize)
        self._margin_kernel = _kernel(mask_margin)
        self._ghost_kernel = np.asarray(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), np.uint8
        )
        self._motion_kernel = _kernel(5)
        self._exclude_kernel = _kernel(2 * math.ceil(_EXCLUDE_PAD * self.work_scale) + 1)
        self._warmup: list[UInt8Array] = []
        self._bg: Float32Array | None = None
        self._bg_median: npt.NDArray[np.float64] | None = None
        self._median_age = 0
        self._shape: tuple[int, ...] | None = None
        self._fg: BoolArray | None = None
        self._overload = 0
        self._noise_ref: float | None = None
        self._mog: cv2.BackgroundSubtractorMOG2 | None = None
        self._age: UInt8Array | None = None
        self._held: UInt8Array | None = None
        self._hint: list[BBox] | None = None
        self._recent: Float32Array | None = None
        self._still: BoolArray | None = None
        self._msq: Float32Array | None = None
        self._msq_frames = 0
        self._threshold_map: Float32Array | None = None
        self._motion: BoolArray | None = None
        self.last_threshold = self.threshold
        self.noise_sigma = 0.0
        self.last_features: list[BlobFeatures] = []

    # ------------------------------------------------------------- properties

    @property
    def ready(self) -> bool:
        """Whether the initial background has been learned."""
        return self._bg is not None

    @property
    def foreground_mask(self) -> BoolArray | None:
        """Pixels of the latest frame's *reported* detections (``H x W``), or ``None``.

        Exposed so re-identification can restrict colour histograms to fish
        pixels instead of the whole bounding box. Rejected blobs (clutter)
        are not in it.
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

    @property
    def threshold_map(self) -> Float32Array | None:
        """Per-pixel presence threshold of the latest frame (``H x W``), for debugging/UI.

        High where the background itself fluctuates (caustics, shimmer).
        """
        return self._threshold_map

    @property
    def motion_mask(self) -> BoolArray | None:
        """MOG2 motion evidence of the latest frame (``H x W``), or ``None`` when gated off."""
        return self._motion

    def reset(self) -> None:
        """Discard every model and start learning again."""
        self._warmup.clear()
        self._bg = None
        self._bg_median = None
        self._shape = None
        self._fg = None
        self._overload = 0
        self._noise_ref = None
        self._mog = None
        self._age = None
        self._held = None
        self._recent = None
        self._still = None
        self._msq = None
        self._threshold_map = None
        self._motion = None
        self.last_features = []

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
        dist = _channel_max(absdiff)

        step = self.sample_step
        self.noise_sigma = _MAD_TO_SIGMA * float(np.median(absdiff[::step, ::step]))
        thr = max(self._floor(), self.noise_k * self.noise_sigma)
        self.last_threshold = thr
        height, width = dist.shape
        small_size = self._small_size(width, height)

        thr_map = self._presence_threshold(thr, (width, height))
        present = np.greater(dist, thr_map)
        if self.motion_gate:
            evidence = self._motion_evidence(norm, small_size, (width, height))
            if self._age is not None:
                evidence |= self._age <= self.hysteresis_frames
            present &= evidence
        mask = present.view(np.uint8)
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

        detections, accepted, objects, oversized = self._blobs(mask, dist, thr_map, norm, bg)
        # Blobs that could start a track might be fish: never learn them as background dynamics.
        candidates = [d.bbox for d in detections if d.score >= _CANDIDATE_SCORE]
        reported = [d.bbox for d in detections]
        self._update_memory(objects)
        if self._hint is None:  # standalone: every reported blob may be a fish
            self._update_background(norm, bg, accepted)
            self._update_dynamic(dist, reported + oversized, small_size)
        else:
            self._update_with_hint(norm, bg, dist, accepted, candidates, oversized, small_size)
        self._fg = accepted.astype(bool)
        return detections

    # ---------------------------------------------------------------- stages

    def _small_size(self, width: int, height: int) -> tuple[int, int]:
        s = self.work_scale
        return max(1, round(width * s)), max(1, round(height * s))

    def _presence_threshold(self, thr: float, size: tuple[int, int]) -> float | Float32Array:
        """Global threshold, raised per pixel where the background fluctuates."""
        if self._msq is None or self.dynamic_k <= 0.0:
            self._threshold_map = None
            return thr
        small = np.sqrt(self._msq)
        small *= np.float32(self.dynamic_k)
        np.maximum(small, np.float32(thr), out=small)
        full = np.asarray(cv2.resize(small, size, interpolation=cv2.INTER_LINEAR), np.float32)
        self._threshold_map = full
        return full

    def _motion_evidence(
        self, norm: Float32Array, small_size: tuple[int, int], size: tuple[int, int]
    ) -> BoolArray:
        """MOG2 foreground of the lighting-normalised frame, at full resolution.

        Also refreshes the *still* map: pixels whose value has not changed
        from a short-term (about a third of a second) running mean. Memory
        may only keep still things alive (see :meth:`_blobs`).
        """
        small = np.asarray(cv2.resize(norm, small_size, interpolation=cv2.INTER_AREA), np.float32)
        fg = self._mog_model().apply(cv2.convertScaleAbs(small))
        if self._motion_kernel is not None:
            fg = cv2.dilate(fg, self._motion_kernel)
        motion = np.asarray(cv2.resize(fg, size, interpolation=cv2.INTER_NEAREST), np.uint8) > 0
        self._motion = motion
        if self._recent is None:
            self._recent = small.copy()
        change = _channel_max(np.asarray(cv2.absdiff(small, self._recent), np.float32))
        still = np.less(change, max(_STILL_SIGMAS * self.noise_sigma, _STILL_FLOOR)).view(np.uint8)
        self._still = np.asarray(cv2.resize(still, size, interpolation=cv2.INTER_NEAREST), bool)
        cv2.accumulateWeighted(small, self._recent, _STILL_RATE)
        return motion

    def _mog_model(self) -> cv2.BackgroundSubtractorMOG2:
        if self._mog is None:
            self._mog = cv2.createBackgroundSubtractorMOG2(
                history=self.motion_history,
                varThreshold=self.motion_var_threshold,
                detectShadows=False,
            )
        return self._mog

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
        if self.motion_gate:
            height, width = work.shape[:2]
            small = cv2.resize(work, self._small_size(width, height), interpolation=cv2.INTER_AREA)
            self._mog_model().apply(small)
        if len(self._warmup) >= self.history_frames:
            stack = np.stack(self._warmup)
            self._bg = np.median(stack, axis=0).astype(np.float32)
            self._bg_median = None
            self._warmup.clear()

    def _floor(self) -> float:
        """The global threshold floor, scaled down in dim scenes.

        The difference a fish makes is proportional to the light falling
        on it (it reflects a fraction of it, as does the water behind it),
        so a fixed floor tuned for a lit tank would hide every fish of an
        IR night scene whose median level is less than half as bright.
        Above :data:`_REFERENCE_LEVEL` the floor is ``threshold`` as is.
        """
        if self._bg_median is None:
            return self.threshold
        level = float(np.max(self._bg_median))
        return self.threshold * min(1.0, level / _REFERENCE_LEVEL)

    def _gain(self, frame: Float32Array, bg: Float32Array) -> Float32Array:
        step = self.sample_step
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        f_med = np.median(frame[::step, ::step].reshape(-1, channels), axis=0)
        self._median_age += 1
        if self._bg_median is None or self._median_age >= _MEDIAN_REFRESH:
            # The background drifts slowly: refreshing its median now and
            # then saves a full median per frame.
            self._bg_median = np.median(bg[::step, ::step].reshape(-1, channels), axis=0)
            self._median_age = 0
        gain = np.clip(self._bg_median / np.maximum(f_med, 1.0), *_GAIN_LIMITS).astype(np.float32)
        return gain if channels > 1 else gain.reshape(())

    def _blobs(
        self,
        mask: UInt8Array,
        dist: Float32Array,
        thr_map: float | Float32Array,
        norm: Float32Array,
        bg: Float32Array,
    ) -> tuple[list[Detection], UInt8Array, UInt8Array, list[BBox]]:
        """Score connected components; absorb ghosts (mutates ``bg``).

        Contrast is the distance to the background in multiples of the
        *local* threshold ``thr_map``. Returns the detections, the mask of
        the pixels of reported blobs, the mask of every *object* (reported
        blobs plus blobs rejected only for being too large) and the boxes of
        the oversized blobs. Oversized blobs are objects, not fluctuating
        background: they get hysteresis memory (otherwise, once MOG2 absorbs
        a big resting object unevenly, its still-moving remainder shrinks
        into the size window and is reported) and they must not teach the
        dynamic threshold (or a fish close to the glass would carve itself
        into fish-sized fragments).
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = mask.shape
        max_area = self.max_area_frac * height * width
        min_area = max(float(self.min_area), self.min_area_frac * height * width)
        min_std = _TEXTURE_SIGMAS * max(self.noise_sigma, 1.0)
        accepted = np.zeros_like(mask)
        objects = np.zeros_like(mask)
        blobs: list[_Blob] = []
        features: list[BlobFeatures] = []
        oversized: list[BBox] = []
        reference = self.min_thickness_frac * width
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area > max_area:
                oversized.append((float(x), float(y), float(x + w), float(y + h)))
                objects[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i] = 1
                continue
            if area < min_area:
                continue
            blob = labels[y : y + h, x : x + w] == i
            whole_solidity = blob_solidity(blob)
            parts = split_touching(
                blob, _SPLIT_THICKNESS * reference, self.split_solidity, solidity=whole_solidity
            )
            for part, part_dist in parts:
                px, py, pw, ph = cv2.boundingRect(part.view(np.uint8))
                part_area = int(np.count_nonzero(part))
                if part_area < min_area:
                    continue
                gx, gy = x + px, y + py
                x1, y1 = max(0, gx - _GHOST_PAD), max(0, gy - _GHOST_PAD)
                x2, y2 = min(width, gx + pw + _GHOST_PAD), min(height, gy + ph + _GHOST_PAD)
                inside = np.zeros((y2 - y1, x2 - x1), dtype=bool)
                inside[gy - y1 : gy - y1 + ph, gx - x1 : gx - x1 + pw] = part[
                    py : py + ph, px : px + pw
                ]
                box = (float(gx), float(gy), float(gx + pw), float(gy + ph))
                # Cheapest features first: most clutter is rejected before the
                # ghost and illumination tests, which dominate the cost.
                local_thr = thr_map if isinstance(thr_map, float) else thr_map[y1:y2, x1:x2][inside]
                contrast = float(np.mean(dist[y1:y2, x1:x2][inside] / local_thr))
                thickness = 2.0 * float(np.max(part_dist[part]))
                solidity = (
                    whole_solidity
                    if len(parts) == 1
                    else blob_solidity(part[py : py + ph, px : px + pw])
                )
                score = self._score(contrast, part_area, min_area) * self._shape_factor(
                    thickness, solidity, None, width
                )
                if score <= self.min_score:
                    features.append(
                        BlobFeatures(box, part_area, thickness, solidity, None, contrast, 0.0)
                    )
                    continue
                if not self._alive(inside, x1, y1, x2, y2):
                    features.append(
                        BlobFeatures(box, part_area, thickness, solidity, None, contrast, 0.0)
                    )
                    continue
                frame_roi, bg_roi = norm[y1:y2, x1:x2], bg[y1:y2, x1:x2]
                if self._is_ghost(inside, frame_roi, bg_roi):
                    region = cv2.dilate(inside.view(np.uint8), self._ghost_kernel).astype(bool)
                    bg_roi[region] = frame_roi[region]
                    continue
                ncc = illumination_ncc(frame_roi, bg_roi, inside, min_std)
                if ncc is not None:
                    lo, hi = _ILLUMINATION_RAMP
                    score *= 1.0 - _ramp(ncc, lo, hi)
                features.append(
                    BlobFeatures(box, part_area, thickness, solidity, ncc, contrast, score)
                )
                if score <= self.min_score:
                    continue
                accepted[y1:y2, x1:x2][inside] = 1
                objects[y1:y2, x1:x2][inside] = 1
                blobs.append(_Blob(*box, score))
        self.last_features = features
        detections = [
            Detection((b.x1, b.y1, b.x2, b.y2), b.score, 0, self.label)
            for b in _merge_overlapping(blobs, self.merge_overlap)
        ]
        return detections, accepted, objects, oversized

    def _alive(self, inside: BoolArray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Whether a blob is moving, or resting rather than swaying.

        Memory keeps a fish that stopped: MOG2 has absorbed it, yet it is
        still a fish. But memory must not keep a *plant* alive: when a fish
        swims away from the leaves it passed, the protected background there
        is out of date and the leaves (which MOG2 explains as background
        modes) would stay present and remembered forever, and the fish's
        track would latch onto them. The difference is stillness: a resting
        fish does not change from one third of a second to the next, a
        swaying leaf does. A blob with little motion evidence must be still.
        """
        if self._motion is None or self._still is None:
            return True
        n = float(np.count_nonzero(inside))
        moving = float(np.count_nonzero(self._motion[y1:y2, x1:x2][inside])) / n
        if moving >= _MEMORY_MOTION:
            return True
        still = float(np.count_nonzero(self._still[y1:y2, x1:x2][inside])) / n
        return still >= _MEMORY_STILL

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

    def _score(self, contrast: float, area: float, min_area: float | None = None) -> float:
        """Map blob contrast (multiples of the threshold) and area to ``[0, 1]``.

        Geometric mean of two saturating terms: a blob needs *both* a clear
        contrast and a plausible size to score high. A strong fish scores
        ~0.8-1.0 (high-confidence for ByteTrack); a faint, fading or tiny
        blob lands in the low-confidence band used only to keep existing
        tracks alive.
        """
        contrast_term = 1.0 - math.exp(-max(contrast - 1.0, 0.0))
        area_term = 1.0 - math.exp(-area / (self.min_area if min_area is None else min_area))
        return float(min(1.0, math.sqrt(contrast_term * area_term)))

    def _shape_factor(
        self, thickness: float, solidity: float, ncc: float | None, frame_width: int
    ) -> float:
        """Product of the thickness, solidity and illumination ramps, in ``[0, 1]``.

        Each ramp is 1 for every fish we measured on the simulator (5th
        percentile: thickness 17 px at 640 px width, solidity 0.73, NCC
        below 0.45 at the 95th) and falls to 0 where only clutter lives, so
        the factor demotes doubtful blobs to ByteTrack's low band before it
        rejects them outright.
        """
        reference = self.min_thickness_frac * frame_width
        factor = _ramp(thickness, 0.5 * reference, reference)
        factor *= _ramp(solidity, *_SOLIDITY_RAMP)
        if ncc is not None:
            lo, hi = _ILLUMINATION_RAMP
            factor *= 1.0 - _ramp(ncc, lo, hi)
        return factor

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

    def _update_memory(self, objects: UInt8Array) -> None:
        """Age every pixel by one frame; pixels of objects restart at 0.

        Also counts, per pixel, for how many consecutive frames it has
        belonged to an object (``_held``, saturating).
        """
        inside = objects.view(bool)
        if self._held is None:
            self._held = np.zeros(objects.shape, np.uint8)
        np.add(self._held, 1, out=self._held, where=inside & (self._held < _AGE_NEVER))
        self._held[~inside] = 0
        if not self.motion_gate:
            return
        if self._age is None:
            self._age = np.full(objects.shape, _AGE_NEVER, np.uint8)
        else:
            np.add(self._age, 1, out=self._age, where=self._age < _AGE_NEVER)
        self._age[inside] = 0

    def protect(self, boxes: Iterable[BBox]) -> None:
        """Tell the detector where the fish are; applies from the next frame on.

        Without hints (standalone use) every reported blob is protected
        from background *and* dynamic-threshold learning for as long as it
        is reported: the detector cannot tell a resting fish from a leaf or
        a caustic that keeps being reported, and losing a resting fish is
        the worse error. That protection is self-reinforcing, though - a
        protected background goes stale, so the blob stays present and
        keeps being reported. A tracker that *knows* which blobs are fish
        (``FishTracker`` calls this every frame with its fish tracks, an
        empty list included) breaks the loop:

        * background: pixels of reported blobs inside these boxes stay
          protected while they are still (a resting fish); any other blob
          only for ``protect_frames`` consecutive frames - long enough for
          a new fish to be classified - after which it is learned like any
          other background;
        * dynamic threshold: fish pixels and young blobs strong enough to
          start a track (score >= 0.6) are not learned; everything else
          is, so flickering light that keeps producing weak blobs raises
          its own threshold within seconds.
        """
        self._hint = list(boxes)

    def _update_with_hint(
        self,
        norm: Float32Array,
        bg: Float32Array,
        dist: Float32Array,
        accepted: UInt8Array,
        candidates: list[BBox],
        oversized: list[BBox],
        small_size: tuple[int, int],
    ) -> None:
        """Background and dynamic-threshold learning guided by the tracker (see :meth:`protect`)."""
        if self._hint is None or self._held is None:  # pragma: no cover - detect() guarantees both
            return
        fish = np.zeros_like(accepted)
        for box in self._hint:
            _fill_box(fish, box, 1.0)
        young = np.less(self._held, self.protect_frames).view(np.uint8)
        young &= accepted
        settled = fish
        if self._still is not None:
            settled = fish & self._still.view(np.uint8)
        self._update_background(norm, bg, accepted & (young | settled))
        if self.dynamic_k <= 0.0:
            return
        # Not learned as background dynamics: fish, and young blobs that could
        # be fish (a new fish before the tracker has classified it). Clutter
        # that flickers is always young, but scores too low to be a candidate.
        candidate = np.zeros_like(accepted)
        for box in candidates:
            _fill_box(candidate, box, 1.0)
        keep_out = np.asarray(
            cv2.resize(
                (young & candidate) | (fish & accepted),
                small_size,
                interpolation=cv2.INTER_NEAREST,
            ),
            np.uint8,
        )
        if self._exclude_kernel is not None:
            keep_out = np.asarray(cv2.dilate(keep_out, self._exclude_kernel), np.uint8)
        for box in oversized:
            _fill_box(keep_out, box, self.work_scale)
        self._learn_dynamic(dist, keep_out, small_size)

    def _update_dynamic(
        self, dist: Float32Array, objects: list[BBox], small_size: tuple[int, int]
    ) -> None:
        """Learn the per-pixel mean square deviation from the background, except on objects."""
        if self.dynamic_k <= 0.0:
            return
        keep_out = np.zeros((small_size[1], small_size[0]), np.uint8)
        for box in objects:
            _fill_box(keep_out, box, self.work_scale)
        self._learn_dynamic(dist, keep_out, small_size)

    def _learn_dynamic(
        self, dist: Float32Array, keep_out: UInt8Array, small_size: tuple[int, int]
    ) -> None:
        small = np.asarray(cv2.resize(dist, small_size, interpolation=cv2.INTER_AREA), np.float32)
        np.multiply(small, small, out=small)
        if self._msq is None:
            self._msq = np.zeros_like(small)
            self._msq_frames = 0
        learn = np.ascontiguousarray(1 - keep_out)
        # A running mean while young, then the exponential window: the
        # threshold is useful within seconds. The prior (zero deviation for
        # _MSQ_PRIOR_FRAMES frames) keeps one early mistake - a fish not yet
        # excluded - from setting a pixel's threshold on its own.
        self._msq_frames += 1
        rate = max(self.dynamic_rate, 1.0 / (self._msq_frames + _MSQ_PRIOR_FRAMES))
        cv2.accumulateWeighted(small, self._msq, rate, mask=learn)


def _fill_box(mask: UInt8Array, box: BBox, scale: float) -> None:
    """Set a box, grown by the exclusion margin, to 1 in a mask of resolution ``scale``."""
    x1, y1, x2, y2 = box
    mx = _EXCLUDE_MARGIN * (x2 - x1) + _EXCLUDE_PAD
    my = _EXCLUDE_MARGIN * (y2 - y1) + _EXCLUDE_PAD
    cv2.rectangle(
        mask,
        (math.floor((x1 - mx) * scale), math.floor((y1 - my) * scale)),
        (math.ceil((x2 + mx) * scale), math.ceil((y2 + my) * scale)),
        1,
        thickness=-1,
    )


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
