"""Measure fish-tracking quality against ground truth (simulator or labelled video).

Why this module exists: every observation the tracker emits is credited as
*fish* entropy and attributed to a named fish in receipts and the beacon
summary. A swaying plant, a caustic on the sand or a bubble that becomes a
"fish" would make both claims false. "Looks fine in the demo" is not a
quality bar; these numbers are, and ``scripts/eval_tracking.py`` publishes
them (seeded, reproducible) in the README.

Protocol
    The source is read frame by frame and the tracker processes every frame;
    the first ``warmup`` frames (background learning, track confirmation)
    are processed but not scored. For each scored frame:

    * **Detections** are matched one-to-one to ground-truth boxes with the
      Hungarian algorithm on ``1 - IoU`` (pairs below ``iou_threshold`` are
      forbidden) -> detection precision and recall.
    * **Fish tracks** are the snapshots with ``confirmed=True``, i.e. tracks
      the tracker *classified as fish* (including fish coasting through a
      short occlusion at their predicted box). They are matched to ground
      truth the same way -> fish recall (fraction of ground-truth
      fish-frames covered by a fish track), fish-track precision, centre
      error, CLEAR-MOT identity switches (a ground-truth fish matched to a
      different track id than last time), and re-identification statistics
      on ``fish_id``. *Observed recall* counts only matches whose track
      also produced an observation this frame: the fraction of fish-frames
      that actually fed the entropy channel.
    * **Observations** (what the fish entropy channel receives) are
      *contaminated* when the observing track's box does not lie mostly on a
      real fish: less than ``coverage_threshold`` of its area intersects any
      ground-truth box. Coverage rather than IoU so that a fish half hidden
      behind a plant, whose box is a fragment of the true box, still counts
      as a fish, while a plant blob that merely brushes a fish does not.

Identity metrics (``fish_id`` vs ground-truth fish)
    *Purity*: of all matched fish-track-frames carrying a ``fish_id``, the
    fraction whose ``fish_id`` belongs to the ground-truth fish that
    identity is most often seen on - "when the receipt says Nemo, it was
    Nemo". *Coverage*: the fraction of ground-truth fish-frames that carry
    the identity globally assigned to that fish (one-to-one Hungarian
    assignment of identities to fish by co-occurrence, as in IDF1);
    misses, unidentified tracks (``fish_id=None``, e.g. IR frames without
    colour) and split identities all lower it.

Limitations, stated rather than hidden: IoU matching treats a correct but
fragmentary box (fish behind a front plant) as a miss plus a false
detection, and the simulator's ground truth includes fish that are fully
hidden behind front plants. Both make the reported recall conservative.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from puffer.types import BBox, Frame, TrackingResult
from puffer.vision.geometry import FloatArray, boxes_to_array, iou_matrix, linear_assignment

__all__ = [
    "GroundTruthSource",
    "TrackerLike",
    "TrackingMetrics",
    "TruthBox",
    "box_coverage",
    "evaluate_tracking",
    "match_boxes",
]


@runtime_checkable
class TruthBox(Protocol):
    """One ground-truth fish in one frame (``puffer.sources.FishTruth`` fits)."""

    @property
    def fish_id(self) -> str: ...

    @property
    def bbox(self) -> BBox: ...


class GroundTruthSource(Protocol):
    """A frame source that knows where the fish are (``SyntheticAquarium`` fits)."""

    def read(self) -> Frame | None: ...

    def ground_truth(self) -> Sequence[TruthBox]:
        """Fish visible in the frame most recently returned by :meth:`read`."""
        ...


class TrackerLike(Protocol):
    """Anything with ``process(frame) -> TrackingResult`` (``FishTracker`` fits)."""

    def process(self, frame: Frame) -> TrackingResult: ...


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    """Tracking quality over the scored frames of one run (or several, :meth:`pool`).

    Fields are raw counts, so runs over several seeds can be pooled
    exactly; the quality figures are derived properties. Ratios are in
    ``[0, 1]`` and are ``nan`` when undefined (e.g. precision with no
    detections at all), never silently 0 or 1.
    """

    frames: int
    seconds: float  # simulated time covered by the scored frames
    gt_fish: int  # distinct ground-truth fish seen (summed over pooled runs)
    gt_fish_frames: int
    detections: int
    detection_matches: int
    fish_track_frames: int
    fish_matches: int
    observed_matches: int
    observations: int
    contaminated_observations: int
    id_switches: int
    fish_id_switches: int
    identities: int  # distinct fish_ids on fish tracks (summed over pooled runs)
    identified_frames: int  # fish matches carrying a fish_id
    pure_frames: int  # ... whose fish_id is on its majority ground-truth fish
    assigned_frames: int  # ground-truth fish-frames carrying their assigned identity
    center_error_sum: float
    processed_frames: int
    process_seconds: float
    iou_threshold: float
    runs: int = 1

    # ------------------------------------------------------------- figures

    @property
    def detection_precision(self) -> float:
        return _ratio(self.detection_matches, self.detections)

    @property
    def detection_recall(self) -> float:
        return _ratio(self.detection_matches, self.gt_fish_frames)

    @property
    def fish_track_precision(self) -> float:
        """Fraction of fish-track frames that lie on a real fish."""
        return _ratio(self.fish_matches, self.fish_track_frames)

    @property
    def fish_recall(self) -> float:
        """Fraction of ground-truth fish-frames covered by a confirmed fish track."""
        return _ratio(self.fish_matches, self.gt_fish_frames)

    @property
    def observed_recall(self) -> float:
        """Fraction of ground-truth fish-frames that fed the entropy channel."""
        return _ratio(self.observed_matches, self.gt_fish_frames)

    @property
    def contamination(self) -> float:
        """Fraction of observations (entropy samples) not on a real fish."""
        return _ratio(self.contaminated_observations, self.observations)

    @property
    def identity_purity(self) -> float:
        return _ratio(self.pure_frames, self.identified_frames)

    @property
    def identity_coverage(self) -> float:
        return _ratio(self.assigned_frames, self.gt_fish_frames)

    @property
    def mean_center_error_px(self) -> float:
        return _ratio(self.center_error_sum, self.fish_matches)

    @property
    def id_switches_per_minute(self) -> float:
        return _ratio(60.0 * self.id_switches, self.seconds)

    @property
    def fish_id_switches_per_minute(self) -> float:
        return _ratio(60.0 * self.fish_id_switches, self.seconds)

    @property
    def process_fps(self) -> float:
        """Frames per second of ``tracker.process`` alone (frame rendering excluded)."""
        return _ratio(self.processed_frames, self.process_seconds)

    @property
    def detections_per_frame(self) -> float:
        return _ratio(self.detections, self.frames)

    @property
    def observations_per_frame(self) -> float:
        return _ratio(self.observations, self.frames)

    @property
    def fish_per_frame(self) -> float:
        """Mean number of ground-truth fish per scored frame."""
        return _ratio(self.gt_fish_frames, self.frames)

    _FIGURES = (
        "detection_precision",
        "detection_recall",
        "fish_track_precision",
        "fish_recall",
        "observed_recall",
        "contamination",
        "identity_purity",
        "identity_coverage",
        "mean_center_error_px",
        "id_switches_per_minute",
        "fish_id_switches_per_minute",
        "process_fps",
        "detections_per_frame",
        "observations_per_frame",
        "fish_per_frame",
    )

    def to_dict(self) -> dict[str, float | int]:
        """Counts and derived figures, JSON-ready (``nan`` stays ``nan``)."""
        out: dict[str, float | int] = asdict(self)
        for name in self._FIGURES:
            out[name] = getattr(self, name)
        return out

    @classmethod
    def pool(cls, runs: Sequence[TrackingMetrics]) -> TrackingMetrics:
        """Combine runs (e.g. seeds) by summing counts: every figure is then pooled exactly."""
        if not runs:
            raise ValueError("nothing to pool")
        if len({r.iou_threshold for r in runs}) != 1:
            raise ValueError("runs used different IoU thresholds")
        sums = {
            f.name: sum(getattr(r, f.name) for r in runs)
            for f in fields(cls)
            if f.name != "iou_threshold"
        }
        return cls(iou_threshold=runs[0].iou_threshold, **sums)


def match_boxes(
    pred: Sequence[BBox], truth: Sequence[BBox], iou_threshold: float
) -> list[tuple[int, int]]:
    """One-to-one ``(pred_index, truth_index)`` pairs maximising total IoU.

    Hungarian assignment on ``1 - IoU``; pairs with IoU below
    ``iou_threshold`` can never be matched (see
    :func:`puffer.vision.geometry.linear_assignment` for why gating inside
    the solver matters).
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not pred or not truth:
        return []
    cost = 1.0 - iou_matrix(boxes_to_array(pred), boxes_to_array(truth))
    matches, _, _ = linear_assignment(cost, 1.0 - iou_threshold)
    return matches


def _union_area(rects: FloatArray) -> float:
    """Exact area of the union of ``(N, 4)`` rectangles (coordinate compression).

    N is a handful of fish, so the ``O(N^3)`` grid is cheaper than anything clever.
    """
    if rects.shape[0] == 0:
        return 0.0
    xs = np.unique(rects[:, [0, 2]])
    ys = np.unique(rects[:, [1, 3]])
    cx = (xs[:-1] + xs[1:]) / 2.0
    cy = (ys[:-1] + ys[1:]) / 2.0
    inside = (
        (rects[:, 0, None, None] <= cx[None, None, :])
        & (cx[None, None, :] < rects[:, 2, None, None])
        & (rects[:, 1, None, None] <= cy[None, :, None])
        & (cy[None, :, None] < rects[:, 3, None, None])
    ).any(axis=0)
    cell = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return float((cell * inside).sum())


def box_coverage(pred: Sequence[BBox], truth: Sequence[BBox]) -> FloatArray:
    """For each predicted box, the fraction of its area covered by the union of truth boxes.

    The union (not the best single box) so that one box around two
    overlapping fish counts as being on fish.
    """
    p = boxes_to_array(pred)
    t = boxes_to_array(truth)
    out: FloatArray = np.zeros(p.shape[0], dtype=np.float64)
    for i, (x1, y1, x2, y2) in enumerate(p):
        area = (x2 - x1) * (y2 - y1)
        if area <= 0.0 or t.shape[0] == 0:
            continue
        clipped = np.column_stack(
            [
                np.maximum(t[:, 0], x1),
                np.maximum(t[:, 1], y1),
                np.minimum(t[:, 2], x2),
                np.minimum(t[:, 3], y2),
            ]
        )
        clipped = clipped[(clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])]
        out[i] = _union_area(clipped) / area
    return out


def _ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def _centre(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class _Accumulator:
    """Per-frame bookkeeping for :func:`evaluate_tracking`."""

    def __init__(self, iou_threshold: float, coverage_threshold: float) -> None:
        self.iou = iou_threshold
        self.coverage = coverage_threshold
        self.frames = 0
        self.first_t: float | None = None
        self.last_t = 0.0
        self.gt_ids: set[str] = set()
        self.gt_frames = 0
        self.detections = 0
        self.det_tp = 0
        self.fish_track_frames = 0
        self.fish_tp = 0
        self.observed_tp = 0
        self.observations = 0
        self.contaminated = 0
        self.id_switches = 0
        self.fish_id_switches = 0
        self.centre_error_sum = 0.0
        self.last_track: dict[str, int] = {}
        self.last_fish_id: dict[str, str] = {}
        self.cooccurrence: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.fish_ids: set[str] = set()

    def add(self, result: TrackingResult, truth: Sequence[TruthBox]) -> None:
        self.frames += 1
        if self.first_t is None:
            self.first_t = result.t
        self.last_t = result.t
        self.gt_frames += len(truth)
        self.gt_ids.update(g.fish_id for g in truth)
        gt_boxes = [g.bbox for g in truth]

        det_boxes = [d.bbox for d in result.detections]
        self.detections += len(det_boxes)
        self.det_tp += len(match_boxes(det_boxes, gt_boxes, self.iou))

        fish = [s for s in result.tracks if s.confirmed]
        self.fish_track_frames += len(fish)
        self.fish_ids.update(s.fish_id for s in fish if s.fish_id is not None)
        observed = {o.track_id for o in result.observations}
        for ti, gi in match_boxes([s.bbox for s in fish], gt_boxes, self.iou):
            snap, gt = fish[ti], truth[gi]
            self.fish_tp += 1
            self.observed_tp += snap.track_id in observed
            (tx, ty), (gx, gy) = _centre(snap.bbox), _centre(gt.bbox)
            self.centre_error_sum += math.hypot(tx - gx, ty - gy)
            previous = self.last_track.get(gt.fish_id)
            if previous is not None and previous != snap.track_id:
                self.id_switches += 1
            self.last_track[gt.fish_id] = snap.track_id
            if snap.fish_id is not None:
                self.cooccurrence[snap.fish_id][gt.fish_id] += 1
                previous_id = self.last_fish_id.get(gt.fish_id)
                if previous_id is not None and previous_id != snap.fish_id:
                    self.fish_id_switches += 1
                self.last_fish_id[gt.fish_id] = snap.fish_id

        boxes_by_track = {s.track_id: s.bbox for s in result.tracks}
        self.observations += len(result.observations)
        obs_boxes = [
            boxes_by_track.get(o.track_id, (o.cx - 0.5, o.cy - 0.5, o.cx + 0.5, o.cy + 0.5))
            for o in result.observations
        ]
        coverage = box_coverage(obs_boxes, gt_boxes)
        self.contaminated += int(np.count_nonzero(coverage < self.coverage))

    def identity_counts(self) -> tuple[int, int, int]:
        """``(identified, pure, assigned)`` frame counts of re-identification."""
        identified = sum(sum(c.values()) for c in self.cooccurrence.values())
        pure = sum(max(c.values()) for c in self.cooccurrence.values())
        ids = sorted(self.cooccurrence)
        gts = sorted({g for c in self.cooccurrence.values() for g in c})
        assigned = 0
        if ids and gts:
            counts: npt.NDArray[np.float64] = np.array(
                [[self.cooccurrence[f][g] for g in gts] for f in ids], dtype=np.float64
            )
            # Maximise co-occurrence: Hungarian on (max - count), no gating.
            cost = np.max(counts) - counts
            matches, _, _ = linear_assignment(cost, float(np.max(cost)) + 1.0)
            assigned = int(sum(counts[i, j] for i, j in matches))
        return identified, pure, assigned

    def seconds(self) -> float:
        """Time span of the scored frames, counting the last frame's period."""
        if self.first_t is None or self.frames < 2:
            return 0.0
        span = self.last_t - self.first_t
        return span * self.frames / (self.frames - 1)


def evaluate_tracking(
    source: GroundTruthSource,
    tracker: TrackerLike,
    n_frames: int,
    *,
    warmup: int = 150,
    iou_threshold: float = 0.3,
    coverage_threshold: float = 0.5,
    on_frame: Callable[[Frame, TrackingResult, Sequence[TruthBox]], None] | None = None,
) -> TrackingMetrics:
    """Run ``tracker`` over ``n_frames`` of ``source`` and score frames from ``warmup`` on.

    ``iou_threshold`` gates box matching (0.3: motion-blob boxes include
    fins and halo, so 0.5 would score geometry rather than "is this the
    fish"); ``coverage_threshold`` decides contamination (module docstring).
    ``process_fps`` times ``tracker.process`` alone, not frame rendering.
    ``on_frame`` sees every frame (scored or not), e.g. for video export.
    Stops early if the source ends.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if n_frames <= warmup:
        raise ValueError("n_frames must exceed warmup")
    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1]")
    acc = _Accumulator(iou_threshold, coverage_threshold)
    busy = 0.0
    processed = 0
    for i in range(n_frames):
        frame = source.read()
        if frame is None:
            break
        truth = list(source.ground_truth())
        start = time.perf_counter()
        result = tracker.process(frame)
        busy += time.perf_counter() - start
        processed += 1
        if on_frame is not None:
            on_frame(frame, result, truth)
        if i >= warmup:
            acc.add(result, truth)

    identified, pure, assigned = acc.identity_counts()
    return TrackingMetrics(
        frames=acc.frames,
        seconds=acc.seconds(),
        gt_fish=len(acc.gt_ids),
        gt_fish_frames=acc.gt_frames,
        detections=acc.detections,
        detection_matches=acc.det_tp,
        fish_track_frames=acc.fish_track_frames,
        fish_matches=acc.fish_tp,
        observed_matches=acc.observed_tp,
        observations=acc.observations,
        contaminated_observations=acc.contaminated,
        id_switches=acc.id_switches,
        fish_id_switches=acc.fish_id_switches,
        identities=len(acc.fish_ids),
        identified_frames=identified,
        pure_frames=pure,
        assigned_frames=assigned,
        center_error_sum=acc.centre_error_sum,
        processed_frames=processed,
        process_seconds=busy,
        iou_threshold=iou_threshold,
    )
