"""ByteTrack multi-object tracker (Zhang et al., "ByteTrack: Multi-Object
Tracking by Associating Every Detection Box", ECCV 2022).

The key idea of ByteTrack is to *not* throw away low-confidence detections:
a fish that is half hidden behind a plant or blurred during a burst still
produces a weak blob, and associating that weak blob with an existing track
(in a second matching stage) keeps the identity alive instead of breaking
the track. One frame of :meth:`ByteTracker.update` is:

1. Split detections into high (``score >= track_thresh``) and low
   (``low_thresh < score < track_thresh``) confidence.
2. Kalman-predict every tracked and lost track.
3. First association: high detections vs. confirmed *and lost* tracks by
   IoU (Hungarian algorithm, cost ``1 - IoU`` optionally fused with the
   detection score), gated by ``match_thresh``. Lost tracks matched here are
   re-activated with their old id.
4. Second association: low detections vs. the remaining *tracked* tracks
   (plain IoU, ``low_match_thresh``). Tracks still unmatched become lost.
5. Tentative tracks vs. the remaining high detections
   (``unconfirmed_match_thresh``); unmatched tentative tracks are deleted.
6. Remaining high detections with ``score >= new_track_thresh`` start new
   tentative tracks; a track is confirmed after ``min_hits`` matched frames.
7. Lost tracks older than the buffer are removed; tracked/lost duplicates
   (IoU > 0.85) are resolved in favour of the older track.

Deviations from the reference implementation, all deliberate:

* Tentative tracks are Kalman-predicted too (the reference only predicts
  activated tracks, so a tentative fish is matched against a stale box).
* Confirmation needs ``min_hits`` hits on every frame, including the first
  one (the reference activates everything on frame 1). The background model
  is still settling on the first frames, so instant confirmation would mint
  spurious confirmed tracks - and those feed the entropy channel.
* Track ids are per-tracker counters, not a process-wide global.
* Every update records the Kalman innovation of the box centre, which is
  the raw material of the fish entropy channel.
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Sequence

import numpy as np

from puffer.types import BBox, Detection, Point
from puffer.vision.geometry import (
    FloatArray,
    bbox_to_xyah,
    boxes_to_array,
    iou_matrix,
    linear_assignment,
    xyah_to_bbox,
)
from puffer.vision.kalman import KalmanFilter

_DUPLICATE_IOU_DISTANCE = 0.15  # ByteTrack's remove_duplicate_stracks threshold


class TrackState(enum.Enum):
    """Life cycle of a track."""

    TENTATIVE = "tentative"  # born, fewer than min_hits matched frames
    CONFIRMED = "confirmed"  # matched in the current frame and confirmed
    LOST = "lost"  # confirmed track that missed detections; coasting on the Kalman model
    REMOVED = "removed"  # terminated; never comes back


class Track:
    """One tracked object: Kalman state plus bookkeeping.

    ``innovation`` is the centre residual (measured minus predicted, pixels)
    of the most recent Kalman update, or ``None`` before the first update -
    a freshly initiated track has no prediction to be surprised by, and a
    fabricated zero residual must never reach the entropy channel.
    ``history`` holds ``(t, cx, cy)`` of the filtered centre after each
    matched frame, oldest first.
    """

    __slots__ = (
        "age",
        "covariance",
        "detection",
        "frame_id",
        "history",
        "hits",
        "innovation",
        "mean",
        "score",
        "start_frame",
        "state",
        "time_since_update",
        "track_id",
    )

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        mean: FloatArray,
        covariance: FloatArray,
        frame_id: int,
        t: float,
        history_len: int,
    ) -> None:
        self.track_id = track_id
        self.mean = mean
        self.covariance = covariance
        self.detection = detection
        self.score = detection.score
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.innovation: Point | None = None
        self.history: deque[tuple[float, float, float]] = deque(maxlen=history_len)
        self.history.append((t, float(mean[0]), float(mean[1])))

    @property
    def bbox(self) -> BBox:
        """Filtered (Kalman) box ``(x1, y1, x2, y2)``."""
        return xyah_to_bbox(self.mean[:4])

    @property
    def center(self) -> Point:
        """Filtered box centre in pixels."""
        return (float(self.mean[0]), float(self.mean[1]))

    @property
    def velocity(self) -> Point:
        """Filtered centre velocity in pixels per *frame*."""
        return (float(self.mean[4]), float(self.mean[5]))

    @property
    def is_confirmed(self) -> bool:
        """Whether the track has passed ``min_hits`` (lost tracks were confirmed)."""
        return self.state in (TrackState.CONFIRMED, TrackState.LOST)

    @property
    def is_tracked(self) -> bool:
        """Whether the track was matched in the most recent frame."""
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"Track(id={self.track_id}, state={self.state.value}, hits={self.hits}, "
            f"centre=({cx:.1f}, {cy:.1f}))"
        )


class ByteTracker:
    """ByteTrack with a constant-velocity Kalman filter.

    Parameters mirror the reference implementation; ``low_match_thresh``
    (0.5) and ``unconfirmed_match_thresh`` (0.7) are the constants it
    hard-codes for the second and third association stages.
    ``track_buffer`` is expressed at 30 fps and rescaled by ``frame_rate``.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        min_hits: int = 3,
        frame_rate: float = 30.0,
        *,
        low_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
        fuse_score: bool = True,
        history_len: int = 64,
        kalman: KalmanFilter | None = None,
    ) -> None:
        if not 0.0 <= low_thresh < track_thresh <= 1.0:
            raise ValueError("require 0 <= low_thresh < track_thresh <= 1")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.unconfirmed_match_thresh = float(unconfirmed_match_thresh)
        self.min_hits = int(min_hits)
        self.frame_rate = float(frame_rate)
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.fuse_score = fuse_score
        self.history_len = int(history_len)
        self.kalman = kalman if kalman is not None else KalmanFilter()
        self.frame_id = 0
        self._next_id = 1
        self._tracked: list[Track] = []
        self._lost: list[Track] = []
        self._last_t: float | None = None
        self._frame_period = 1.0 / self.frame_rate

    # ------------------------------------------------------------------ views

    @property
    def tracked_tracks(self) -> list[Track]:
        """Tracks matched in the latest frame (tentative and confirmed)."""
        return list(self._tracked)

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Confirmed tracks matched in the latest frame."""
        return [t for t in self._tracked if t.state is TrackState.CONFIRMED]

    @property
    def lost_tracks(self) -> list[Track]:
        """Confirmed tracks coasting on their Kalman prediction."""
        return list(self._lost)

    @property
    def frame_period(self) -> float:
        """Smoothed measured seconds per frame (``1 / frame_rate`` until measured).

        The Kalman filter works in pixels per frame; this converts its
        velocities to pixels per second using the *actual* timestamps, so a
        camera that delivers 24 fps instead of the nominal 30 still reports
        correct speeds.
        """
        return self._frame_period

    def reset(self) -> None:
        """Forget every track and restart ids and the frame clock."""
        self.frame_id = 0
        self._next_id = 1
        self._tracked.clear()
        self._lost.clear()
        self._last_t = None
        self._frame_period = 1.0 / self.frame_rate

    # ----------------------------------------------------------------- update

    def update(self, detections: Sequence[Detection], t: float) -> list[Track]:
        """Advance one frame with ``detections`` observed at time ``t`` (seconds).

        Returns the tracks matched in this frame (tentative and confirmed);
        lost tracks are available through :attr:`lost_tracks`.
        """
        self.frame_id += 1
        self._update_clock(t)

        high = [d for d in detections if d.score >= self.track_thresh]
        low = [d for d in detections if self.low_thresh < d.score < self.track_thresh]

        tentative = [tr for tr in self._tracked if tr.state is TrackState.TENTATIVE]
        confirmed = [tr for tr in self._tracked if tr.state is TrackState.CONFIRMED]
        pool = confirmed + self._lost
        self._predict(pool + tentative)

        # 1) high-score detections vs confirmed + lost tracks
        cost = self._iou_cost(pool, high, fuse=self.fuse_score)
        matches, u_pool, u_high = linear_assignment(cost, self.match_thresh)
        for ti, di in matches:
            self._apply_update(pool[ti], high[di], t)

        # 2) low-score detections vs still-unmatched *tracked* tracks
        remaining = [pool[i] for i in u_pool if pool[i].state is TrackState.CONFIRMED]
        cost = self._iou_cost(remaining, low, fuse=False)
        matches, u_remaining, _ = linear_assignment(cost, self.low_match_thresh)
        for ti, di in matches:
            self._apply_update(remaining[ti], low[di], t)
        for i in u_remaining:
            remaining[i].state = TrackState.LOST

        # 3) tentative tracks vs leftover high-score detections
        leftover = [high[i] for i in u_high]
        cost = self._iou_cost(tentative, leftover, fuse=self.fuse_score)
        matches, u_tentative, u_leftover = linear_assignment(cost, self.unconfirmed_match_thresh)
        for ti, di in matches:
            self._apply_update(tentative[ti], leftover[di], t)
        for i in u_tentative:
            tentative[i].state = TrackState.REMOVED

        # 4) births
        births: list[Track] = []
        for i in u_leftover:
            det = leftover[i]
            if det.score >= self.new_track_thresh:
                births.append(self._birth(det, t))

        # 5) lost-track buffer
        for tr in pool:
            if tr.state is TrackState.LOST and self.frame_id - tr.frame_id > self.max_time_lost:
                tr.state = TrackState.REMOVED

        everything = pool + tentative + births
        self._tracked = [tr for tr in everything if tr.is_tracked]
        self._lost = [tr for tr in everything if tr.state is TrackState.LOST]
        self._remove_duplicates()
        return list(self._tracked)

    # ---------------------------------------------------------------- helpers

    def _update_clock(self, t: float) -> None:
        if self._last_t is not None:
            dt = t - self._last_t
            # Ignore stalls and clock jumps: they say nothing about the frame rate.
            if 0.0 < dt < 5.0 * self._frame_period:
                self._frame_period += 0.1 * (dt - self._frame_period)
        self._last_t = t

    def _predict(self, tracks: list[Track]) -> None:
        if not tracks:
            return
        means = np.stack([tr.mean for tr in tracks])
        covs = np.stack([tr.covariance for tr in tracks])
        for i, tr in enumerate(tracks):
            if tr.state is TrackState.LOST:
                means[i, 7] = 0.0  # ByteTrack: a coasting box stops changing size
        means, covs = self.kalman.multi_predict(means, covs)
        for i, tr in enumerate(tracks):
            tr.mean = means[i]
            tr.covariance = covs[i]
            tr.age += 1
            tr.time_since_update += 1

    @staticmethod
    def _iou_cost(tracks: list[Track], dets: list[Detection], *, fuse: bool) -> FloatArray:
        track_boxes = boxes_to_array([tr.bbox for tr in tracks])
        det_boxes = boxes_to_array([d.bbox for d in dets])
        iou = iou_matrix(track_boxes, det_boxes)
        if fuse and iou.size:
            scores = np.array([d.score for d in dets], dtype=np.float64)
            iou = iou * scores[None, :]
        cost: FloatArray = 1.0 - iou
        return cost

    def _apply_update(self, track: Track, det: Detection, t: float) -> None:
        upd = self.kalman.update(track.mean, track.covariance, bbox_to_xyah(det.bbox))
        track.mean = upd.mean
        track.covariance = upd.covariance
        track.innovation = (float(upd.innovation[0]), float(upd.innovation[1]))
        track.detection = det
        track.score = det.score
        track.hits += 1
        track.time_since_update = 0
        track.frame_id = self.frame_id
        if track.state is TrackState.LOST or (
            track.state is TrackState.TENTATIVE and track.hits >= self.min_hits
        ):
            track.state = TrackState.CONFIRMED
        track.history.append((t, float(track.mean[0]), float(track.mean[1])))

    def _birth(self, det: Detection, t: float) -> Track:
        mean, cov = self.kalman.initiate(bbox_to_xyah(det.bbox))
        track = Track(self._next_id, det, mean, cov, self.frame_id, t, self.history_len)
        self._next_id += 1
        if self.min_hits <= 1:
            track.state = TrackState.CONFIRMED
        return track

    def _remove_duplicates(self) -> None:
        """Drop the younger of any tracked/lost pair that overlaps almost entirely."""
        if not self._tracked or not self._lost:
            return
        dist = 1.0 - iou_matrix(
            boxes_to_array([tr.bbox for tr in self._tracked]),
            boxes_to_array([tr.bbox for tr in self._lost]),
        )
        drop: set[int] = set()
        for p, q in zip(*np.nonzero(dist < _DUPLICATE_IOU_DISTANCE), strict=True):
            a, b = self._tracked[int(p)], self._lost[int(q)]
            older_is_tracked = (a.frame_id - a.start_frame) > (b.frame_id - b.start_frame)
            victim = b if older_is_tracked else a
            victim.state = TrackState.REMOVED
            drop.add(id(victim))
        self._tracked = [tr for tr in self._tracked if id(tr) not in drop]
        self._lost = [tr for tr in self._lost if id(tr) not in drop]
