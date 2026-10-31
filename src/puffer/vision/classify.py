"""Track-level classification: is this track a fish?

The detector judges single frames, and some clutter looks like a fish in
any single frame: a broad leaf swinging into view, a cluster of bubbles, a
bright caustic that happens to be compact. Over a few seconds the
*trajectory* gives it away:

* plants, decorations, reflections of the room, a glint on the glass:
  they **oscillate around a fixed anchor** - a leaf swings back and forth
  but its base stays put;
* bubbles: **tiny blobs that rise straight up**;
* caustics and other light effects: short-lived, erratic, and only
  marginally fish-like blob by blob (low detector scores).

A fish *travels*. :class:`TrackClassifier` therefore accumulates evidence
per track and promotes a track to :attr:`TrackClass.FISH` once

1. **persistence** - it was matched in at least ``min_frames`` frames;
2. **confidence** - the median detection score of its matches is at least
   ``min_score``: a fish is a solid, contrasty, background-occluding blob
   in most frames, clutter that slips past the detector does so marginally;
3. **travel** - its robust position (median of the last few centres) has
   stayed at least ``min_travel_bl`` body lengths away from its anchor
   (median of its first few centres) for ``travel_frames`` consecutive
   frames. Body length is the median longer side of its boxes, so a big
   leaf needs to travel further than a small bubble cluster, and an
   oscillation never accumulates;
4. **not a bubble** - that displacement is not a rise within
   ``bubble_max_angle_deg`` of vertical by a blob smaller than
   ``bubble_area_frac`` of the frame.

**FISH is sticky.** A resting fish does not travel; it is a fish because
it swam there. Once promoted, a track keeps the label for life (including
re-activation after a short occlusion: ByteTrack keeps the id). The
non-fish labels are informational and may change: ``CANDIDATE`` (not
enough evidence yet), ``ANCHORED`` (persistent for ``anchored_frames``
matches without travelling: plant, decoration - or a fish that has rested
since we first saw it, which is promoted the moment it swims off) and
``BUBBLE``.

Consequences, by design: a fish that never moves after start-up is not
credited (we have not seen it swim, so we cannot tell it from an ornament),
and every new track pays a short classification delay (median ~1 s on the
simulator) before its innovations reach the entropy channel. Both are the
price of not crediting plants.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from puffer.types import BBox, Detection, Point

__all__ = ["ClassifiableTrack", "TrackClass", "TrackClassifier", "TrackVerdict"]


class TrackClass(enum.StrEnum):
    """What a track is believed to be."""

    CANDIDATE = "candidate"  # not enough evidence yet
    FISH = "fish"  # travelled like a fish (sticky)
    ANCHORED = "anchored"  # persistent, never travelled: plant, ornament, reflection
    BUBBLE = "bubble"  # small blob rising straight up


class ClassifiableTrack(Protocol):
    """What the classifier needs from a track (satisfied by ``bytetrack.Track``)."""

    @property
    def track_id(self) -> int: ...

    @property
    def time_since_update(self) -> int: ...

    @property
    def center(self) -> Point: ...

    @property
    def detection(self) -> Detection: ...


@dataclass(frozen=True, slots=True)
class TrackVerdict:
    """The current classification of one track and the evidence behind it.

    ``travel_bl`` is the largest displacement from the anchor so far in body
    lengths, ``score`` the median detection score, ``frames`` the number of
    matched frames, ``reason`` a short human-readable explanation (for the
    dashboard and for debugging a misclassification).
    """

    track_id: int
    label: TrackClass
    frames: int
    travel_bl: float
    score: float
    reason: str

    @property
    def is_fish(self) -> bool:
        return self.label is TrackClass.FISH


class _Evidence:
    """Accumulated per-track statistics (bounded memory)."""

    __slots__ = (
        "anchor",
        "areas",
        "away",
        "first",
        "label",
        "last_frame",
        "lengths",
        "matches",
        "max_travel",
        "reason",
        "recent",
        "scores",
    )

    def __init__(self, k: int, memory: int, frame: int) -> None:
        self.matches = 0
        self.first: list[Point] = []
        self.anchor: Point | None = None
        self.recent: deque[Point] = deque(maxlen=k)
        self.lengths: deque[float] = deque(maxlen=memory)
        self.areas: deque[float] = deque(maxlen=memory)
        self.scores: deque[float] = deque(maxlen=memory)
        self.max_travel = 0.0  # body lengths
        self.away = 0  # consecutive frames at least min_travel_bl from the anchor
        self.label = TrackClass.CANDIDATE
        self.reason = "new track"
        self.last_frame = frame


def _median_point(points: Iterable[Point]) -> Point:
    arr = np.asarray(list(points), dtype=np.float64)
    return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))


def _box_dims(box: BBox) -> tuple[float, float]:
    return max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6)


class TrackClassifier:
    """Decide which tracks are fish from their trajectories (see module docstring).

    Parameters
    ----------
    min_frames:
        Matched frames before a track can be promoted (persistence).
    min_score:
        Minimum median detection score (confidence).
    min_travel_bl, travel_frames:
        Displacement from the anchor, in body lengths, that counts as travel,
        and for how many consecutive frames it must hold. Half a body length
        is beyond the sway of a leaf's centre and reached by a cruising fish
        within a second; a fish that swam off stays away, whereas a clutter
        blob whose outline jitters (and with it its centre) only crosses the
        line now and then.
    min_travel_frac:
        Travel must also cover this fraction of the frame width (26 px at
        640): body length is measured on the blob, and a clutter fragment
        has a small "body", so half of it is a few pixels of jitter. Fish
        travel at the scale of the tank.
    position_window:
        Centres in the robust anchor and current-position medians; a few
        frames of median filtering make one-frame jumps (a blob briefly
        merging with a neighbour) irrelevant.
    bubble_max_angle_deg, bubble_area_frac:
        A displacement within this angle of straight up by a blob whose
        median area is below this fraction of the frame is a rising bubble
        (0.003 of the frame: a cluster of bubbles, a fifth of a small fish).
    anchored_frames:
        Matched frames without travel after which a track is called
        ``ANCHORED`` (informational).
    memory:
        Recent matches kept for the medians of length, area and score.
    forget_after:
        Frames after which the evidence of an unseen track is dropped.
    """

    def __init__(
        self,
        min_frames: int = 10,
        min_score: float = 0.6,
        min_travel_bl: float = 0.5,
        *,
        min_travel_frac: float = 0.04,
        travel_frames: int = 5,
        position_window: int = 5,
        bubble_max_angle_deg: float = 30.0,
        bubble_area_frac: float = 0.003,
        anchored_frames: int = 90,
        memory: int = 90,
        forget_after: int = 300,
    ) -> None:
        if min(min_frames, travel_frames, position_window, memory, forget_after) < 1:
            raise ValueError("frame counts must be >= 1")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if min_travel_bl < 0.0 or min_travel_frac < 0.0:
            raise ValueError("min_travel_bl and min_travel_frac must be >= 0")
        if not 0.0 <= bubble_max_angle_deg <= 90.0:
            raise ValueError("bubble_max_angle_deg must be in [0, 90]")
        if not 0.0 <= bubble_area_frac < 1.0:
            raise ValueError("bubble_area_frac must be in [0, 1)")
        self.min_frames = int(min_frames)
        self.min_score = float(min_score)
        self.min_travel_bl = float(min_travel_bl)
        self.min_travel_frac = float(min_travel_frac)
        self.travel_frames = int(travel_frames)
        self.position_window = int(position_window)
        self.bubble_max_angle_deg = float(bubble_max_angle_deg)
        self.bubble_area_frac = float(bubble_area_frac)
        self.anchored_frames = int(anchored_frames)
        self.memory = int(memory)
        self.forget_after = int(forget_after)
        self._evidence: dict[int, _Evidence] = {}
        self._frame = 0

    @classmethod
    def permissive(cls) -> TrackClassifier:
        """A classifier that calls every track a fish from its first match.

        For detectors that only ever report fish (a perfect oracle in tests,
        a model trained on the actual tank) when even the short
        classification delay is unwanted. Bubbles are still rejected.
        """
        return cls(
            min_frames=1,
            min_score=0.0,
            min_travel_bl=0.0,
            min_travel_frac=0.0,
            travel_frames=1,
            position_window=1,
        )

    # ------------------------------------------------------------------ views

    def is_fish(self, track_id: int) -> bool:
        """Whether ``track_id`` has been promoted to fish."""
        ev = self._evidence.get(track_id)
        return ev is not None and ev.label is TrackClass.FISH

    def verdict(self, track_id: int) -> TrackVerdict | None:
        """Current verdict for ``track_id`` (``None`` if never seen or forgotten)."""
        ev = self._evidence.get(track_id)
        if ev is None:
            return None
        score = float(np.median(ev.scores)) if ev.scores else 0.0
        return TrackVerdict(track_id, ev.label, ev.matches, ev.max_travel, score, ev.reason)

    def promote(self, track_id: int, reason: str, min_score: float | None = None) -> None:
        """Declare ``track_id`` a fish on outside evidence (e.g. re-identification).

        Only a track that already meets the persistence and confidence
        requirements can be promoted: outside evidence replaces the travel
        test, nothing else.
        """
        ev = self._evidence.get(track_id)
        if ev is None or ev.label is TrackClass.FISH or not self.qualifies(track_id, min_score):
            return
        ev.label = TrackClass.FISH
        ev.reason = reason

    def demote(self, track_id: int, reason: str) -> None:
        """Withdraw a fish label on outside evidence (the track now follows something else).

        The track starts over as a candidate with no evidence: whatever it
        follows now has to prove itself like any new track.
        """
        if track_id not in self._evidence:
            return
        ev = self._evidence[track_id] = _Evidence(self.position_window, self.memory, self._frame)
        ev.reason = reason

    def qualifies(self, track_id: int, min_score: float | None = None) -> bool:
        """Whether ``track_id`` meets persistence and confidence (travel aside).

        ``min_score`` overrides the classifier's own confidence requirement
        (outside evidence weaker than travel deserves a stricter one).
        """
        ev = self._evidence.get(track_id)
        required = self.min_score if min_score is None else max(self.min_score, min_score)
        return (
            ev is not None
            and ev.matches >= self.min_frames
            and bool(ev.scores)
            and float(np.median(ev.scores)) >= required
        )

    def reset(self) -> None:
        """Forget every track."""
        self._evidence.clear()
        self._frame = 0

    # ----------------------------------------------------------------- update

    def update(
        self, tracks: Sequence[ClassifiableTrack], frame_size: tuple[int, int]
    ) -> dict[int, TrackVerdict]:
        """Fold in one frame; return the verdict of every track in ``tracks``.

        ``tracks`` are the tracker's active tracks; only those matched in this
        frame (``time_since_update == 0``) contribute evidence. ``frame_size``
        is ``(width, height)`` in pixels.
        """
        self._frame += 1
        out: dict[int, TrackVerdict] = {}
        for track in tracks:
            ev = self._evidence.get(track.track_id)
            if ev is None:
                ev = self._evidence[track.track_id] = _Evidence(
                    self.position_window, self.memory, self._frame
                )
            if track.time_since_update == 0:
                ev.last_frame = self._frame
                self._observe(ev, track)
                self._decide(ev, frame_size)
            verdict = self.verdict(track.track_id)
            if verdict is not None:
                out[track.track_id] = verdict
        self._forget()
        return out

    def _observe(self, ev: _Evidence, track: ClassifiableTrack) -> None:
        center = track.center
        w, h = _box_dims(track.detection.bbox)
        ev.matches += 1
        ev.recent.append(center)
        ev.lengths.append(max(w, h))
        ev.areas.append(w * h)
        ev.scores.append(float(track.detection.score))
        if ev.anchor is None:
            ev.first.append(center)
            if len(ev.first) >= self.position_window:
                ev.anchor = _median_point(ev.first)
                ev.first = []

    def _decide(self, ev: _Evidence, frame_size: tuple[int, int]) -> None:
        if ev.label is TrackClass.FISH or ev.anchor is None:
            return  # sticky, or not enough positions for an anchor yet
        frame_area = float(frame_size[0] * frame_size[1])
        body = float(np.median(ev.lengths))
        cx, cy = _median_point(ev.recent)
        dx, dy = cx - ev.anchor[0], cy - ev.anchor[1]
        distance = math.hypot(dx, dy)
        travel = distance / max(body, 1.0)
        ev.max_travel = max(ev.max_travel, travel)
        far = travel >= self.min_travel_bl and distance >= self.min_travel_frac * frame_size[0]
        ev.away = ev.away + 1 if far else 0
        if ev.matches < self.min_frames:
            ev.reason = f"persistence {ev.matches}/{self.min_frames} frames"
            return
        score = float(np.median(ev.scores))
        if score < self.min_score:
            ev.label = TrackClass.CANDIDATE
            ev.reason = f"median score {score:.2f} < {self.min_score:.2f}"
            return
        if ev.away < self.travel_frames:
            moved = ev.max_travel >= self.min_travel_bl  # it went away, then came back
            if ev.matches >= self.anchored_frames and not moved:
                ev.label = TrackClass.ANCHORED
                ev.reason = f"stays within {ev.max_travel:.2f} body lengths of its anchor"
            elif far:
                ev.reason = f"away {ev.away}/{self.travel_frames} frames"
            else:
                ev.reason = f"travel {travel:.2f} < {self.min_travel_bl:.2f} body lengths"
            return
        area = float(np.median(ev.areas))
        off_vertical = math.degrees(math.atan2(abs(dx), -dy))  # 0 = straight up
        if (
            dy < 0.0
            and off_vertical <= self.bubble_max_angle_deg
            and area < self.bubble_area_frac * frame_area
        ):
            ev.label = TrackClass.BUBBLE
            ev.reason = f"small blob rising {off_vertical:.0f} deg off vertical"
            return
        ev.label = TrackClass.FISH
        ev.reason = f"travelled {travel:.2f} body lengths"

    def _forget(self) -> None:
        horizon = self._frame - self.forget_after
        for track_id in [tid for tid, ev in self._evidence.items() if ev.last_frame < horizon]:
            del self._evidence[track_id]

    def forget(self, track_ids: Iterable[int]) -> None:
        """Drop the evidence of tracks that no longer exist (removed by the tracker)."""
        for track_id in track_ids:
            self._evidence.pop(track_id, None)
