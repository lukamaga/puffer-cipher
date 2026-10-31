"""End-to-end fish tracking: detector -> ByteTrack -> re-ID -> motion features.

:class:`FishTracker` is the single entry point the engine uses. One call to
:meth:`FishTracker.process` per frame returns a :class:`TrackingResult`:

* ``detections`` - raw detector output (for debugging overlays);
* ``tracks`` - a :class:`TrackSnapshot` per track matched this frame
  (tentative ones included, flagged ``confirmed=False``) with the stable
  ``fish_id`` from re-identification and a short trail for the UI;
* ``observations`` - a :class:`FishObservation` per *confirmed* track that
  received a measurement update this frame. Only these reach the fish
  entropy channel: tentative tracks may be noise, and coasting (lost)
  tracks have no measurement, hence no genuine innovation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from puffer.types import Detector, FishObservation, Frame, TrackingResult, TrackSnapshot
from puffer.vision.bytetrack import ByteTracker, Track
from puffer.vision.detector import MotionDetector
from puffer.vision.features import FishCharacter, MotionFeatures, body_length
from puffer.vision.reid import FishIdentity, FishReID


@runtime_checkable
class ForegroundProvider(Protocol):
    """A detector that can share its per-pixel foreground mask (e.g. :class:`MotionDetector`)."""

    @property
    def foreground_mask(self) -> npt.NDArray[np.bool_] | None: ...


class FishTracker:
    """Compose detection, tracking, re-identification and motion features.

    Parameters
    ----------
    detector:
        Any :class:`puffer.types.Detector`; default :class:`MotionDetector`.
    tracker:
        Default :class:`ByteTracker` with enough history for ``trail_len``.
    reid:
        Default :class:`FishReID` with built-in names.
    trail_len:
        Number of recent centres in each snapshot's ``trail``.
    """

    def __init__(
        self,
        detector: Detector | None = None,
        tracker: ByteTracker | None = None,
        reid: FishReID | None = None,
        trail_len: int = 30,
    ) -> None:
        if trail_len < 0:
            raise ValueError("trail_len must be >= 0")
        self.detector: Detector = detector if detector is not None else MotionDetector()
        self.tracker = (
            tracker if tracker is not None else ByteTracker(history_len=max(64, trail_len))
        )
        self.reid = reid if reid is not None else FishReID()
        self.trail_len = int(trail_len)
        self._motion: dict[int, MotionFeatures] = {}
        self._characters: dict[str, FishCharacter] = {}

    @property
    def characters(self) -> dict[str, FishCharacter]:
        """Behaviour statistics by ``fish_id`` (a new dict)."""
        return dict(self._characters)

    @property
    def identities(self) -> dict[str, FishIdentity]:
        """Re-identification gallery by ``fish_id``."""
        return self.reid.identities

    def process(self, frame: Frame) -> TrackingResult:
        """Run the whole vision pipeline on one frame."""
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame.timestamp)
        foreground = (
            self.detector.foreground_mask if isinstance(self.detector, ForegroundProvider) else None
        )
        fish_ids = self.reid.update(frame.image, tracks, foreground)
        period = self.tracker.frame_period

        snapshots: list[TrackSnapshot] = []
        observations: list[FishObservation] = []
        for track in tracks:
            fish_id = fish_ids.get(track.track_id)
            snapshots.append(self._snapshot(track, fish_id, period))
            if track.is_confirmed and track.time_since_update == 0 and track.innovation is not None:
                observation = self._observe(track, frame.timestamp, period, fish_id)
                observations.append(observation)
        self._prune()
        return TrackingResult(
            frame_index=frame.index,
            t=frame.timestamp,
            detections=tuple(detections),
            tracks=tuple(snapshots),
            observations=tuple(observations),
        )

    def _snapshot(self, track: Track, fish_id: str | None, period: float) -> TrackSnapshot:
        vx, vy = track.velocity
        history = list(track.history)[-self.trail_len :] if self.trail_len else []
        return TrackSnapshot(
            track_id=track.track_id,
            fish_id=fish_id,
            bbox=track.bbox,
            score=track.score,
            center=track.center,
            velocity=(vx / period, vy / period),
            trail=tuple((x, y) for _, x, y in history),
            confirmed=track.is_confirmed,
            age_frames=track.age,
        )

    def _observe(
        self, track: Track, t: float, period: float, fish_id: str | None
    ) -> FishObservation:
        motion = self._motion.get(track.track_id)
        if motion is None:
            motion = self._motion[track.track_id] = MotionFeatures(track.track_id)
        observation = motion.observe(track, t, period, fish_id)
        if fish_id is not None:
            character = self._characters.get(fish_id)
            if character is None:
                character = self._characters[fish_id] = FishCharacter(fish_id)
            character.update(observation, body_length(track.bbox))
        return observation

    def _prune(self) -> None:
        alive = {tr.track_id for tr in self.tracker.tracked_tracks}
        alive.update(tr.track_id for tr in self.tracker.lost_tracks)
        for track_id in [tid for tid in self._motion if tid not in alive]:
            del self._motion[track_id]
