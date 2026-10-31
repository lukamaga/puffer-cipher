"""End-to-end fish tracking: detector -> ByteTrack -> classification -> re-ID -> features.

:class:`FishTracker` is the single entry point the engine uses. One call to
:meth:`FishTracker.process` per frame returns a :class:`TrackingResult`:

* ``detections`` - raw detector output (for debugging overlays);
* ``tracks`` - a :class:`TrackSnapshot` per track matched this frame with
  the stable ``fish_id`` from re-identification and a short trail for the
  UI. ``confirmed`` is ``True`` only for tracks the
  :class:`~puffer.vision.classify.TrackClassifier` has classified as
  **fish**; tentative tracks and candidates that have not (yet) travelled
  like a fish - a swaying plant, a bubble - are reported with
  ``confirmed=False`` so a dashboard can still show them. Fish tracks that
  briefly lost their detection (one fish passing behind another: a blob
  detector sees one blob) are reported for up to ``coast_frames`` frames
  at their predicted position with ``score=0.0``;
* ``observations`` - a :class:`FishObservation` per *fish* track that
  received a measurement update this frame. Only these reach the fish
  entropy channel: a candidate may be a plant, and coasting (lost) tracks
  have no measurement, hence no genuine innovation.

Re-identification sees every matched track (so a candidate's appearance
model is ready the moment it is promoted) but may only bind and enrol
identities for fish tracks: a plant never gets a name. When the detector
can share its foreground mask (:class:`MotionDetector`), colour histograms
are restricted to the pixels of reported blobs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from puffer.types import (
    BBox,
    Detection,
    Detector,
    FishObservation,
    Frame,
    TrackingResult,
    TrackSnapshot,
)
from puffer.vision.bytetrack import ByteTracker, Track
from puffer.vision.classify import TrackClass, TrackClassifier, TrackVerdict
from puffer.vision.detector import MotionDetector
from puffer.vision.features import FishCharacter, MotionFeatures, body_length
from puffer.vision.reid import FishIdentity, FishReID, bhattacharyya_distance

_REFERENCE_FOLLOW = 0.3
"""Appearance distance below which a fish track's reference histogram adapts."""
_REFERENCE_RATE = 0.02
"""Adaptation rate of that reference (slow: a hijack must not drag it along)."""


def _blend(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32], rate: float
) -> npt.NDArray[np.float32]:
    out: npt.NDArray[np.float32] = ((1.0 - rate) * a + rate * b).astype(np.float32)
    return out


_RECOGNISE_SCORE = 0.85
"""Median detection score a candidate needs to be promoted by re-identification.

Recognition replaces the travel test, so it demands a blob that is
unmistakably solid: on the simulator no clutter track scored this high
while resembling a missing fish, whereas caustics on sandy-coloured fish
did at 0.6-0.7."""


@runtime_checkable
class ForegroundProvider(Protocol):
    """A detector that can share its per-pixel foreground mask (e.g. :class:`MotionDetector`)."""

    @property
    def foreground_mask(self) -> npt.NDArray[np.bool_] | None: ...


@runtime_checkable
class FishHintReceiver(Protocol):
    """A detector that wants to know where the fish are (e.g. :class:`MotionDetector`)."""

    def protect(self, boxes: Iterable[BBox]) -> None: ...


@dataclass(frozen=True, slots=True)
class _ReIDView:
    """A track as re-identification sees it: ``is_confirmed`` means *classified fish*."""

    track_id: int
    is_confirmed: bool
    time_since_update: int
    detection: Detection


class FishTracker:
    """Compose detection, tracking, fish classification, re-identification and motion features.

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
    classifier:
        Default :class:`TrackClassifier`. Pass
        :meth:`TrackClassifier.permissive` to trust every confirmed track
        (a detector that only ever sees fish).
    recognise_distance:
        Bhattacharyya distance within which a candidate track that looks
        like a known fish missing from view is promoted without travelling
        again (0 disables; stricter than re-ID's own matching threshold).
    coast_frames:
        For how many frames a fish track that lost its detection - a fish
        passing behind another fish, a rock, a front plant - is still
        reported, at its Kalman-predicted box with ``score=0.0``. It yields
        no observation (no measurement, no innovation). 0 reports only
        tracks matched this frame.
    switch_distance, switch_frames:
        A fish track whose colour appearance stays more than
        ``switch_distance`` (Bhattacharyya) from its appearance at promotion
        for ``switch_frames`` frames has latched onto something else and is
        demoted (see :meth:`_check_continuity`; 0 disables).
    """

    def __init__(
        self,
        detector: Detector | None = None,
        tracker: ByteTracker | None = None,
        reid: FishReID | None = None,
        trail_len: int = 30,
        *,
        classifier: TrackClassifier | None = None,
        recognise_distance: float = 0.3,
        coast_frames: int = 20,
        switch_distance: float = 0.6,
        switch_frames: int = 5,
    ) -> None:
        if trail_len < 0:
            raise ValueError("trail_len must be >= 0")
        self.detector: Detector = detector if detector is not None else MotionDetector()
        self.tracker = (
            tracker if tracker is not None else ByteTracker(history_len=max(64, trail_len))
        )
        self.reid = reid if reid is not None else FishReID()
        self.classifier = classifier if classifier is not None else TrackClassifier()
        self.recognise_distance = float(recognise_distance)
        if coast_frames < 0:
            raise ValueError("coast_frames must be >= 0")
        self.coast_frames = int(coast_frames)
        if switch_frames < 1:
            raise ValueError("switch_frames must be >= 1")
        self.switch_distance = float(switch_distance)
        self.switch_frames = int(switch_frames)
        self.trail_len = int(trail_len)
        self._motion: dict[int, MotionFeatures] = {}
        self._characters: dict[str, FishCharacter] = {}
        self._verdicts: dict[int, TrackVerdict] = {}
        self._reference: dict[int, npt.NDArray[np.float32]] = {}
        self._drift: dict[int, int] = {}

    @property
    def characters(self) -> dict[str, FishCharacter]:
        """Behaviour statistics by ``fish_id`` (a new dict)."""
        return dict(self._characters)

    @property
    def identities(self) -> dict[str, FishIdentity]:
        """Re-identification gallery by ``fish_id``."""
        return self.reid.identities

    @property
    def verdicts(self) -> dict[int, TrackVerdict]:
        """Classification of every track matched in the latest frame, by ``track_id``.

        Explains *why* a track is or is not a fish (dashboard, debugging).
        """
        return dict(self._verdicts)

    def track_class(self, track_id: int) -> TrackClass | None:
        """Current class of ``track_id`` (``None`` if unknown)."""
        verdict = self.classifier.verdict(track_id)
        return verdict.label if verdict is not None else None

    def process(self, frame: Frame) -> TrackingResult:
        """Run the whole vision pipeline on one frame."""
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame.timestamp)
        self._verdicts = self.classifier.update(tracks, (frame.width, frame.height))
        self._recognise(tracks)
        fish = {tr.track_id for tr in tracks if tr.is_confirmed and self._is_fish(tr.track_id)}
        foreground = (
            self.detector.foreground_mask if isinstance(self.detector, ForegroundProvider) else None
        )
        views = [
            _ReIDView(tr.track_id, tr.track_id in fish, tr.time_since_update, tr.detection)
            for tr in tracks
        ]
        fish_ids = self.reid.update(frame.image, views, foreground)
        fish -= self._check_continuity(tracks, fish)
        period = self.tracker.frame_period

        snapshots: list[TrackSnapshot] = []
        observations: list[FishObservation] = []
        for track in tracks:
            is_fish = track.track_id in fish
            fish_id = fish_ids.get(track.track_id) if is_fish else None
            snapshots.append(self._snapshot(track, fish_id, period, is_fish))
            if is_fish and track.time_since_update == 0 and track.innovation is not None:
                observations.append(self._observe(track, frame.timestamp, period, fish_id))
        coasting = [
            tr
            for tr in self.tracker.lost_tracks
            if tr.time_since_update <= self.coast_frames and self._is_fish(tr.track_id)
        ]
        for track in coasting:
            fish_id = self.reid.fish_id_for(track.track_id)
            snapshots.append(self._snapshot(track, fish_id, period, True, coasting=True))
        self._prune()
        if isinstance(self.detector, FishHintReceiver):
            # Fish in view and fish coasting through an occlusion: the
            # background model must not learn either (see MotionDetector.protect).
            hidden = [tr.bbox for tr in self.tracker.lost_tracks if self._is_fish(tr.track_id)]
            self.detector.protect([tr.bbox for tr in tracks if tr.track_id in fish] + hidden)
        return TrackingResult(
            frame_index=frame.index,
            t=frame.timestamp,
            detections=tuple(detections),
            tracks=tuple(snapshots),
            observations=tuple(observations),
        )

    def _is_fish(self, track_id: int) -> bool:
        return self.classifier.is_fish(track_id)

    def _check_continuity(self, tracks: list[Track], fish: set[int]) -> set[int]:
        """Demote fish tracks that no longer look like the fish they were promoted for.

        IoU association has a classic failure: a fish lingers in front of
        plants, swims off, and its track stays on the leaves it overlapped
        (they now differ from a background model that could not see them
        while the fish was there). The track is "fish" for life, so without
        a check every swaying leaf behind it would be credited as fish
        entropy. Colour gives it away: the crops the track now matches look
        nothing like the track's appearance when it was promoted. After
        ``switch_frames`` consecutive crops beyond ``switch_distance`` the
        track is demoted, its appearance and identity binding are dropped
        (so the real fish, on whatever track it now has, can be
        re-identified), and whatever it follows must prove itself again.
        The reference follows slow, small changes (a fish turning shows
        another side). IR frames carry no colour: at night this check is
        off. Returns the demoted track ids.
        """
        demoted: set[int] = set()
        if self.switch_distance <= 0.0:
            return demoted
        for track in tracks:
            track_id = track.track_id
            if track_id not in fish or track.time_since_update != 0:
                continue
            current = self.reid.track_appearance(track_id, latest=True)
            if current is None:
                continue
            reference = self._reference.get(track_id)
            if reference is None:
                model = self.reid.track_appearance(track_id)
                if model is not None:
                    self._reference[track_id] = model
                continue
            distance = float(bhattacharyya_distance(reference, current)[0, 0])
            if distance <= self.switch_distance:
                self._drift[track_id] = 0
                if distance <= _REFERENCE_FOLLOW:
                    self._reference[track_id] = _blend(reference, current, _REFERENCE_RATE)
                continue
            self._drift[track_id] = self._drift.get(track_id, 0) + 1
            if self._drift[track_id] >= self.switch_frames:
                self.classifier.demote(
                    track_id, f"appearance changed (distance {distance:.2f}): followed clutter"
                )
                self.reid.forget_track(track_id)
                self._forget_track(track_id)
                demoted.add(track_id)
        return demoted

    def _forget_track(self, track_id: int) -> None:
        self._reference.pop(track_id, None)
        self._drift.pop(track_id, None)
        self._motion.pop(track_id, None)

    def _recognise(self, tracks: list[Track]) -> None:
        """Promote candidates that re-identification recognises as a known, missing fish.

        A fish that reappears on a fresh track (after an occlusion longer
        than the lost buffer, or a merge with another fish) would otherwise
        have to prove itself by travelling again. If its appearance so far
        is within ``recognise_distance`` of an identity that no fish track
        in view holds, it is that fish. Plants cannot use this shortcut to
        become fish unless they look like a missing fish *and* meet the
        persistence and confidence tests; IR frames carry no colour, so at
        night the shortcut is off.
        """
        if self.recognise_distance <= 0.0:
            return
        held: set[str] = set()
        candidates: list[int] = []
        for tr in tracks:
            if self.classifier.is_fish(tr.track_id):
                fish_id = self.reid.fish_id_for(tr.track_id)
                if fish_id is not None:
                    held.add(fish_id)
            elif tr.is_confirmed and self.classifier.qualifies(tr.track_id, _RECOGNISE_SCORE):
                candidates.append(tr.track_id)
        for track_id in candidates:
            match = self.reid.closest_identity(track_id, exclude=held)
            if match is not None and match[1] <= self.recognise_distance:
                identity = self.reid.identities[match[0]]
                self.classifier.promote(
                    track_id, f"re-identified as {identity.name}", _RECOGNISE_SCORE
                )
                held.add(match[0])

    def _snapshot(
        self,
        track: Track,
        fish_id: str | None,
        period: float,
        is_fish: bool,
        *,
        coasting: bool = False,
    ) -> TrackSnapshot:
        vx, vy = track.velocity
        history = list(track.history)[-self.trail_len :] if self.trail_len else []
        return TrackSnapshot(
            track_id=track.track_id,
            fish_id=fish_id,
            bbox=track.bbox,
            score=0.0 if coasting else track.score,
            center=track.center,
            velocity=(vx / period, vy / period),
            trail=tuple((x, y) for _, x, y in history),
            confirmed=is_fish,
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
        known = set(self._motion) | set(self._reference)
        for track_id in known - alive:
            self._forget_track(track_id)
