"""TrackClassifier: fish travel, plants sway around an anchor, bubbles rise."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from puffer.types import BBox, Detection, Point
from puffer.vision.classify import TrackClass, TrackClassifier

FRAME = (640, 480)


@dataclass
class FakeTrack:
    track_id: int
    center: Point
    detection: Detection
    time_since_update: int = 0


def box(cx: float, cy: float, w: float, h: float) -> BBox:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


Path = Callable[[int], tuple[float, float]]


def feed(
    clf: TrackClassifier,
    path: Path,
    n: int,
    *,
    size: tuple[float, float] = (60.0, 25.0),
    score: float = 0.95,
    track_id: int = 1,
    start: int = 0,
) -> list[TrackClass]:
    """Feed ``n`` frames of one track following ``path(frame)``; return the labels."""
    labels = []
    for i in range(start, start + n):
        cx, cy = path(i)
        det = Detection(box(cx, cy, *size), score)
        verdicts = clf.update([FakeTrack(track_id, (cx, cy), det)], FRAME)
        labels.append(verdicts[track_id].label)
    return labels


def swimming(i: int) -> tuple[float, float]:
    return (100.0 + 1.5 * i, 200.0 + 0.3 * i)  # 45 px/s at 30 fps


def swaying(i: int) -> tuple[float, float]:
    # a leaf: +-8 px around its anchor with a 4.8 s period
    return (300.0 + 8.0 * math.sin(2 * math.pi * i / 144), 300.0 + 2.0 * math.cos(i / 20))


def rising(i: int) -> tuple[float, float]:
    return (400.0 + 1.5 * math.sin(i / 3), 440.0 - 2.5 * i)  # 75 px/s up, small wobble


def test_swimming_track_becomes_fish_and_stays_fish_at_rest() -> None:
    clf = TrackClassifier()
    labels = feed(clf, swimming, 60)
    first_fish = labels.index(TrackClass.FISH)
    assert clf.min_frames <= first_fish + 1 <= 30  # within a second
    assert labels[first_fish:] == [TrackClass.FISH] * (60 - first_fish)
    # it stops and rests for 10 s: still a fish (sticky)
    resting = feed(clf, lambda i: swimming(59), 300, start=60)
    assert set(resting) == {TrackClass.FISH}
    verdict = clf.verdict(1)
    assert verdict is not None
    assert verdict.is_fish
    assert "travelled" in verdict.reason


def test_swaying_plant_is_anchored_never_fish() -> None:
    clf = TrackClassifier()
    labels = feed(clf, swaying, 600, size=(40.0, 90.0))
    assert TrackClass.FISH not in labels
    assert labels[-1] is TrackClass.ANCHORED
    verdict = clf.verdict(1)
    assert verdict is not None
    assert verdict.travel_bl < clf.min_travel_bl


def test_rising_bubble_cluster_is_a_bubble() -> None:
    clf = TrackClassifier()
    labels = feed(clf, rising, 60, size=(14.0, 22.0))
    assert TrackClass.FISH not in labels
    assert labels[-1] is TrackClass.BUBBLE


def test_large_object_rising_is_a_fish() -> None:
    """Only *small* rising blobs are bubbles: a fish swimming up is a fish."""
    clf = TrackClassifier()
    labels = feed(clf, rising, 60, size=(60.0, 40.0))
    assert labels[-1] is TrackClass.FISH


def test_bubble_like_start_then_turning_becomes_fish() -> None:
    clf = TrackClassifier()

    def up_then_sideways(i: int) -> tuple[float, float]:
        return rising(min(i, 20)) if i < 20 else (rising(20)[0] + 3.0 * (i - 20), rising(20)[1])

    labels = feed(clf, up_then_sideways, 80, size=(16.0, 14.0))
    assert TrackClass.BUBBLE in labels
    assert labels[-1] is TrackClass.FISH


def test_low_confidence_track_stays_candidate() -> None:
    clf = TrackClassifier()
    labels = feed(clf, swimming, 60, score=0.45)
    assert set(labels) == {TrackClass.CANDIDATE}
    verdict = clf.verdict(1)
    assert verdict is not None
    assert "median score" in verdict.reason


def test_jittering_small_blob_needs_travel_at_tank_scale() -> None:
    """A 20 px fragment whose centre jumps 12 px moves 0.6 'body lengths' - not travel."""
    clf = TrackClassifier()
    labels = feed(
        clf, lambda i: (200.0 + (12.0 if (i // 3) % 2 else 0.0), 300.0), 300, size=(20, 18)
    )
    assert TrackClass.FISH not in labels


def test_persistence_required() -> None:
    clf = TrackClassifier(min_frames=20)
    labels = feed(clf, lambda i: (100.0 + 6.0 * i, 200.0), 25)
    assert labels[:19] == [TrackClass.CANDIDATE] * 19
    assert labels[-1] is TrackClass.FISH


def test_only_matched_tracks_contribute() -> None:
    clf = TrackClassifier()
    feed(clf, swimming, 5)
    det = Detection(box(*swimming(5), 60, 25), 0.95)
    coasting = FakeTrack(1, swimming(5), det, time_since_update=3)
    before = clf.verdict(1)
    clf.update([coasting], FRAME)
    assert clf.verdict(1) == before


def test_promote_demote_and_qualifies() -> None:
    clf = TrackClassifier()
    feed(clf, lambda i: (300.0, 200.0), 5)  # resting candidate
    assert not clf.qualifies(1)  # too few frames
    clf.promote(1, "outside evidence")
    assert not clf.is_fish(1)
    feed(clf, lambda i: (300.0, 200.0), 20, start=5)
    assert clf.qualifies(1)
    assert not clf.qualifies(1, min_score=0.99)
    clf.promote(1, "re-identified as Nemo")
    assert clf.is_fish(1)
    clf.demote(1, "followed clutter")
    verdict = clf.verdict(1)
    assert verdict is not None
    assert verdict.label is TrackClass.CANDIDATE
    assert verdict.frames == 0
    assert verdict.reason == "followed clutter"
    clf.promote(2, "unknown track")  # no-op
    clf.demote(2, "unknown track")  # no-op
    assert clf.verdict(2) is None


def test_permissive_classifier_trusts_first_match() -> None:
    clf = TrackClassifier.permissive()
    assert feed(clf, lambda i: (300.0, 200.0), 1) == [TrackClass.FISH]


def test_forgets_unseen_tracks() -> None:
    clf = TrackClassifier(forget_after=10)
    feed(clf, swimming, 3, track_id=7)
    for _ in range(12):
        clf.update([], FRAME)
    assert clf.verdict(7) is None
    feed(clf, swimming, 3, track_id=8)
    clf.forget([8])
    assert clf.verdict(8) is None
    clf.reset()
    assert clf.update([], FRAME) == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_frames": 0}, ">= 1"),
        ({"travel_frames": 0}, ">= 1"),
        ({"min_score": 1.5}, "min_score"),
        ({"min_travel_bl": -1.0}, ">= 0"),
        ({"bubble_max_angle_deg": 120.0}, "bubble_max_angle_deg"),
        ({"bubble_area_frac": 1.0}, "bubble_area_frac"),
    ],
)
def test_parameter_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TrackClassifier(**kwargs)  # type: ignore[arg-type]
