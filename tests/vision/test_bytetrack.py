"""ByteTrack association logic on synthetic detection streams."""

from __future__ import annotations

import numpy as np
import pytest

from puffer.types import BBox, Detection
from puffer.vision.bytetrack import ByteTracker, TrackState
from puffer.vision.geometry import bbox_to_xyah

FPS = 30.0


def box(cx: float, cy: float, w: float = 30.0, h: float = 14.0) -> BBox:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def det(cx: float, cy: float, score: float = 0.9, **kw: float) -> Detection:
    return Detection(box(cx, cy, **kw), score)


def run(tracker: ByteTracker, frames: list[list[Detection]], start: int = 0) -> list[list[int]]:
    """Feed frames; return the ids returned for each frame."""
    out = []
    for i, dets in enumerate(frames, start=start):
        out.append([t.track_id for t in tracker.update(dets, i / FPS)])
    return out


def ids_by_x(tracker: ByteTracker) -> list[int]:
    return [t.track_id for t in sorted(tracker.tracked_tracks, key=lambda t: t.center[0])]


def test_no_detections() -> None:
    tracker = ByteTracker()
    assert run(tracker, [[]] * 5) == [[]] * 5
    assert tracker.lost_tracks == []
    assert tracker.frame_id == 5


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="low_thresh"):
        ByteTracker(track_thresh=0.3, low_thresh=0.4)
    with pytest.raises(ValueError, match="min_hits"):
        ByteTracker(min_hits=0)
    with pytest.raises(ValueError, match="frame_rate"):
        ByteTracker(frame_rate=0)


def test_birth_requires_min_hits() -> None:
    tracker = ByteTracker(min_hits=3)
    states = []
    for i in range(4):
        tracks = tracker.update([det(100 + 2 * i, 50)], i / FPS)
        assert len(tracks) == 1
        states.append(tracks[0].state)
    assert states == [
        TrackState.TENTATIVE,
        TrackState.TENTATIVE,
        TrackState.CONFIRMED,
        TrackState.CONFIRMED,
    ]
    assert [t.track_id for t in tracker.confirmed_tracks] == [1]


def test_min_hits_one_confirms_immediately() -> None:
    tracker = ByteTracker(min_hits=1)
    (track,) = tracker.update([det(10, 10)], 0.0)
    assert track.is_confirmed
    assert track.innovation is None  # no measurement update yet, only initiation


def test_tentative_track_missed_once_is_removed() -> None:
    tracker = ByteTracker()
    run(tracker, [[det(100, 50)], [det(101, 50)], []])
    assert tracker.tracked_tracks == []
    assert tracker.lost_tracks == []  # tentative tracks never become lost
    (track,) = tracker.update([det(102, 50)], 4 / FPS)
    assert track.track_id == 2


def test_new_track_threshold() -> None:
    tracker = ByteTracker(track_thresh=0.5, new_track_thresh=0.6)
    run(tracker, [[det(100, 50, score=0.55)]] * 3 + [[det(50, 50, score=0.3)]] * 3)
    assert tracker.tracked_tracks == []


def test_low_score_detections_keep_confirmed_track_alive() -> None:
    tracker = ByteTracker()
    run(tracker, [[det(100 + 2 * i, 50)] for i in range(5)])
    # the fish turns faint (half hidden): only low-score detections for 10 frames
    run(tracker, [[det(110 + 2 * i, 50, score=0.2)] for i in range(10)], start=5)
    tracks = tracker.tracked_tracks
    assert [t.track_id for t in tracks] == [1]
    assert tracks[0].state is TrackState.CONFIRMED
    assert tracks[0].score == pytest.approx(0.2)


def test_low_score_detection_never_matches_tentative_track() -> None:
    tracker = ByteTracker()
    run(tracker, [[det(100, 50)], [det(101, 50, score=0.2)]])
    assert tracker.tracked_tracks == []


def test_innovation_is_measured_minus_predicted() -> None:
    tracker = ByteTracker()
    run(tracker, [[det(100 + 3 * i, 50)] for i in range(10)])
    (track,) = tracker.tracked_tracks
    mean, _ = tracker.kalman.predict(track.mean, track.covariance)
    surprise = det(130 + 6.0, 50 - 4.0)  # constant velocity would give (130, 50)
    tracker.update([surprise], 10 / FPS)
    measured = bbox_to_xyah(surprise.bbox)
    innovation = track.innovation
    assert innovation is not None
    assert innovation == pytest.approx((measured[0] - mean[0], measured[1] - mean[1]))
    assert innovation[0] > 4.0
    assert innovation[1] < -2.5


def test_crossing_keeps_ids() -> None:
    """Two fish swim past each other; during the overlap the rear one is half hidden."""
    tracker = ByteTracker()
    frames = []
    for i in range(80):
        xa, xb = 40 + 3 * i, 280 - 3 * i
        ya, yb = 100.0, 106.0
        overlap = abs(xa - xb) < 30
        frames.append([det(xa, ya), det(xb, yb, score=0.3 if overlap else 0.9)])
    run(tracker, frames[:10])
    before = ids_by_x(tracker)  # [fish A (left), fish B (right)]
    run(tracker, frames[10:], start=10)
    after = ids_by_x(tracker)  # now B is left, A is right
    assert after == before[::-1]
    assert sorted(t.track_id for t in tracker.tracked_tracks) == [1, 2]


def test_short_occlusion_recovered_from_lost_buffer() -> None:
    tracker = ByteTracker()
    frames: list[list[Detection]] = []
    for i in range(60):
        x = 50 + 2.5 * i
        hidden = 20 <= i < 32  # behind a rock for 12 frames
        frames.append([] if hidden else [det(x, 80)])
    run(tracker, frames[:21])
    run(tracker, frames[21:30], start=21)
    assert tracker.tracked_tracks == []
    (lost,) = tracker.lost_tracks
    assert lost.state is TrackState.LOST
    assert lost.is_confirmed
    assert lost.time_since_update == 10
    # the lost track keeps coasting at the learned velocity
    assert lost.center[0] == pytest.approx(50 + 2.5 * 29, abs=3.0)
    run(tracker, frames[30:], start=30)
    assert [t.track_id for t in tracker.tracked_tracks] == [1]


def test_lost_track_removed_after_buffer() -> None:
    tracker = ByteTracker(track_buffer=30)
    run(tracker, [[det(100 + i, 80)] for i in range(10)])
    run(tracker, [[]] * 30, start=10)
    assert len(tracker.lost_tracks) == 1  # still within the 30-frame buffer
    run(tracker, [[]], start=40)
    assert tracker.lost_tracks == []
    (track,) = tracker.update([det(140, 80)], 41 / FPS)
    assert track.track_id == 2
    assert track.state is TrackState.TENTATIVE


def test_buffer_scales_with_frame_rate() -> None:
    assert ByteTracker(track_buffer=30, frame_rate=15).max_time_lost == 15
    assert ByteTracker(track_buffer=30, frame_rate=60).max_time_lost == 60


def test_duplicate_of_older_lost_track_is_dropped() -> None:
    tracker = ByteTracker()
    # A long-lived track that gets lost while moving slowly to the right ...
    run(tracker, [[det(100 + 0.5 * i, 80)] for i in range(40)])
    run(tracker, [[]], start=40)
    (lost,) = tracker.lost_tracks
    # ... while a new fish appears nearby, outside its gate, and is confirmed.
    newcomer = [det(220, 80)]
    run(tracker, [newcomer] * 3, start=41)
    # The coasting box is still far away; both coexist.
    assert {t.track_id for t in tracker.tracked_tracks} == {2}
    # Teleport the lost track's state onto the newcomer (as if it had coasted there).
    lost.mean[:2] = [220.0, 80.0]
    lost.mean[4:] = 0.0
    tracker.update(newcomer, 44 / FPS)
    tracker.update(newcomer, 45 / FPS)
    # The older identity wins; the young duplicate is gone for good.
    assert [t.track_id for t in tracker.tracked_tracks] == [1]


def test_frame_period_is_measured() -> None:
    tracker = ByteTracker(frame_rate=30)
    assert tracker.frame_period == pytest.approx(1 / 30)
    for i in range(100):
        tracker.update([], i / 25.0)
    assert tracker.frame_period == pytest.approx(1 / 25, rel=1e-3)
    tracker.update([], 100 / 25.0 + 10.0)  # a stall is not a frame rate
    assert tracker.frame_period == pytest.approx(1 / 25, rel=1e-3)


def test_history_and_age() -> None:
    tracker = ByteTracker(history_len=5)
    run(tracker, [[det(100 + i, 50)] for i in range(8)])
    (track,) = tracker.tracked_tracks
    assert track.age == 8
    assert track.hits == 8
    assert len(track.history) == 5
    ts = [h[0] for h in track.history]
    assert ts == sorted(ts)
    assert ts[-1] == pytest.approx(7 / FPS)
    assert "confirmed" in repr(track)


def test_reset() -> None:
    tracker = ByteTracker()
    run(tracker, [[det(10, 10)]] * 4)
    tracker.reset()
    assert tracker.tracked_tracks == []
    assert tracker.frame_id == 0
    (track,) = tracker.update([det(10, 10)], 0.0)
    assert track.track_id == 1


def test_many_targets_random_walk(rng: np.random.Generator) -> None:
    """Six well-separated fish with jittery detections: zero id switches."""
    tracker = ByteTracker()
    starts = np.column_stack([np.linspace(40, 560, 6), rng.uniform(60, 400, 6)])
    vel = rng.normal(0, 1.5, size=(6, 2))
    for i in range(150):
        pos = starts + vel * i
        dets = [
            det(float(x), float(y), score=float(rng.uniform(0.6, 1.0)))
            for x, y in pos + rng.normal(0, 0.7, size=pos.shape)
        ]
        tracker.update(dets, i / FPS)
    assert sorted(t.track_id for t in tracker.tracked_tracks) == [1, 2, 3, 4, 5, 6]
