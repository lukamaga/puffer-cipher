"""FishTracker end to end on rendered scenes: tracking, identities, features, speed."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import numpy as np
import pytest

from puffer.types import Detection, Frame, TrackingResult
from puffer.vision.bytetrack import ByteTracker
from puffer.vision.pipeline import FishTracker

from .scenes import (
    BLUE,
    MAGENTA,
    ORANGE,
    WHITE,
    YELLOW,
    FishSpec,
    FishTruth,
    Scene,
    circle,
    linear,
    nearest_truth,
    wander,
    waypoints,
)

Results = list[tuple[TrackingResult, list[FishTruth]]]


def run(scene: Scene, n: int, tracker: FishTracker | None = None) -> Results:
    ft = tracker if tracker is not None else FishTracker()
    return [(ft.process(frame), truth) for frame, truth in scene.frames(n)]


IdCounts = dict[str, Counter[str | None]]


def fish_ids_by_truth(results: Results, t_from: float, t_to: float) -> IdCounts:
    """For each ground-truth fish, how often each fish_id was assigned in [t_from, t_to)."""
    out: IdCounts = defaultdict(Counter)
    for result, truth in results:
        if not t_from <= result.t < t_to:
            continue
        for snap in result.tracks:
            if snap.confirmed:
                gt = nearest_truth(snap.center, truth, 10.0)
                if gt is not None:
                    out[gt.name][snap.fish_id] += 1
    return out


def dominant(counter: Counter[str | None]) -> str | None:
    return counter.most_common(1)[0][0]


def test_result_structure() -> None:
    fish = [
        FishSpec("a", ORANGE, 30, 12, linear((30, 40), (50, 5))),
        FishSpec("b", BLUE, 26, 10, circle((200, 160), 40, 1.2)),
    ]
    ft = FishTracker(trail_len=10)
    results = run(Scene(fish, seed=1), 120, ft)
    last, _ = results[-1]
    assert last.frame_index == 119
    assert last.t == pytest.approx(119 / 30)
    assert len(last.detections) == 2
    assert len(last.tracks) == 2
    for snap in last.tracks:
        assert snap.confirmed
        assert snap.fish_id in ft.identities
        assert len(snap.trail) == 10
        assert snap.trail[-1] == pytest.approx(snap.center)
        assert snap.age_frames > 60
        assert 0.0 < snap.score <= 1.0
    obs_ids = {o.track_id for o in last.observations}
    assert obs_ids == {s.track_id for s in last.tracks}
    # no observations during warm-up and before confirmation
    assert all(not r.observations for r, _ in results[:32])
    assert set(ft.characters) == set(ft.identities)
    assert all(ch.n_obs > 50 for ch in ft.characters.values())


def test_tentative_tracks_produce_snapshots_but_no_observations() -> None:
    fish = [FishSpec("a", ORANGE, 30, 12, linear((30, 60), (50, 15)))]
    results = run(Scene(fish, seed=2), 33)
    first_seen = next(r for r, _ in results if r.tracks)
    assert not first_seen.tracks[0].confirmed
    assert first_seen.observations == ()


def test_observations_match_ground_truth_velocity_and_heading() -> None:
    fish = [
        FishSpec("east", ORANGE, 30, 12, linear((20, 50), (60, 0))),
        FishSpec("southwest", BLUE, 28, 11, linear((300, 90), (-40, 25))),
        # enters after the warm-up: a slow fish present during learning would be
        # partly baked into the background and smeared until it swims clear
        FishSpec("slow", YELLOW, 24, 10, linear((-30, 200), (15, -8))),
    ]
    settle_s = 1.0  # Kalman velocity convergence after the box stops being clipped
    fully_visible_since: dict[str, float] = {}
    errors: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for result, truth in run(Scene(fish, seed=3), 180):
        for fish_truth in truth:
            if fish_truth.visible_frac < 1.0:
                fully_visible_since.pop(fish_truth.name, None)
            else:
                fully_visible_since.setdefault(fish_truth.name, result.t)
        for o in result.observations:
            gt = nearest_truth((o.cx, o.cy), truth, 6.0)
            if gt is None or result.t < 2.0 or gt.name not in fully_visible_since:
                continue
            if result.t - fully_visible_since[gt.name] < settle_s:
                continue
            gvx, gvy = gt.velocity
            speed_err = math.hypot(o.vx - gvx, o.vy - gvy) / math.hypot(gvx, gvy)
            heading_err = abs(math.remainder(o.heading - math.atan2(gvy, gvx), 2 * math.pi))
            errors[gt.name].append((speed_err, heading_err))
    assert set(errors) == {"east", "southwest", "slow"}
    for name, errs in errors.items():
        assert len(errs) >= 20, name
        rel = np.array([e[0] for e in errs])
        hdg = np.array([e[1] for e in errs])
        assert float(np.median(rel)) < 0.08, name
        assert float(np.percentile(rel, 90)) < 0.2, name
        assert float(np.median(hdg)) < 0.08, name


def test_innovation_is_small_for_smooth_motion_and_reported() -> None:
    fish = [FishSpec("cruiser", ORANGE, 30, 12, linear((20, 100), (50, 0)))]
    innovations = [
        math.hypot(*o.innovation)
        for result, _ in run(Scene(fish, seed=4), 150)
        for o in result.observations
        if result.t > 2.0
    ]
    assert len(innovations) > 50
    assert 0.0 < float(np.median(innovations)) < 1.0  # sub-pixel, but never exactly zero


def test_identity_survives_leaving_and_reentering_the_tank() -> None:
    explorer = waypoints(
        [(0.0, (40.0, 70.0)), (3.0, (370.0, 70.0)), (5.5, (370.0, 160.0)), (8.0, (150.0, 160.0))]
    )
    fish = [
        FishSpec("explorer", ORANGE, 30, 12, explorer),  # out of view for ~2.5 s
        FishSpec("homebody", BLUE, 26, 10, circle((110, 150), 40, 1.0)),
    ]
    ft = FishTracker()
    results = run(Scene(fish, seed=5), 240, ft)
    before = fish_ids_by_truth(results, 1.5, 2.6)
    after = fish_ids_by_truth(results, 7.0, 8.0)
    assert dominant(after["explorer"]) == dominant(before["explorer"]) is not None
    assert dominant(after["homebody"]) == dominant(before["homebody"]) is not None
    assert len(ft.identities) == 2
    # the explorer really did get a new track: the lost buffer (1 s) had expired
    track_ids = defaultdict(set)
    for result, truth in results:
        for snap in result.tracks:
            gt = nearest_truth(snap.center, truth, 10.0)
            if gt is not None and snap.confirmed:
                track_ids[gt.name].add(snap.track_id)
    assert len(track_ids["explorer"]) >= 2


def test_crossing_fish_keep_identities() -> None:
    fish = [
        FishSpec("left", ORANGE, 30, 12, linear((20, 110), (60, 0))),
        FishSpec("right", BLUE, 30, 12, linear((300, 122), (-60, 0))),
    ]
    results = run(Scene(fish, seed=6), 150)
    before = fish_ids_by_truth(results, 1.2, 1.9)  # crossing happens at t ~ 2.3 s
    after = fish_ids_by_truth(results, 3.6, 4.5)
    for name in ("left", "right"):
        assert dominant(after[name]) == dominant(before[name]) is not None
    assert dominant(before["left"]) != dominant(before["right"])


def test_short_occlusion_keeps_track_id() -> None:
    fish = [FishSpec("shy", MAGENTA, 30, 12, linear((20, 100), (60, 0)))]
    rock = (2.0, 2.4, (125.0, 70.0, 175.0, 130.0))  # 12 frames behind the rock
    scene = Scene(fish, seed=7, occluders=[rock])
    track_ids = {s.track_id for r, _ in run(scene, 130) for s in r.tracks if s.confirmed}
    assert track_ids == {1}


def test_custom_detector_without_foreground_mask() -> None:
    """Any types.Detector works, e.g. YOLO; here a perfect oracle detector."""

    class Oracle:
        def __init__(self, scene: Scene) -> None:
            self.scene = scene

        def detect(self, frame: Frame) -> list[Detection]:
            return [
                Detection(gt.bbox, 0.95)
                for gt in self.scene.truth_at(frame.timestamp)
                if gt.visible_frac > 0.5
            ]

    fish = [FishSpec("a", YELLOW, 30, 12, linear((30, 80), (40, 10)))]
    scene = Scene(fish, seed=8)
    ft = FishTracker(detector=Oracle(scene), tracker=ByteTracker(min_hits=2))
    results = run(scene, 40, ft)
    assert results[1][0].observations  # confirmed on the second frame
    assert len(ft.identities) == 1


def test_rejects_negative_trail() -> None:
    with pytest.raises(ValueError, match="trail_len"):
        FishTracker(trail_len=-1)


def test_zero_trail() -> None:
    fish = [FishSpec("a", ORANGE, 30, 12, linear((30, 60), (50, 15)))]
    results = run(Scene(fish, seed=9), 40, FishTracker(trail_len=0))
    assert all(s.trail == () for r, _ in results for s in r.tracks)


@pytest.mark.slow
def test_throughput_640x480(rng: np.random.Generator) -> None:
    """Report pipeline fps on 640x480 frames with five fish (frames pre-rendered)."""
    colours = [ORANGE, BLUE, YELLOW, MAGENTA, WHITE]
    fish = [
        FishSpec(f"f{i}", c, 50, 20, wander(rng, (120 + 100 * i, 240), (80, 170)))
        for i, c in enumerate(colours)
    ]
    scene = Scene(fish, width=640, height=480, seed=10)
    frames = [scene.render()[0] for _ in range(210)]
    ft = FishTracker()
    for frame in frames[:60]:
        ft.process(frame)
    start = time.perf_counter()
    for frame in frames[60:]:
        ft.process(frame)
    fps = 150 / (time.perf_counter() - start)
    print(f"\nFishTracker throughput at 640x480 with 5 fish: {fps:.1f} fps")
    assert fps >= 15.0  # target is >= 30 fps; keep slack for loaded CI machines
