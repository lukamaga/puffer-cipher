"""FishTracker end to end on rendered scenes: tracking, identities, features, speed."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable

import cv2
import numpy as np
import pytest

from puffer.types import Detection, Frame, TrackingResult
from puffer.vision.bytetrack import ByteTracker
from puffer.vision.classify import TrackClass, TrackClassifier
from puffer.vision.detector import MotionDetector
from puffer.vision.pipeline import FishTracker

from .scenes import (
    BLUE,
    GREEN,
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
    sway,
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
    """Most frequent identity, ignoring frames before the fish was identified."""
    named = Counter({k: v for k, v in counter.items() if k is not None})
    return named.most_common(1)[0][0] if named else None


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
    # Promotion to fish (~0.5-1 s of travel) and enrolment (0.5 s) come first.
    assert all(ch.n_obs > 30 for ch in ft.characters.values())


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
    before = fish_ids_by_truth(results, 1.5, 2.6)  # the explorer leaves at ~2.7 s
    # The homebody circles in place through the warm-up, which leaves a faint
    # smear of it in the median background: it is identified a little later.
    homebody = fish_ids_by_truth(results, 3.0, 5.0)["homebody"]
    after = fish_ids_by_truth(results, 7.0, 8.0)
    assert dominant(after["explorer"]) == dominant(before["explorer"]) is not None
    assert dominant(after["homebody"]) == dominant(homebody) is not None
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
        FishSpec("left", ORANGE, 30, 12, linear((20, 110), (45, 0))),
        FishSpec("right", BLUE, 30, 12, linear((300, 122), (-45, 0))),
    ]
    results = run(Scene(fish, seed=6), 165)
    # identified by ~2 s (warm-up, promotion, enrolment); crossing at t ~ 3.1 s
    before = fish_ids_by_truth(results, 2.2, 2.7)
    after = fish_ids_by_truth(results, 4.5, 5.4)
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
    ft = FishTracker(
        detector=Oracle(scene),
        tracker=ByteTracker(min_hits=2),
        classifier=TrackClassifier.permissive(),
    )
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


# --------------------------------------------------------------------------
# what is (not) a fish: track classification in the pipeline
# --------------------------------------------------------------------------


def credited_names(results: Results, max_dist: float = 10.0) -> Counter[str | None]:
    """Which scene object each *observation* (fish-channel sample) came from."""
    out: Counter[str | None] = Counter()
    for result, truth in results:
        for o in result.observations:
            gt = nearest_truth((o.cx, o.cy), truth, max_dist)
            out[gt.name if gt is not None else None] += 1
    return out


def test_swaying_leaves_are_never_credited() -> None:
    """Leaves swinging around their base: suppressed by the detector or rejected as tracks."""
    fish = [
        FishSpec("leaf", GREEN, 40, 16, sway((90, 170), (7, 3), period=4.0)),
        FishSpec("leaf2", GREEN, 36, 14, sway((250, 60), (5, 6), period=3.1, phase=1.0)),
        FishSpec("swimmer", ORANGE, 30, 12, linear((40, 60), (40, 12))),
    ]
    ft = FishTracker()
    results = run(Scene(fish, seed=31), 300, ft)
    credited = credited_names(results, max_dist=15.0)
    assert credited["swimmer"] > 150
    assert credited["leaf"] == credited["leaf2"] == 0
    leaf_tracks = {
        s.track_id
        for r, truth in results
        for s in r.tracks
        if (gt := nearest_truth(s.center, truth, 12.0)) is not None and gt.name.startswith("leaf")
    }
    assert all(not s.confirmed for r, _ in results for s in r.tracks if s.track_id in leaf_tracks)
    labels = {ft.track_class(tid) for tid in leaf_tracks} - {None}
    assert labels <= {TrackClass.CANDIDATE, TrackClass.ANCHORED}
    assert len(ft.identities) == 1  # no leaf is given a name


def test_swaying_leaf_track_is_never_confirmed() -> None:
    """Even when a leaf is detected every frame (a perfect detector), its track is not a fish."""
    leaf = sway((160, 120), (8, 3), period=4.0)

    class LeafDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            x, y = leaf(frame.timestamp)
            return [Detection((x - 20, y - 8, x + 20, y + 8), 0.95)]

    ft = FishTracker(detector=LeafDetector())
    results = run(Scene([], seed=35), 300, ft)
    assert all(r.tracks and not r.observations for r, _ in results[10:])
    assert ft.track_class(results[-1][0].tracks[0].track_id) is TrackClass.ANCHORED


def test_rising_bubbles_are_not_fish() -> None:
    def bubble(x: float, t0: float) -> FishSpec:
        rise = waypoints([(0.0, (x, 300.0)), (t0, (x, 300.0)), (t0 + 6.0, (x + 4.0, -60.0))])
        return FishSpec(f"bubble@{t0}", WHITE, 10, 9, rise)

    fish = [bubble(250.0 + 8 * k, 1.2 + 1.3 * k) for k in range(6)]
    fish.append(FishSpec("swimmer", BLUE, 30, 12, linear((300, 80), (-40, 10))))
    ft = FishTracker()
    results = run(Scene(fish, seed=32), 330, ft)
    credited = credited_names(results)
    assert credited["swimmer"] > 150
    assert sum(n for name, n in credited.items() if name and name.startswith("bubble")) == 0
    assert credited[None] < 10  # the swimmer's box clipped as it leaves the frame
    seen = {
        v.label
        for r, _ in results
        for v in [ft.classifier.verdict(s.track_id) for s in r.tracks]
        if v
    }
    assert TrackClass.BUBBLE in seen


def test_resting_fish_keeps_its_track_identity_and_credit() -> None:
    """Swims in, rests 8 s: still a confirmed fish, same track, same name, still observed."""
    path = waypoints([(0.0, (20.0, 100.0)), (2.0, (160.0, 100.0)), (10.0, (160.0, 100.0))])
    ft = FishTracker()
    results = run(Scene([FishSpec("sleepy", ORANGE, 30, 12, path)], seed=33), 300, ft)
    resting = [r for r, _ in results if r.t >= 3.0]
    assert all(len([s for s in r.tracks if s.confirmed]) == 1 for r in resting)
    assert len({s.track_id for r in resting for s in r.tracks}) == 1
    assert len({s.fish_id for r in resting for s in r.tracks}) == 1
    assert sum(len(r.observations) for r in resting) >= 0.95 * len(resting)


def test_hidden_fish_coasts_without_observations() -> None:
    fish = [FishSpec("shy", MAGENTA, 30, 12, linear((20, 100), (60, 0)))]
    rock = (2.0, 2.3, (125.0, 70.0, 175.0, 130.0))  # 9 frames behind the rock
    results = run(Scene(fish, seed=7, occluders=[rock]), 120)
    hidden = [r for r, _ in results if 2.05 <= r.t < 2.25]
    assert hidden
    for r in hidden:
        assert [s.score for s in r.tracks if s.confirmed] == [0.0]  # predicted box, no detection
        assert r.observations == ()
    after = [r for r, _ in results if r.t >= 2.6]
    assert {s.track_id for r in after for s in r.tracks} == {1}
    assert all(r.observations for r in after)


def test_tracker_tells_the_detector_where_the_fish_are() -> None:
    class Spy(MotionDetector):
        def __init__(self) -> None:
            super().__init__()
            self.hints: list[list[tuple[float, float, float, float]]] = []

        def protect(self, boxes: Iterable[tuple[float, float, float, float]]) -> None:
            self.hints.append(list(boxes))
            super().protect(self.hints[-1])

    spy = Spy()
    fish = [FishSpec("a", ORANGE, 30, 12, linear((30, 60), (50, 10)))]
    results = run(Scene(fish, seed=34), 90, FishTracker(detector=spy))
    assert len(spy.hints) == 90
    confirmed_last = [s.bbox for s in results[-1][0].tracks if s.confirmed]
    assert confirmed_last
    assert spy.hints[-1] == confirmed_last
    assert spy.hints[0] == []


def test_appearance_switch_demotes_a_hijacked_track() -> None:
    """A fish track that ends up following something of another colour stops being credited."""

    class Swapper:
        """Detects one box that moves smoothly; its colour changes at frame 60."""

        def __init__(self) -> None:
            self.n = 0

        def detect(self, frame: Frame) -> list[Detection]:
            self.n += 1
            x = 40.0 + 2.0 * self.n
            return [Detection((x, 80.0, x + 30.0, 92.0), 0.95)]

    class Painter:
        def __init__(self) -> None:
            self.n = 0

        def frame(self) -> Frame:
            img = np.full((240, 320, 3), (150, 125, 60), np.uint8)
            x = 40 + 2 * (self.n + 1)
            colour = ORANGE if self.n < 60 else GREEN
            cv2.ellipse(img, (x + 15, 86), (15, 6), 0, 0, 360, colour, -1)
            f = Frame(self.n, self.n / 30.0, 0.0, img)
            self.n += 1
            return f

    ft = FishTracker(detector=Swapper(), tracker=ByteTracker(min_hits=2))
    painter = Painter()
    results, reasons = [], []
    for _ in range(90):
        results.append(ft.process(painter.frame()))
        verdict = ft.classifier.verdict(1)
        reasons.append(verdict.reason if verdict is not None else "")
    assert results[55].observations  # a fish before the swap
    demoted = next(i for i, r in enumerate(reasons) if r.startswith("appearance changed"))
    assert 60 < demoted <= 66  # within switch_frames of the swap
    # The green thing must now prove itself like any new track: it travels, so it
    # is accepted again - but only after a fresh persistence + travel period.
    first_again = next(i for i in range(demoted, 90) if results[i].observations)
    assert first_again - demoted >= ft.classifier.min_frames
