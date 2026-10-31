"""Tracking metrics: exact bookkeeping on a scripted run, and regression targets on the simulator.

The simulator targets are the ones the README table reports
(``scripts/eval_tracking.py``): with plants, caustics and bubbles in the
tank, fewer than 3 % of the fish channel's observations may come from
anything but a fish, and at least 90 % of fish-frames must be covered by a
fish track. Seeds are fixed, so a failure is a regression, not bad luck.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from puffer.sources import SyntheticAquarium, SyntheticConfig
from puffer.types import BBox, Detection, FishObservation, Frame, TrackingResult, TrackSnapshot
from puffer.vision import FishTracker
from puffer.vision.evaluate import (
    TrackingMetrics,
    box_coverage,
    evaluate_tracking,
    match_boxes,
)

A: BBox = (0.0, 0.0, 10.0, 10.0)
B: BBox = (20.0, 0.0, 30.0, 10.0)
CLUTTER: BBox = (50.0, 50.0, 60.0, 60.0)


@dataclass(frozen=True)
class Truth:
    fish_id: str
    bbox: BBox


class ScriptedSource:
    """Frames at 30 fps with fixed ground truth (fish A and B)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.i = 0

    def read(self) -> Frame | None:
        if self.i >= self.n:
            return None
        frame = Frame(self.i, self.i / 30.0, 0.0, np.zeros((64, 64, 3), np.uint8))
        self.i += 1
        return frame

    def ground_truth(self) -> Sequence[Truth]:
        return [Truth("A", A), Truth("B", B)]


def snap(track_id: int, bbox: BBox, fish_id: str | None, confirmed: bool = True) -> TrackSnapshot:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return TrackSnapshot(track_id, fish_id, bbox, 0.9, (cx, cy), (0.0, 0.0), (), confirmed, 10)


def obs(track_id: int) -> FishObservation:
    return FishObservation(track_id, None, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, (0.1, -0.1))


class ScriptedTracker:
    """Returns a scripted :class:`TrackingResult` per frame."""

    def __init__(self) -> None:
        self.script = [
            # frame 0: fish A tracked (fish_id x) + a clutter track also credited
            ([A, CLUTTER], [snap(1, A, "x"), snap(2, CLUTTER, None)], [1, 2]),
            # frame 1: both fish; a candidate (confirmed=False) on clutter is ignored
            (
                [A, B],
                [snap(1, A, "x"), snap(3, B, "y"), snap(9, CLUTTER, None, confirmed=False)],
                [1, 3],
            ),
            # frame 2: A on a new track (ID switch); B re-identified as x (fish-ID switch)
            ([], [snap(4, A, "x"), snap(3, B, "x")], [4]),
            # frame 3: nothing at all
            ([], [], []),
        ]

    def process(self, frame: Frame) -> TrackingResult:
        dets, tracks, observed = self.script[frame.index]
        return TrackingResult(
            frame.index,
            frame.timestamp,
            tuple(Detection(b, 0.9) for b in dets),
            tuple(tracks),
            tuple(obs(t) for t in observed),
        )


def test_scripted_run_is_scored_exactly() -> None:
    m = evaluate_tracking(ScriptedSource(4), ScriptedTracker(), 4, warmup=0)
    assert (m.frames, m.gt_fish, m.gt_fish_frames) == (4, 2, 8)
    assert (m.detections, m.detection_matches) == (4, 3)
    assert m.detection_precision == pytest.approx(0.75)
    assert m.detection_recall == pytest.approx(3 / 8)
    assert (m.fish_track_frames, m.fish_matches) == (6, 5)
    assert m.fish_track_precision == pytest.approx(5 / 6)
    assert m.fish_recall == pytest.approx(5 / 8)
    assert m.observed_recall == pytest.approx(4 / 8)
    assert (m.observations, m.contaminated_observations) == (5, 1)
    assert m.contamination == pytest.approx(0.2)
    assert (m.id_switches, m.fish_id_switches, m.identities) == (1, 1, 2)
    # x on A in 3 frames and on B once; y on B once
    assert (m.identified_frames, m.pure_frames, m.assigned_frames) == (5, 4, 4)
    assert m.identity_purity == pytest.approx(0.8)
    assert m.identity_coverage == pytest.approx(0.5)
    assert m.mean_center_error_px == pytest.approx(0.0)
    assert m.seconds == pytest.approx(4 / 30)
    assert m.id_switches_per_minute == pytest.approx(450.0)
    assert m.detections_per_frame == pytest.approx(1.0)
    assert m.fish_per_frame == pytest.approx(2.0)
    assert m.process_fps > 0
    d = m.to_dict()
    assert d["contamination"] == pytest.approx(0.2)
    assert d["frames"] == 4


def test_warmup_frames_are_processed_not_scored() -> None:
    tracker = ScriptedTracker()
    m = evaluate_tracking(ScriptedSource(4), tracker, 4, warmup=2)
    assert m.frames == 2
    assert m.processed_frames == 4
    assert m.observations == 1  # frame 2 only


def test_undefined_ratios_are_nan_not_zero() -> None:
    m = evaluate_tracking(ScriptedSource(4), ScriptedTracker(), 4, warmup=3)
    assert m.detections == 0
    assert math.isnan(m.detection_precision)
    assert math.isnan(m.contamination)
    assert math.isnan(m.identity_purity)


def test_source_that_ends_early() -> None:
    m = evaluate_tracking(ScriptedSource(2), ScriptedTracker(), 10, warmup=0)
    assert m.frames == m.processed_frames == 2


def test_pool_sums_counts() -> None:
    m = evaluate_tracking(ScriptedSource(4), ScriptedTracker(), 4, warmup=0)
    pooled = TrackingMetrics.pool([m, m])
    assert pooled.runs == 2
    assert pooled.observations == 2 * m.observations
    assert pooled.contamination == pytest.approx(m.contamination)
    assert pooled.identity_purity == pytest.approx(m.identity_purity)
    with pytest.raises(ValueError, match="nothing"):
        TrackingMetrics.pool([])
    other = evaluate_tracking(ScriptedSource(4), ScriptedTracker(), 4, warmup=0, iou_threshold=0.5)
    with pytest.raises(ValueError, match="IoU"):
        TrackingMetrics.pool([m, other])


def test_on_frame_callback_sees_every_frame() -> None:
    seen: list[int] = []
    evaluate_tracking(
        ScriptedSource(4),
        ScriptedTracker(),
        4,
        warmup=2,
        on_frame=lambda f, r, t: seen.append(f.index),
    )
    assert seen == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_frames": 4, "warmup": 4}, "exceed"),
        ({"n_frames": 4, "warmup": -1}, "warmup"),
        ({"n_frames": 4, "warmup": 0, "coverage_threshold": 0.0}, "coverage"),
    ],
)
def test_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_tracking(ScriptedSource(4), ScriptedTracker(), **kwargs)  # type: ignore[arg-type]


def test_match_boxes_is_one_to_one_and_gated() -> None:
    pred = [(0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0), (100.0, 100.0, 110.0, 110.0)]
    truth = [(0.5, 0.5, 10.5, 10.5)]
    assert match_boxes(pred, truth, 0.3) in ([(0, 0)], [(1, 0)])
    assert match_boxes(pred[2:], truth, 0.3) == []
    assert match_boxes([], truth, 0.3) == []
    with pytest.raises(ValueError, match="iou_threshold"):
        match_boxes(pred, truth, 0.0)


def test_box_coverage_uses_the_union_of_fish() -> None:
    # One box around two touching fish is entirely "on fish", though neither
    # fish alone covers half of it.
    cov = box_coverage([(0.0, 0.0, 20.0, 10.0)], [(0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)])
    assert cov[0] == pytest.approx(1.0)
    overlapping = box_coverage(
        [(0.0, 0.0, 10.0, 10.0)], [(0.0, 0.0, 6.0, 10.0), (4.0, 0.0, 8.0, 10.0)]
    )
    assert overlapping[0] == pytest.approx(0.8)  # union, not sum
    assert box_coverage([(0.0, 0.0, 4.0, 4.0)], [])[0] == 0.0
    assert box_coverage([], [A]).shape == (0,)


# --------------------------------------------------------------------------
# regression targets on the seeded simulator (DESIGN.md rule 4: synthetic data)
# --------------------------------------------------------------------------


def simulate(seed: int, frames: int, **config: object) -> TrackingMetrics:
    source = SyntheticAquarium(SyntheticConfig(seed=seed, **config))  # type: ignore[arg-type]
    return evaluate_tracking(source, FishTracker(), frames, warmup=150)


def test_simulator_smoke_clutter_is_not_credited() -> None:
    """10 s of the default tank: few false detections, no clutter in the fish channel."""
    m = simulate(1, 300)
    assert m.detections_per_frame < 1.2 * m.fish_per_frame  # was ~3x before (18.6 for 6 fish)
    assert m.detection_precision > 0.85
    assert m.contamination < 0.03
    assert m.fish_recall > 0.75  # identification still settling 5-10 s in
    assert m.identities == 6


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2])
def test_day_targets(seed: int) -> None:
    """30 s of the default day tank: contamination < 3 %, fish recall >= 90 %."""
    m = simulate(seed, 900)
    assert m.contamination < 0.03
    assert m.fish_recall >= 0.90
    assert m.detection_precision > 0.9
    assert m.identities == 6  # one identity per fish, no plant got a name
    assert m.identity_purity > 0.95
    assert m.id_switches_per_minute < 60  # ~2-4 per fish and minute: occlusions, not churn


@pytest.mark.slow
def test_night_is_clean_but_anonymous() -> None:
    """IR night: still no clutter in the fish channel; no colour, hence no identities."""
    m = simulate(1, 900, mode="night")
    assert m.contamination < 0.03
    assert m.fish_recall > 0.5
    assert m.identities == 0
