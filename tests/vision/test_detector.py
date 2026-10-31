"""MotionDetector on synthetic aquarium scenes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from puffer.types import Detection
from puffer.vision.detector import MotionDetector, _Blob, _merge_overlapping, is_grayscale
from puffer.vision.geometry import boxes_to_array, iou_matrix

from .scenes import (
    BLUE,
    MAGENTA,
    ORANGE,
    YELLOW,
    FishSpec,
    Scene,
    circle,
    linear,
    nearest_truth,
    wander,
    waypoints,
)

WARMUP = 30


def centre(d: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = d.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def evaluate(
    scene: Scene,
    detector: MotionDetector,
    n_frames: int,
    *,
    min_visible: float = 0.8,
    skip: int = WARMUP + 15,
    on_frame: Callable[[list[Detection]], None] | None = None,
) -> tuple[float, float]:
    """Return (recall, false positives per frame), ignoring the first ``skip`` frames.

    A fish counts as found when some detection overlaps its true box with
    IoU >= 0.3; a detection is a false positive when it overlaps no fish at
    all (IoU < 0.05), so boxes on fish that are half out of the frame are
    neither hits nor false alarms. The first frames after the warm-up are
    skipped: fish that moved slowly during learning leave a smear that is
    absorbed once they swim away.
    """
    hits = misses = false_pos = frames = 0
    for frame, truth in scene.frames(n_frames):
        dets = detector.detect(frame)
        if frame.index < skip:
            continue
        frames += 1
        iou = iou_matrix(boxes_to_array([d.bbox for d in dets]), [g.bbox for g in truth])
        for j, gt in enumerate(truth):
            if gt.visible_frac >= min_visible:
                found = bool(iou.shape[0]) and float(np.max(iou[:, j])) >= 0.3
                hits += found
                misses += not found
        if iou.shape[1]:
            false_pos += int(np.sum(iou.max(axis=1) < 0.05))
        else:
            false_pos += len(dets)
        if on_frame is not None:
            on_frame(dets)
    recall = hits / (hits + misses) if hits + misses else 1.0
    return recall, false_pos / max(1, frames)


def three_fish(rng: np.random.Generator) -> list[FishSpec]:
    """Three fish in separate regions of a 320x240 tank (no fish-fish overlap)."""
    patrol = waypoints(
        [(0.0, (30.0, 45.0)), (3.5, (290.0, 50.0)), (7.0, (30.0, 40.0)), (10.5, (290.0, 45.0))]
    )
    return [
        FishSpec("a", ORANGE, 30, 12, patrol),
        FishSpec("b", BLUE, 26, 10, circle((90, 130), 45, 1.3)),
        FishSpec("c", YELLOW, 22, 9, wander(rng, (235, 140), (45, 30))),
    ]


def test_warmup_then_ready(rng: np.random.Generator) -> None:
    scene = Scene(three_fish(rng), seed=1)
    detector = MotionDetector(history_frames=WARMUP)
    for i in range(WARMUP):
        assert detector.detect(scene.render()[0]) == []
        assert detector.ready == (i == WARMUP - 1)
    assert detector.foreground_mask is None
    detector.detect(scene.render()[0])
    mask = detector.foreground_mask
    assert mask is not None
    assert mask.shape == (240, 320)
    assert mask.dtype == np.bool_
    bg = detector.background
    assert bg is not None
    assert bg.shape == (240, 320, 3)
    assert bg.dtype == np.uint8


def test_moving_fish_high_recall_few_false_positives(rng: np.random.Generator) -> None:
    scene = Scene(three_fish(rng), seed=2, noise_sigma=4.0)
    scores: list[float] = []

    def collect(dets: list[Detection]) -> None:
        scores.extend(d.score for d in dets)

    recall, fp_rate = evaluate(scene, MotionDetector(), 240, on_frame=collect)
    assert recall >= 0.98
    assert fp_rate <= 0.02
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert float(np.median(scores)) > 0.8  # clear fish are high-confidence for ByteTrack


def test_pure_noise_background_has_no_detections() -> None:
    scene = Scene([], seed=3, noise_sigma=6.0)
    recall, fp_rate = evaluate(scene, MotionDetector(), 150)
    assert recall == 1.0  # vacuous: nothing to find
    assert fp_rate == 0.0


def test_noise_adaptive_threshold() -> None:
    quiet = MotionDetector()
    noisy = MotionDetector()
    for frame, _ in Scene([], seed=4, noise_sigma=1.0).frames(40):
        quiet.detect(frame)
    for frame, _ in Scene([], seed=4, noise_sigma=30.0).frames(40):
        assert noisy.detect(frame) == []
    assert quiet.last_threshold == pytest.approx(quiet.threshold)
    assert noisy.last_threshold > quiet.last_threshold
    assert noisy.noise_sigma > quiet.noise_sigma


def test_resting_fish_stays_detected() -> None:
    """A fish swims in, rests for 10 s, and is detected the whole time."""
    path = waypoints([(0.0, (20.0, 100.0)), (2.0, (160.0, 100.0)), (12.0, (160.0, 100.0))])
    scene = Scene([FishSpec("sleepy", ORANGE, 30, 12, path)], seed=5)
    detector = MotionDetector()
    resting_hits = resting_frames = 0
    resting_scores: list[float] = []
    for frame, truth in scene.frames(int(12 * scene.fps)):
        dets = detector.detect(frame)
        if frame.timestamp >= 2.5:
            resting_frames += 1
            matches = [d for d in dets if nearest_truth(centre(d), truth, 6.0) is not None]
            resting_hits += bool(matches)
            resting_scores.extend(d.score for d in matches)
    assert resting_hits / resting_frames >= 0.99
    # It slowly fades into the background but stays a usable detection.
    assert min(resting_scores) > 0.5


def test_ghost_from_warmup_is_absorbed() -> None:
    """A fish resting during warm-up leaves a ghost when it swims off; it must vanish."""
    path = waypoints([(0.0, (80.0, 80.0)), (1.5, (80.0, 80.0)), (3.0, (260.0, 160.0))])
    scene = Scene([FishSpec("early", BLUE, 30, 12, path)], seed=6)
    detector = MotionDetector()
    ghost_frames = 0
    for frame, truth in scene.frames(150):
        dets = detector.detect(frame)
        if frame.timestamp > 2.5:
            ghost_frames += sum(nearest_truth(centre(d), truth, 10.0) is None for d in dets)
            assert any(nearest_truth(centre(d), truth, 8.0) for d in dets)
    assert ghost_frames == 0


def test_without_ghost_suppression_the_ghost_persists() -> None:
    """Same scene as above with the ghost test and morphology disabled: why both exist."""
    path = waypoints([(0.0, (80.0, 80.0)), (1.5, (80.0, 80.0)), (3.0, (260.0, 160.0))])
    scene = Scene([FishSpec("early", BLUE, 30, 12, path)], seed=6)
    detector = MotionDetector(ghost_edge_ratio=0.0, open_ksize=1, close_ksize=1, mask_margin=1)
    assert detector.background is None
    ghost_frames = 0
    for frame, truth in scene.frames(150):
        dets = detector.detect(frame)
        if frame.timestamp > 2.5:
            ghost_frames += any(nearest_truth(centre(d), truth, 10.0) is None for d in dets)
    assert ghost_frames > 50


def test_lighting_drift_and_flicker(rng: np.random.Generator) -> None:
    def lighting(t: float) -> float:
        return (1.0 - 0.03 * t) * (1.0 + 0.04 * math.sin(2 * math.pi * 2.0 * t))

    scene = Scene(three_fish(rng), seed=7, lighting=lighting)
    recall, fp_rate = evaluate(scene, MotionDetector(), 300)  # 10 s: 30 % dimmer at the end
    assert recall >= 0.98
    assert fp_rate <= 0.02


def test_grayscale_ir_frames(rng: np.random.Generator) -> None:
    fish = [
        FishSpec("bright", (230, 230, 230), 30, 12, linear((30, 60), (60, 20))),
        FishSpec("dark", (20, 20, 20), 28, 11, circle((170, 140), 50, 1.1)),
    ]
    scene = Scene(fish, seed=8, gray=True)
    detector = MotionDetector()
    recall, fp_rate = evaluate(scene, detector, 150)
    background = detector.background
    assert background is not None
    assert background.shape == (240, 320, 3)  # learned as one channel, shown as BGR
    assert is_grayscale(background)
    assert recall >= 0.98
    assert fp_rate <= 0.02


def test_switch_between_colour_and_ir_restarts_learning(rng: np.random.Generator) -> None:
    detector = MotionDetector()
    for frame, _ in Scene(three_fish(rng), seed=9).frames(40):
        detector.detect(frame)
    assert detector.ready
    ir_frame, _ = Scene(three_fish(rng), seed=9, gray=True).render()
    assert detector.detect(ir_frame) == []
    assert not detector.ready


def test_sudden_scene_change_relearns(rng: np.random.Generator) -> None:
    detector = MotionDetector(relearn_frames=3)
    scene = Scene([FishSpec("a", MAGENTA, 30, 12, linear((40, 60), (40, 20)))], seed=10)
    for frame, _ in scene.frames(40):
        detector.detect(frame)
    assert detector.ready
    # The camera is knocked over: the view has nothing in common with the background.
    template, _ = scene.render()
    for i in range(3):
        image = rng.integers(0, 256, size=template.image.shape, dtype=np.uint8)
        assert detector.detect(replace(template, index=template.index + i, image=image)) == []
    assert not detector.ready
    for frame, _ in Scene([], seed=99).frames(WARMUP):
        detector.detect(frame)
    assert detector.ready


def test_fish_leaving_the_frame() -> None:
    fish = [FishSpec("runner", YELLOW, 30, 12, linear((200, 120), (90, 0)))]
    detector = MotionDetector()
    seen_after_exit = 0
    for frame, truth in Scene(fish, seed=11).frames(120):
        dets = detector.detect(frame)
        if truth[0].visible_frac == 0.0:
            seen_after_exit += len(dets)
    assert seen_after_exit == 0


def test_area_filter() -> None:
    tiny = [FishSpec("tiny", ORANGE, 6, 4, linear((40, 60), (50, 10)))]
    recall, fp_rate = evaluate(Scene(tiny, seed=12), MotionDetector(min_area=150), 80)
    assert recall == 0.0
    assert fp_rate == 0.0
    swim_in = waypoints([(0.0, (-150.0, 120.0)), (3.0, (-150.0, 120.0)), (5.0, (160.0, 120.0))])
    whale = [FishSpec("whale", ORANGE, 200, 120, swim_in)]  # a quarter of the frame
    recall, fp_rate = evaluate(Scene(whale, seed=13), MotionDetector(max_area_frac=0.2), 200)
    assert recall == 0.0
    assert fp_rate == 0.0


def test_score_monotonic() -> None:
    detector = MotionDetector()
    assert detector._score(5.0, 600) > detector._score(1.5, 600)
    assert detector._score(5.0, 600) > detector._score(5.0, 70)
    assert detector._score(0.5, 600) == 0.0
    assert 0.0 <= detector._score(100.0, 1e6) <= 1.0


def test_merge_overlapping_boxes() -> None:
    blobs = [
        _Blob(10, 10, 40, 30, 0.6),
        _Blob(30, 12, 45, 28, 0.9),  # mostly inside the first box
        _Blob(100, 100, 120, 110, 0.7),  # separate
    ]
    merged = sorted(_merge_overlapping(blobs, 0.6), key=lambda b: b.x1)
    assert merged == [_Blob(10, 10, 45, 30, 0.9), _Blob(100, 100, 120, 110, 0.7)]
    assert _merge_overlapping(blobs, 0.99) == blobs


def test_is_grayscale() -> None:
    gray = np.repeat(np.arange(64, dtype=np.uint8).reshape(8, 8, 1), 3, axis=2)
    assert is_grayscale(gray)
    assert is_grayscale(gray[..., 0])
    colour = gray.copy()
    colour[..., 2] = 255
    assert not is_grayscale(colour)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"history_frames": 0}, "history_frames"),
        ({"min_area": 0}, "min_area"),
        ({"max_area_frac": 0.0}, "max_area_frac"),
        ({"update_rate": 0.001, "foreground_update_rate": 0.01}, "foreground_update_rate"),
        ({"blur_ksize": 4}, "odd"),
    ],
)
def test_parameter_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MotionDetector(**kwargs)  # type: ignore[arg-type]
