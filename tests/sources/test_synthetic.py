"""Tests for the aquarium simulator (behaviour, rendering, LED, determinism, speed)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any, Literal, cast

import cv2
import numpy as np
import pytest

from puffer.hardware.led import SimulatedLed
from puffer.sources import _fish_art
from puffer.sources.synthetic import (
    DEFAULT_CAST,
    SPECIES,
    FishSpec,
    FishTruth,
    SyntheticAquarium,
    SyntheticConfig,
)
from puffer.types import FrameSource, LedSchedule, LightMode

QuietConfig = Callable[..., SyntheticConfig]
LedFactory = Callable[[float], LedSchedule]


def _mode(name: str) -> Literal["day", "night", "cycle"]:
    assert name in ("day", "night", "cycle")
    return cast(Literal["day", "night", "cycle"], name)


def _read(source: SyntheticAquarium, n: int) -> list[np.ndarray]:
    frames = []
    for _ in range(n):
        frame = source.read()
        assert frame is not None
        frames.append(frame.image)
    return frames


def _roi_mean(image: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    x, y, w, h = roi
    return float(image[y : y + h, x : x + w].mean())


# --------------------------------------------------------------------------- contract


def test_implements_frame_source_protocol() -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=1))
    assert isinstance(aquarium, FrameSource)
    assert aquarium.source_id == "synthetic"
    assert aquarium.fps == 30.0


def test_frames_follow_the_frame_contract() -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=1, fps=25.0, start_time=10.0))
    before = time.time()
    for i in range(5):
        frame = aquarium.read()
        assert frame is not None
        assert frame.index == i
        assert frame.timestamp == pytest.approx(10.0 + i / 25.0, abs=1e-12)
        assert frame.image.shape == (480, 640, 3)
        assert frame.image.dtype == np.uint8
        assert frame.image.flags.c_contiguous
        assert frame.mode is LightMode.DAY
        assert frame.source_id == "synthetic"
        assert before - 1.0 <= frame.wall_time <= time.time() + 1.0
    assert aquarium.frame_index == 5


def test_custom_resolution_scales_the_scene() -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=1, width=320, height=240, led_xy=(300, 16)))
    frame = aquarium.read()
    assert frame is not None
    assert frame.image.shape == (240, 320, 3)
    for truth in aquarium.ground_truth():
        x1, y1, x2, y2 = truth.bbox
        assert 0 <= x1 < x2 <= 320
        assert 0 <= y1 < y2 <= 240


def test_close_ends_the_stream() -> None:
    with SyntheticAquarium(SyntheticConfig(seed=1)) as aquarium:
        assert aquarium.read() is not None
    assert aquarium.read() is None


# ---------------------------------------------------------------------- determinism


def test_fixed_seed_reproduces_frames_and_ground_truth() -> None:
    a = SyntheticAquarium(SyntheticConfig(seed=42))
    b = SyntheticAquarium(SyntheticConfig(seed=42))
    for fa, fb in zip(_read(a, 6), _read(b, 6), strict=True):
        assert np.array_equal(fa, fb)
    assert a.ground_truth() == b.ground_truth()


def test_different_seeds_give_different_tanks() -> None:
    a = _read(SyntheticAquarium(SyntheticConfig(seed=1)), 2)[-1]
    b = _read(SyntheticAquarium(SyntheticConfig(seed=2)), 2)[-1]
    assert not np.array_equal(a, b)


def test_unseeded_simulator_draws_its_seed_from_the_os() -> None:
    a, b = SyntheticAquarium(SyntheticConfig()), SyntheticAquarium(SyntheticConfig())
    assert a.seed != b.seed
    assert a.seed.bit_length() > 64


# ------------------------------------------------------------------------ day/night


def test_night_frames_are_darker_and_monochrome() -> None:
    day = _read(SyntheticAquarium(SyntheticConfig(seed=3)), 3)[-1]
    night_source = SyntheticAquarium(SyntheticConfig(seed=3, mode="night"))
    frame = None
    for _ in range(3):
        frame = night_source.read()
    assert frame is not None
    assert frame.mode is LightMode.NIGHT
    night = frame.image
    assert np.array_equal(night[..., 0], night[..., 1])
    assert np.array_equal(night[..., 1], night[..., 2])
    assert night.mean() < 0.75 * day.mean()
    chroma = np.abs(day.astype(int)[..., 0] - day.astype(int)[..., 2]).mean()
    assert chroma > 20  # the day frame really is in colour


def test_cycle_mode_switches_between_day_and_night() -> None:
    config = SyntheticConfig(
        seed=1, mode="cycle", cycle_day_s=1.0, cycle_night_s=0.5, transition_s=0.4, fps=10.0
    )
    aquarium = SyntheticAquarium(config)
    modes = []
    for _ in range(30):
        frame = aquarium.read()
        assert frame is not None
        modes.append(frame.mode)
    expected = ([LightMode.DAY] * 10 + [LightMode.NIGHT] * 5) * 2
    assert modes == expected
    # Lights dim before the camera switches to IR, and come back on at once.
    assert aquarium.light_at(0.0) == 1.0
    assert aquarium.light_at(0.7) < 1.0
    assert aquarium.light_at(0.95) < aquarium.light_at(0.8)
    assert aquarium.light_at(1.2) == 0.0
    assert aquarium.light_at(1.5) == 1.0


def test_dusk_frames_get_darker() -> None:
    config = SyntheticConfig(
        seed=1, mode="cycle", cycle_day_s=2.0, cycle_night_s=1.0, transition_s=1.0, fps=10.0
    )
    frames = _read(SyntheticAquarium(config), 20)
    assert frames[19].mean() < 0.7 * frames[5].mean()


# ------------------------------------------------------------------------------ LED


@pytest.mark.parametrize("mode", ["day", "night"])
def test_led_shows_the_schedule_seen_through_camera_latency(
    mode: str, periodic_led: LedFactory
) -> None:
    latency, period, fps = 0.1, 0.3, 30.0
    schedule = periodic_led(period)
    aquarium = SyntheticAquarium(
        SyntheticConfig(seed=5, mode=_mode(mode), led_latency_s=latency, fps=fps),
        led_schedule=schedule,
    )
    brightness, delayed, immediate = [], [], []
    for _ in range(54):
        frame = aquarium.read()
        assert frame is not None
        brightness.append(_roi_mean(frame.image, aquarium.led_roi))
        delayed.append(schedule.state_at(frame.timestamp - latency))
        immediate.append(schedule.state_at(frame.timestamp))
    b = np.array(brightness)
    threshold = (b.min() + b.max()) / 2.0
    observed = list(b > threshold)
    assert observed == delayed
    assert observed != immediate  # the latency is really applied
    on, off = b[np.array(delayed)], b[~np.array(delayed)]
    assert on.min() - off.max() > 40.0  # a clean two-cluster signal


def test_led_is_dark_without_a_schedule(periodic_led: LedFactory) -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=5))
    levels = [_roi_mean(image, aquarium.led_roi) for image in _read(aquarium, 10)]
    assert max(levels) - min(levels) < 5.0
    lit = SyntheticAquarium(
        SyntheticConfig(seed=5, start_time=1.0), led_schedule=periodic_led(1000.0)
    )
    assert _roi_mean(_read(lit, 1)[0], lit.led_roi) > max(levels) + 40.0


def test_led_roi_matches_configuration() -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=1, led_xy=(100, 50), led_radius=4))
    x, y, w, h = aquarium.led_roi
    assert (x + w // 2, y + h // 2) == (100, 50)
    assert w == h >= 2 * 4


def test_simulated_led_closes_the_loop() -> None:
    led = SimulatedLed()
    led.set(True, 0.2)
    led.set(False, 0.5)
    aquarium = SyntheticAquarium(
        SyntheticConfig(seed=2, fps=10.0, led_latency_s=0.0), led_schedule=led
    )
    levels = [_roi_mean(image, aquarium.led_roi) for image in _read(aquarium, 8)]
    lit = [level > (min(levels) + max(levels)) / 2 for level in levels]
    assert lit == [False, False, True, True, True, False, False, False]


# -------------------------------------------------------------------- ground truth


def test_ground_truth_stays_in_the_tank_and_fish_really_move() -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=11, n_fish=8, fps=15.0))
    assert aquarium.ground_truth() == []
    x0, y0, x1, y1 = aquarium.tank_bounds
    paths: dict[str, list[tuple[float, float]]] = {spec.fish_id: [] for spec in aquarium.fish_specs}
    speeds: dict[str, list[float]] = {spec.fish_id: [] for spec in aquarium.fish_specs}
    for _ in range(225):  # 15 s
        assert aquarium.read() is not None
        truth = aquarium.ground_truth()
        assert len(truth) == 8
        for fish in truth:
            bx1, by1, bx2, by2 = fish.bbox
            assert 0.0 <= bx1 < bx2 <= 640.0
            assert 0.0 <= by1 < by2 <= 480.0
            assert x0 <= fish.center[0] <= x1
            assert y0 <= fish.center[1] <= y1
            assert bx1 <= fish.center[0] <= bx2
            assert by1 <= fish.center[1] <= by2
            assert -math.pi <= fish.heading <= math.pi
            paths[fish.fish_id].append(fish.center)
            speeds[fish.fish_id].append(fish.speed)
    for spec in aquarium.fish_specs:
        path = np.array(paths[spec.fish_id])
        travelled = np.hypot(*np.diff(path, axis=0).T).sum()
        spread = path.max(axis=0) - path.min(axis=0)
        speed = np.array(speeds[spec.fish_id])
        assert travelled > 4.0 * spec.speed_px_s, spec.name
        assert spread.max() > 60.0, spec.name
        assert speed.std() > 0.2 * speed.mean(), spec.name  # darts, pauses, wandering


def test_fish_sleep_at_night() -> None:
    def mean_speed(mode: str) -> float:
        aquarium = SyntheticAquarium(SyntheticConfig(seed=9, mode=_mode(mode), fps=10.0))
        values = []
        for _ in range(150):
            aquarium.read()
            values += [fish.speed for fish in aquarium.ground_truth()]
        return float(np.mean(values))

    day, night = mean_speed("day"), mean_speed("night")
    assert night < 0.35 * day


def test_ground_truth_box_covers_the_rendered_fish(quiet_config: QuietConfig) -> None:
    cast = (DEFAULT_CAST[1],)
    empty = _read(SyntheticAquarium(quiet_config()), 1)[0]
    source = SyntheticAquarium(quiet_config(fish=cast))
    image = _read(source, 1)[0]
    (truth,) = source.ground_truth()
    changed = np.abs(image.astype(int) - empty.astype(int)).max(axis=2) > 25
    ys, xs = np.nonzero(changed)
    drawn = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    ix = max(0.0, min(drawn[2], truth.bbox[2]) - max(drawn[0], truth.bbox[0]))
    iy = max(0.0, min(drawn[3], truth.bbox[3]) - max(drawn[1], truth.bbox[1]))
    union = (drawn[2] - drawn[0]) * (drawn[3] - drawn[1]) + _area(truth.bbox) - ix * iy
    iou = ix * iy / union
    assert iou > 0.7
    assert truth.fish_id == cast[0].fish_id
    assert truth.name == cast[0].name


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def test_speed_and_heading_agree_with_the_motion() -> None:
    fps = 30.0
    aquarium = SyntheticAquarium(SyntheticConfig(seed=4, n_fish=3, fps=fps))
    aquarium.read()
    previous = aquarium.ground_truth()
    assert [t.fish_id for t in previous] == ["f1", "f2", "f3"]
    assert all(isinstance(t, FishTruth) for t in previous)
    ratios, angle_errors = [], []
    for _ in range(90):
        aquarium.read()
        current = aquarium.ground_truth()
        for before, now in zip(previous, current, strict=True):
            dx, dy = now.center[0] - before.center[0], now.center[1] - before.center[1]
            if now.speed > 15.0:
                ratios.append(math.hypot(dx, dy) * fps / now.speed)
                diff = math.atan2(dy, dx) - now.heading
                angle_errors.append(abs((diff + math.pi) % (2 * math.pi) - math.pi))
        previous = current
    assert np.median(ratios) == pytest.approx(1.0, abs=0.15)
    assert np.median(angle_errors) < 0.2


# -------------------------------------------------------------------------- cast


def test_default_cast_and_variants() -> None:
    assert [s.fish_id for s in SyntheticConfig().cast] == [s.fish_id for s in DEFAULT_CAST[:6]]
    big = SyntheticConfig(n_fish=20).cast
    assert len({s.fish_id for s in big}) == 20
    assert len({s.name for s in big}) == 20
    assert big[8].name == f"{DEFAULT_CAST[0].name} II"
    assert big[8].body_bgr != DEFAULT_CAST[0].body_bgr
    assert {s.species for s in DEFAULT_CAST} == set(SPECIES)
    assert len({s.body_bgr for s in DEFAULT_CAST}) == len(DEFAULT_CAST)


def test_explicit_fish_list_overrides_n_fish() -> None:
    fish = [DEFAULT_CAST[2], DEFAULT_CAST[6]]
    config = SyntheticConfig(n_fish=6, fish=fish)
    assert config.cast == tuple(fish)
    assert SyntheticAquarium(config).fish_specs == tuple(fish)


def test_fish_spec_validation_and_colour() -> None:
    spec = FishSpec("x", "Rex", "clownfish", (0, 128, 255), (255, 255, 255), 40, 30)
    assert spec.color_hex == "#ff8000"
    with pytest.raises(ValueError, match="unknown species"):
        FishSpec("x", "Rex", "shark", (0, 0, 0), (0, 0, 0), 40, 30)
    with pytest.raises(ValueError, match="burstiness"):
        FishSpec("x", "Rex", "guppy", (0, 0, 0), (0, 0, 0), 40, 30, burstiness=2.0)
    with pytest.raises(ValueError, match="colours"):
        FishSpec("x", "Rex", "guppy", (0, 0, 300), (0, 0, 0), 40, 30)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_fish": 33}, "n_fish"),
        ({"mode": "dusk"}, "mode"),
        ({"fps": 0.0}, "fps"),
        ({"width": 100}, "frame size"),
        ({"jpeg_quality": 0}, "jpeg_quality"),
        ({"led_xy": (700, 10)}, "outside"),
        ({"noise_sigma_day": -1.0}, "noise"),
        ({"led_latency_s": -0.1}, "led_latency_s"),
        ({"cycle_day_s": 0.0}, "cycle"),
        ({"fish": [DEFAULT_CAST[0]] * 33}, "at most"),
        ({"plant_sway": -1.0}, "plant_sway"),
        ({"led_radius": 0}, "led_radius"),
    ],
)
def test_config_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticConfig(**overrides)


@pytest.mark.parametrize("species", SPECIES)
def test_every_species_renders_an_animated_sprite(species: str) -> None:
    spec = FishSpec("x", "Rex", species, (40, 120, 200), (200, 80, 40), 48, 30)
    sprite = _fish_art.build_sprite(spec)
    assert len(sprite.frames) == _fish_art.PHASES
    first, half = sprite.frames[0], sprite.frames[_fish_art.PHASES // 2]
    assert first.dtype == np.float32
    assert first.shape[2] == 4
    assert np.min(first[..., 3]) >= 0.0
    assert np.max(first[..., 3]) <= 1.0 + 1e-6
    assert not np.allclose(first, half)  # the tail beats
    hull = sprite.hulls[0]
    assert np.ptp(hull[:, 0]) == pytest.approx(48, rel=0.25)


# ------------------------------------------------------------------- sensor noise


def _temporal_sigma(frames: list[np.ndarray], region: tuple[slice, slice]) -> float:
    a = frames[0][region].astype(float)
    b = frames[1][region].astype(float)
    return float(np.std(a - b) / math.sqrt(2.0))


def test_read_noise_matches_configuration(quiet_config: QuietConfig) -> None:
    # Clean open water: no fish, bubbles, caustics or plants in this region,
    # so frame-to-frame differences are pure sensor noise.
    region = (slice(70, 180), slice(200, 440))
    day = _read(SyntheticAquarium(quiet_config(noise_sigma_day=3.0)), 2)
    expected = math.sqrt(3.0**2 + 1.0 / 12.0)  # plus output rounding
    assert _temporal_sigma(day, region) == pytest.approx(expected, rel=0.08)
    night = _read(SyntheticAquarium(quiet_config(mode="night", noise_sigma_night=6.0)), 2)
    assert _temporal_sigma(night, region) == pytest.approx(math.sqrt(36.0 + 1.0 / 12.0), rel=0.08)


def test_noise_free_render_is_static(quiet_config: QuietConfig) -> None:
    region = (slice(70, 180), slice(200, 440))
    frames = _read(SyntheticAquarium(quiet_config()), 2)
    assert np.array_equal(frames[0][region], frames[1][region])


def test_plants_sway_unless_the_current_is_off(quiet_config: QuietConfig) -> None:
    def changed_pixels(sway: float) -> int:
        a, b = _read(SyntheticAquarium(quiet_config(plant_sway=sway, fps=5.0)), 2)
        surface = slice(0, 50)  # the meniscus shimmers regardless
        diff = np.abs(a.astype(int) - b.astype(int))
        diff[surface] = 0
        return int(np.count_nonzero(diff.max(axis=2) > 10))

    assert changed_pixels(0.0) == 0
    assert changed_pixels(1.0) > 200


def test_shot_noise_adds_signal_dependent_variance(quiet_config: QuietConfig) -> None:
    region = (slice(70, 180), slice(200, 440))
    read_only = _read(SyntheticAquarium(quiet_config(noise_sigma_day=2.0)), 2)
    with_shot = _read(
        SyntheticAquarium(quiet_config(noise_sigma_day=2.0, shot_noise=True, shot_gain_day=0.2)), 2
    )
    level = float(read_only[0][region].mean())
    expected = math.sqrt(4.0 + 0.2 * level + 1.0 / 12.0)
    assert _temporal_sigma(with_shot, region) == pytest.approx(expected, rel=0.1)
    assert _temporal_sigma(with_shot, region) > 1.3 * _temporal_sigma(read_only, region)


def test_jpeg_round_trip_option() -> None:
    plain = _read(SyntheticAquarium(SyntheticConfig(seed=8)), 1)[0]
    jpeg = _read(SyntheticAquarium(SyntheticConfig(seed=8, jpeg_quality=40)), 1)[0]
    assert jpeg.shape == plain.shape
    assert jpeg.dtype == np.uint8
    diff = np.abs(jpeg.astype(int) - plain.astype(int))
    assert 0.5 < diff.mean() < 15.0


# ------------------------------------------------------------------- performance


def test_realtime_mode_paces_frames() -> None:
    aquarium = SyntheticAquarium(SyntheticConfig(seed=1, fps=50.0, realtime=True))
    start = time.perf_counter()
    _read(aquarium, 6)
    assert time.perf_counter() - start >= 5 / 50.0 - 0.01


def test_render_speed_single_core() -> None:
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        aquarium = SyntheticAquarium(SyntheticConfig(seed=1))
        _read(aquarium, 30)  # warm caches
        n = 120
        start = time.perf_counter()
        _read(aquarium, n)
        fps = n / (time.perf_counter() - start)
    finally:
        cv2.setNumThreads(previous)
    print(f"\nsimulator render speed: {fps:.0f} fps at 640x480 on one core")
    assert fps >= 30.0


@pytest.mark.parametrize(("size", "expected"), [((640, 480), (608, 32)), ((320, 240), (304, 16))])
def test_default_led_position_scales_with_the_frame(size, expected):
    config = SyntheticConfig(seed=1, width=size[0], height=size[1])
    assert config.led_xy == expected
    assert config.led_center == expected
