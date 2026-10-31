"""SyntheticAquarium: a living, deterministic aquarium simulator.

No real aquarium is attached to a CI runner, so the simulator is the
project's demo source and its test bench: it is what the README GIF shows,
what the vision tracker and the LED liveness verifier are tested against,
and what the end-to-end pipeline runs on by default.

It therefore has to behave like a camera pointed at real fish:

* **Fish behave like fish** (see :mod:`puffer.sources._school`): each fish
  follows an Ornstein-Uhlenbeck velocity process around a wandering
  preferred heading, with Poisson-timed darts and pauses, soft avoidance of
  walls, surface, sand, the LED housing and each other, a species-specific
  preferred depth, and sleep at night.
* **The tank looks like a tank** (see :mod:`puffer.sources._scene`): water
  gradient and light shafts, textured sand with pebbles, rocks, driftwood,
  plants swaying in the current in front of and behind the fish, a
  bubbler, physically motivated caustics (refracted rays focusing on the
  floor) and a lens vignette.
* **Fish look like fish** (see :mod:`puffer.sources._fish_art`): eight
  species with shaded bodies, patterns, translucent fins and tail beats.
* **The camera is a camera** (see :mod:`puffer.sources._sensor`): Poisson
  shot noise plus Gaussian read noise, optional JPEG round trip, and an IR
  night mode (monochrome, dimmer, noisier) that the day/night cycle
  switches into like a real NoIR camera with an IR illuminator.
* **The liveness LED is in the picture**: a small dot at ``led_xy`` shows
  ``led_schedule.state_at(timestamp - led_latency_s)``, i.e. the commanded
  challenge as seen through a camera with latency.

Time is simulated: ``Frame.timestamp = start_time + index / fps``. The LED
schedule is evaluated on the same clock, so liveness verification works
whether or not the simulator runs in real time. All randomness comes from
one seed (``SyntheticConfig.seed``) split into independent streams for
behaviour, bubbles and sensor noise, so a fixed seed reproduces every frame
bit-for-bit. The simulator's "entropy" is therefore pseudo-random by
construction; reports must say so (DESIGN.md rule 4).
"""

from __future__ import annotations

import colorsys
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt

from puffer.sources import _fish_art, _scene, _school, _sensor
from puffer.sources.base import FrameSourceBase, RealtimePacer
from puffer.types import BBox, Frame, LedSchedule, LightMode, Point

__all__ = [
    "DEFAULT_CAST",
    "REFERENCE_WIDTH",
    "SPECIES",
    "FishSpec",
    "FishTruth",
    "SyntheticAquarium",
    "SyntheticConfig",
]

REFERENCE_WIDTH = 640
"""Frame width at which ``FishSpec`` sizes and speeds are specified."""

SPECIES: tuple[str, ...] = tuple(_fish_art.ANATOMY)
"""Species the renderer knows how to draw."""

Bgr = tuple[int, int, int]
U8 = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class FishSpec:
    """One fish of the cast: identity, looks and temperament.

    ``length_px`` and ``speed_px_s`` (cruising speed) are given for a
    :data:`REFERENCE_WIDTH`-pixel-wide frame and scale with the frame width,
    so the same tank looks the same at any resolution. ``burstiness`` (0..1)
    sets how often and how hard the fish darts, ``depth_pref`` (0 = surface,
    1 = sand) the depth it likes to cruise at, and ``wander`` how quickly its
    preferred heading drifts (radians per sqrt-second).
    """

    fish_id: str
    name: str
    species: str
    body_bgr: Bgr
    accent_bgr: Bgr
    length_px: float
    speed_px_s: float
    burstiness: float = 0.4
    depth_pref: float = 0.5
    wander: float = 0.6

    def __post_init__(self) -> None:
        if self.species not in _fish_art.ANATOMY:
            raise ValueError(f"unknown species {self.species!r}; choose from {', '.join(SPECIES)}")
        if not 8.0 <= self.length_px <= 400.0:
            raise ValueError("length_px must be within [8, 400]")
        if not self.speed_px_s > 0.0:
            raise ValueError("speed_px_s must be > 0")
        if not 0.0 <= self.burstiness <= 1.0:
            raise ValueError("burstiness must be within [0, 1]")
        if not 0.0 <= self.depth_pref <= 1.0:
            raise ValueError("depth_pref must be within [0, 1]")
        if self.wander < 0.0:
            raise ValueError("wander must be >= 0")
        for colour in (self.body_bgr, self.accent_bgr):
            if len(colour) != 3 or not all(0 <= c <= 255 for c in colour):
                raise ValueError(f"colours are (b, g, r) tuples in 0..255, got {colour}")

    @property
    def color_hex(self) -> str:
        """Body colour as ``#rrggbb`` (the dashboard's fish colour)."""
        b, g, r = self.body_bgr
        return f"#{r:02x}{g:02x}{b:02x}"


DEFAULT_CAST: tuple[FishSpec, ...] = (
    FishSpec("f1", "Biscuit", "clownfish", (20, 112, 255), (250, 250, 250), 58, 48, 0.5, 0.62),
    FishSpec("f2", "Captain Fin", "blue_tang", (230, 95, 25), (0, 205, 255), 70, 60, 0.35, 0.45),
    FishSpec(
        "f3", "Professor Puff", "pufferfish", (96, 178, 196), (35, 70, 105), 56, 30, 0.15, 0.78
    ),
    FishSpec("f4", "Sunny", "yellow_tang", (10, 214, 255), (240, 245, 245), 64, 55, 0.3, 0.38),
    FishSpec("f5", "Duchess", "betta", (55, 20, 200), (230, 95, 70), 64, 32, 0.2, 0.22, 0.8),
    FishSpec("f6", "Moonbeam", "angelfish", (205, 214, 214), (45, 40, 40), 66, 34, 0.15, 0.5, 0.4),
    FishSpec("f7", "Zap", "neon_tetra", (255, 205, 0), (50, 25, 225), 34, 72, 0.8, 0.32, 1.0),
    FishSpec("f8", "Pixel", "guppy", (165, 175, 160), (30, 110, 255), 40, 52, 0.6, 0.25, 0.9),
)
"""Built-in cast, most photogenic first (the default tank shows six)."""

Mode = Literal["day", "night", "cycle"]
_MODES: tuple[str, ...] = ("day", "night", "cycle")
_MAX_FISH = 32
_DUSK_LEVEL = 0.4  # room light when the camera switches to IR


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Everything that defines a simulated tank and camera.

    ``seed=None`` seeds from ``os.urandom`` (a fresh tank each run); a fixed
    seed reproduces every frame exactly. ``mode`` is ``"day"``, ``"night"``
    or ``"cycle"`` (``cycle_day_s`` of day, of which the last
    ``transition_s`` dim like dusk, then ``cycle_night_s`` of IR night).
    Noise sigmas are the Gaussian read noise in digital numbers; with
    ``shot_noise`` the variance also grows by ``shot_gain_* x intensity``.
    ``led_latency_s`` is how late the camera sees the LED (capture +
    processing delay). ``plant_sway`` scales the water current that sways
    the plants (0 = perfectly still plants, handy to benchmark a tracker
    without background clutter). ``realtime`` paces :meth:`SyntheticAquarium.read` to
    ``fps``; the frame clock stays simulated
    (``start_time + index / fps``), so to share it with a real-time LED
    controller pass ``start_time=time.monotonic()``.
    """

    width: int = 640
    height: int = 480
    fps: float = 30.0
    n_fish: int = 6
    fish: Sequence[FishSpec] | None = None
    seed: int | None = None
    mode: Mode = "day"
    cycle_day_s: float = 120.0
    cycle_night_s: float = 60.0
    noise_sigma_day: float = 2.0
    noise_sigma_night: float = 6.0
    shot_noise: bool = True
    jpeg_quality: int | None = None
    led_xy: tuple[int, int] = (608, 32)
    led_radius: int = 5
    led_latency_s: float = 0.05
    bubbles: bool = True
    caustics: bool = True
    realtime: bool = False
    start_time: float = 0.0
    shot_gain_day: float = 0.05
    shot_gain_night: float = 0.15
    transition_s: float = 8.0
    plant_sway: float = 1.0

    def __post_init__(self) -> None:
        if self.fish is not None:
            object.__setattr__(self, "fish", tuple(self.fish))
        if not (160 <= self.width <= 4096 and 120 <= self.height <= 4096):
            raise ValueError("frame size must be within 160x120 .. 4096x4096")
        if not 0.5 <= self.fps <= 240.0:
            raise ValueError("fps must be within [0.5, 240]")
        if not 0 <= self.n_fish <= _MAX_FISH:
            raise ValueError(f"n_fish must be within [0, {_MAX_FISH}]")
        if self.fish is not None and len(self.fish) > _MAX_FISH:
            raise ValueError(f"at most {_MAX_FISH} fish")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {', '.join(_MODES)}, got {self.mode!r}")
        if self.cycle_day_s <= 0 or self.cycle_night_s <= 0 or self.transition_s < 0:
            raise ValueError("cycle durations must be > 0 and transition_s >= 0")
        if (
            min(
                self.noise_sigma_day,
                self.noise_sigma_night,
                self.shot_gain_day,
                self.shot_gain_night,
            )
            < 0
        ):
            raise ValueError("noise parameters must be >= 0")
        if self.jpeg_quality is not None and not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be within [1, 100] or None")
        x, y = self.led_xy
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(
                f"led_xy {self.led_xy} is outside the {self.width}x{self.height} frame"
            )
        if not 1 <= self.led_radius <= 64:
            raise ValueError("led_radius must be within [1, 64]")
        if self.led_latency_s < 0:
            raise ValueError("led_latency_s must be >= 0")
        if not 0.0 <= self.plant_sway <= 5.0:
            raise ValueError("plant_sway must be within [0, 5]")

    @property
    def cast(self) -> tuple[FishSpec, ...]:
        """The fish in this tank: ``fish`` if given, else ``n_fish`` of the default cast."""
        if self.fish is not None:
            return tuple(self.fish)
        return tuple(_cast_member(i) for i in range(self.n_fish))


def _roman(n: int) -> str:
    numerals = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out


def _shift_hue(bgr: Bgr, degrees: float) -> Bgr:
    b, g, r = (c / 255.0 for c in bgr)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb((h + degrees / 360.0) % 1.0, lightness, s)
    return round(b2 * 255), round(g2 * 255), round(r2 * 255)


def _cast_member(i: int) -> FishSpec:
    """Fish ``i`` of an arbitrarily large tank: the default cast, then
    colour variants ("Biscuit II", ...) so every fish stays distinguishable."""
    base = DEFAULT_CAST[i % len(DEFAULT_CAST)]
    generation = i // len(DEFAULT_CAST)
    if generation == 0:
        return base
    shift = 40.0 * generation
    return replace(
        base,
        fish_id=f"f{i + 1}",
        name=f"{base.name} {_roman(generation + 1)}",
        body_bgr=_shift_hue(base.body_bgr, shift),
        accent_bgr=_shift_hue(base.accent_bgr, shift),
        length_px=base.length_px * (0.92 + 0.04 * (generation % 3)),
    )


@dataclass(frozen=True, slots=True)
class FishTruth:
    """Ground truth for one fish in the most recent frame.

    ``bbox`` covers the visible fish (alpha above 0.35, fins included) in
    pixels; ``center`` is its reference point; ``heading`` is
    ``atan2(vy, vx)`` of its velocity in radians and ``speed`` is in px/s.
    """

    fish_id: str
    name: str
    bbox: BBox
    center: Point
    heading: float
    speed: float


class SyntheticAquarium(FrameSourceBase):
    """Endless frames of the simulated tank; implements ``types.FrameSource``.

    ``led_schedule`` is the liveness challenge to display (``None``: the LED
    stays dark). It is sampled at ``timestamp - led_latency_s``: the camera
    sees the LED slightly late, which the verifier must estimate. Pass a
    :class:`puffer.hardware.SimulatedLed` driven by a ``LedController`` to
    close the full loop, or the challenge schedule itself.
    """

    def __init__(
        self, config: SyntheticConfig | None = None, led_schedule: LedSchedule | None = None
    ) -> None:
        cfg = config if config is not None else SyntheticConfig()
        self.config = cfg
        self.source_id = "synthetic"
        self.fps = float(cfg.fps)
        self.led_schedule = led_schedule
        self.seed = cfg.seed if cfg.seed is not None else int.from_bytes(os.urandom(16), "big")
        behaviour_ss, bubble_ss, sensor_ss = np.random.SeedSequence(self.seed).spawn(3)

        self._scene = _scene.build_static_scene(cfg.width, cfg.height, REFERENCE_WIDTH)
        layout = self._scene.layout
        self._layout = layout
        self._plants = _scene.build_plants(cfg.width, cfg.height, REFERENCE_WIDTH)
        self._caustics = (
            _scene.build_caustics(cfg.width, cfg.height, REFERENCE_WIDTH) if cfg.caustics else None
        )
        self._bubbles = (
            _scene.Bubbles(layout, np.random.default_rng(bubble_ss)) if cfg.bubbles else None
        )
        self._lamp = _scene.LedLamp(cfg.led_xy, cfg.led_radius)
        self._ir = _scene.ir_converter(self._scene)

        self._specs = cfg.cast
        self._sprites = [
            _fish_art.build_sprite(replace(spec, length_px=spec.length_px * layout.scale))
            for spec in self._specs
        ]
        x0, y0, x1, y1 = layout.swim_bounds
        led_keep_out = (float(cfg.led_xy[0]), float(cfg.led_xy[1]), 3.0 * cfg.led_radius)
        self._school = _school.School(
            self._specs,
            _school.Box(x0, y0, x1, y1),
            scale=layout.scale,
            rng=np.random.default_rng(behaviour_ss),
            keep_out=[led_keep_out],
            awake=self.light_at(cfg.start_time),
        )

        sensor_rng = np.random.default_rng(sensor_ss)
        self._sensor_day = _sensor.SensorModel(
            sensor_rng,
            read_sigma=cfg.noise_sigma_day,
            shot_gain=cfg.shot_gain_day,
            shot_noise=cfg.shot_noise,
            jpeg_quality=cfg.jpeg_quality,
        )
        self._sensor_night = _sensor.SensorModel(
            sensor_rng,
            read_sigma=cfg.noise_sigma_night,
            shot_gain=cfg.shot_gain_night,
            shot_noise=cfg.shot_noise,
            jpeg_quality=cfg.jpeg_quality,
        )
        self._pacer = RealtimePacer(cfg.fps) if cfg.realtime else None
        self._index = 0
        self._truth: list[FishTruth] = []
        self._closed = False

    # ------------------------------------------------------------ properties

    @property
    def fish_specs(self) -> tuple[FishSpec, ...]:
        """The cast, in ``fish_id`` order."""
        return self._specs

    @property
    def led_roi(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` of the LED and its brightest halo."""
        return self._lamp.roi

    @property
    def tank_bounds(self) -> BBox:
        """``(x1, y1, x2, y2)`` of the water region fish swim in."""
        return self._layout.swim_bounds

    @property
    def frame_index(self) -> int:
        """Index of the next frame :meth:`read` will return."""
        return self._index

    # ----------------------------------------------------------- day / night

    def mode_at(self, t: float) -> LightMode:
        """Camera mode at simulated time ``t``."""
        cfg = self.config
        if cfg.mode == "cycle":
            phase = (t - cfg.start_time) % (cfg.cycle_day_s + cfg.cycle_night_s)
            return LightMode.DAY if phase < cfg.cycle_day_s else LightMode.NIGHT
        return LightMode.NIGHT if cfg.mode == "night" else LightMode.DAY

    def light_at(self, t: float) -> float:
        """Room light level in [0, 1] at time ``t`` (0 = lights off, IR only).

        Lights switch on at full brightness (aquarium timers do), and dim
        over the last ``transition_s`` of the day before the camera switches
        to its IR mode.
        """
        cfg = self.config
        if cfg.mode != "cycle":
            return 0.0 if cfg.mode == "night" else 1.0
        phase = (t - cfg.start_time) % (cfg.cycle_day_s + cfg.cycle_night_s)
        if phase >= cfg.cycle_day_s:
            return 0.0
        remaining = cfg.cycle_day_s - phase
        span = min(cfg.transition_s, cfg.cycle_day_s / 2.0)
        if span <= 0.0 or remaining >= span:
            return 1.0
        u = remaining / span
        return _DUSK_LEVEL + (1.0 - _DUSK_LEVEL) * u * u * (3.0 - 2.0 * u)

    # ------------------------------------------------------------- frames

    def read(self) -> Frame | None:
        """Simulate one frame period and return the rendered frame."""
        if self._closed:
            return None
        if self._pacer is not None:
            self._pacer.wait()
        cfg = self.config
        index = self._index
        t = cfg.start_time + index / cfg.fps
        light = self.light_at(t)
        if index > 0:
            dt = 1.0 / cfg.fps
            self._school.step(dt, light)
            if self._bubbles is not None:
                self._bubbles.step(dt)
        mode = self.mode_at(t)
        image = self._render(t, light, mode)
        self._index += 1
        return Frame(
            index=index,
            timestamp=t,
            wall_time=time.time(),
            image=image,
            mode=mode,
            source_id=self.source_id,
        )

    def ground_truth(self) -> list[FishTruth]:
        """Fish positions in the most recent frame (empty before the first)."""
        return list(self._truth)

    def close(self) -> None:
        """Stop producing frames; :meth:`read` then returns ``None``."""
        self._closed = True

    def _render(self, t: float, light: float, mode: LightMode) -> npt.NDArray[np.uint8]:
        night = mode is LightMode.NIGHT
        scene = self._scene
        img = (scene.bg_night if night else scene.bg_day).copy()
        if self._caustics is not None and not night:
            self._caustics.apply(img, t, scene.caustic_gain)
        self._plants.draw(img, t, front=False, sway=self.config.plant_sway)
        self._truth = self._draw_fish(img)
        if self._bubbles is not None:
            self._bubbles.draw(img)
        self._plants.draw(img, t, front=True, sway=self.config.plant_sway)
        _scene.shimmer_surface(img, self._layout, t)

        schedule = self.led_schedule
        led_on = (
            bool(schedule.state_at(t - self.config.led_latency_s))
            if schedule is not None
            else False
        )
        if night:
            gray = self._ir(img)
            self._lamp.draw(gray, led_on)
            return cast(U8, cv2.cvtColor(self._sensor_night.apply(gray), cv2.COLOR_GRAY2BGR))
        cv2.multiply(img, scene.vignette, dst=img, scale=1.0 / 255.0)
        if light < 1.0:
            img = cast(U8, cv2.convertScaleAbs(img, alpha=light))
        self._lamp.draw(img, led_on)
        return self._sensor_day.apply(img)

    def _draw_fish(self, img: npt.NDArray[np.uint8]) -> list[FishTruth]:
        school = self._school
        if school.n == 0:
            return []
        boxes: dict[int, BBox] = {}
        water = self._scene.water_bgr
        for i in np.argsort(-school.z, kind="stable"):  # far fish first
            z = float(school.z[i])
            cx, cy = float(school.pos[i, 0]), float(school.pos[i, 1])
            row = min(len(water) - 1, max(0, round(cy)))
            box = _fish_art.draw_sprite(
                img,
                self._sprites[i],
                phase=float(school.phase[i]),
                center=(cx, cy),
                facing=float(school.facing[i]),
                pitch=float(school.pitch[i]),
                scale=1.0 - 0.18 * z,
                haze=0.3 * z,
                haze_bgr=water[row],
            )
            if box is not None:
                boxes[int(i)] = box
        speed, heading = school.speed, school.heading
        return [
            FishTruth(
                fish_id=spec.fish_id,
                name=spec.name,
                bbox=boxes[i],
                center=(float(school.pos[i, 0]), float(school.pos[i, 1])),
                heading=float(heading[i]),
                speed=float(speed[i]),
            )
            for i, spec in enumerate(self._specs)
            if i in boxes
        ]
