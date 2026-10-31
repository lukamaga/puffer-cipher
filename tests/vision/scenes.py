"""Tiny synthetic aquarium scenes with exact ground truth for vision tests.

Deliberately independent of ``puffer.sources`` (the real simulator): the
vision tests must pin down detector/tracker behaviour against geometry we
control exactly. A scene is a static textured background, coloured
anti-aliased ellipses ("fish") moving along analytic trajectories, a global
lighting gain (drift / flicker) and Gaussian sensor noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from puffer.types import BBox, Frame, LightMode, Point

Path = Callable[[float], Point]
BGR = tuple[int, int, int]


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------


def linear(start: Point, velocity: Point) -> Path:
    """Constant velocity (px/s) from ``start`` at t = 0."""

    def path(t: float) -> Point:
        return (start[0] + velocity[0] * t, start[1] + velocity[1] * t)

    return path


def waypoints(points: Sequence[tuple[float, Point]]) -> Path:
    """Piecewise-linear path through ``(t, (x, y))`` knots; holds the ends."""
    ts = [p[0] for p in points]
    xs = [p[1][0] for p in points]
    ys = [p[1][1] for p in points]

    def path(t: float) -> Point:
        return (float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys)))

    return path


def circle(center: Point, radius: float, omega: float, phase: float = 0.0) -> Path:
    """Uniform circular motion, ``omega`` in rad/s."""

    def path(t: float) -> Point:
        a = omega * t + phase
        return (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))

    return path


def wander(rng: np.random.Generator, center: Point, amplitude: Point, n_terms: int = 3) -> Path:
    """Smooth pseudo-random path: a sum of sinusoids with random frequencies and phases."""
    freqs = rng.uniform(0.05, 0.4, size=(2, n_terms))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=(2, n_terms))
    weights = rng.dirichlet(np.ones(n_terms), size=2)

    def path(t: float) -> Point:
        s = np.sin(2.0 * math.pi * freqs * t + phases)
        dx, dy = (weights * s).sum(axis=1)
        return (center[0] + amplitude[0] * float(dx), center[1] + amplitude[1] * float(dy))

    return path


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FishSpec:
    name: str
    color: BGR
    length: float  # full body length in px (major axis)
    width: float  # full body width in px (minor axis)
    path: Path
    fin_color: BGR | None = None  # optional tail-half colour (two-tone fish)


@dataclass(frozen=True)
class FishTruth:
    name: str
    center: Point
    velocity: Point  # px / s
    bbox: BBox  # axis-aligned box of the ellipse, clipped to the frame
    visible_frac: float  # visible fraction of the unclipped box area


def textured_background(rng: np.random.Generator, width: int, height: int) -> npt.NDArray[np.uint8]:
    """Blue-green water with low-frequency mottling and a gravel strip at the bottom."""
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    water = np.array([150.0, 125.0, 60.0], dtype=np.float32) * (1.1 - 0.3 * yy)
    raw = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    mottling = np.asarray(cv2.GaussianBlur(raw, (0, 0), 9), dtype=np.float32)
    mottling *= np.float32(12.0 / max(float(mottling.std()), 1e-6))
    img = water + mottling[..., None]
    gravel_top = int(height * 0.85)
    gravel = rng.normal(0.0, 18.0, (height - gravel_top, width, 1)).astype(np.float32)
    img[gravel_top:] = np.array([95.0, 120.0, 135.0], dtype=np.float32) + gravel
    return np.clip(img, 0, 255).astype(np.uint8)


def _velocity(path: Path, t: float, h: float = 1e-3) -> Point:
    (x0, y0), (x1, y1) = path(t - h), path(t + h)
    return ((x1 - x0) / (2 * h), (y1 - y0) / (2 * h))


class Scene:
    """Deterministic frame generator.

    ``lighting(t)`` multiplies the whole image (lamps dim everything), then
    noise with standard deviation ``noise_sigma`` is added. ``gray=True``
    renders an IR-style frame (luminance replicated into three channels).
    ``occluders`` are ``(t_start, t_end, bbox)`` rectangles painted with the
    background - i.e. the fish swims *behind* a rock.
    """

    def __init__(
        self,
        fish: Sequence[FishSpec],
        *,
        width: int = 320,
        height: int = 240,
        fps: float = 30.0,
        noise_sigma: float = 3.0,
        seed: int = 0,
        lighting: Callable[[float], float] | None = None,
        gray: bool = False,
        occluders: Sequence[tuple[float, float, BBox]] = (),
    ) -> None:
        self.fish = list(fish)
        self.width, self.height, self.fps = width, height, fps
        self.noise_sigma = noise_sigma
        self.lighting = lighting if lighting is not None else (lambda t: 1.0)
        self.gray = gray
        self.occluders = list(occluders)
        self._rng = np.random.default_rng(seed)
        self.background = textured_background(self._rng, width, height)
        self._angles = {f.name: 0.0 for f in self.fish}
        self.index = 0

    def truth_at(self, t: float) -> list[FishTruth]:
        out = []
        for f in self.fish:
            cx, cy = f.path(t)
            vx, vy = _velocity(f.path, t)
            if math.hypot(vx, vy) > 1.0:
                self._angles[f.name] = math.atan2(vy, vx)
            th = self._angles[f.name]
            a, b = f.length / 2.0, f.width / 2.0
            hx = math.sqrt((a * math.cos(th)) ** 2 + (b * math.sin(th)) ** 2)
            hy = math.sqrt((a * math.sin(th)) ** 2 + (b * math.cos(th)) ** 2)
            full = (cx - hx, cy - hy, cx + hx, cy + hy)
            clipped = (
                max(0.0, full[0]),
                max(0.0, full[1]),
                min(float(self.width), full[2]),
                min(float(self.height), full[3]),
            )
            cw, ch = clipped[2] - clipped[0], clipped[3] - clipped[1]
            frac = 1.0 if clipped == full else max(0.0, cw) * max(0.0, ch) / (4.0 * hx * hy)
            out.append(FishTruth(f.name, (cx, cy), (vx, vy), clipped, frac))
        return out

    def render(self) -> tuple[Frame, list[FishTruth]]:
        """Render the next frame and its ground truth."""
        t = self.index / self.fps
        truth = self.truth_at(t)
        img = self.background.copy()
        for f, gt in zip(self.fish, truth, strict=True):
            if gt.visible_frac <= 0.0:
                continue
            self._draw_fish(img, f, gt.center, self._angles[f.name])
        for t0, t1, (x1, y1, x2, y2) in self.occluders:
            if t0 <= t < t1:
                ys, xs = slice(int(y1), int(y2)), slice(int(x1), int(x2))
                img[ys, xs] = self.background[ys, xs]
        out = img.astype(np.float32) * float(self.lighting(t))
        if self.noise_sigma > 0:
            noise = self._rng.standard_normal(out.shape, dtype=np.float32)
            out += noise * np.float32(self.noise_sigma)
        frame_img = np.clip(out, 0, 255).astype(np.uint8)
        if self.gray:
            luminance = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            frame_img = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)
        frame = Frame(
            index=self.index,
            timestamp=t,
            wall_time=1_793_454_000.0 + t,
            image=frame_img,
            mode=LightMode.NIGHT if self.gray else LightMode.DAY,
            source_id="test-scene",
        )
        self.index += 1
        return frame, truth

    def frames(self, n: int) -> Iterator[tuple[Frame, list[FishTruth]]]:
        for _ in range(n):
            yield self.render()

    @staticmethod
    def _draw_fish(img: npt.NDArray[np.uint8], f: FishSpec, center: Point, angle: float) -> None:
        shift = 4
        scale = 1 << shift
        c = (round(center[0] * scale), round(center[1] * scale))
        axes = (round(f.length / 2 * scale), round(f.width / 2 * scale))
        deg = math.degrees(angle)
        cv2.ellipse(img, c, axes, deg, 0, 360, f.color, -1, cv2.LINE_AA, shift)
        if f.fin_color is not None:
            cv2.ellipse(img, c, axes, deg, 90, 270, f.fin_color, -1, cv2.LINE_AA, shift)


# --------------------------------------------------------------------------
# matching helpers
# --------------------------------------------------------------------------


def nearest_truth(point: Point, truth: Sequence[FishTruth], max_dist: float) -> FishTruth | None:
    """Ground-truth fish closest to ``point`` within ``max_dist`` pixels."""
    best: FishTruth | None = None
    best_d = max_dist
    for gt in truth:
        d = math.hypot(point[0] - gt.center[0], point[1] - gt.center[1])
        if d <= best_d:
            best, best_d = gt, d
    return best


ORANGE: BGR = (30, 120, 245)
BLUE: BGR = (200, 60, 20)
YELLOW: BGR = (40, 220, 235)
MAGENTA: BGR = (190, 40, 210)
WHITE: BGR = (235, 235, 235)
