"""The aquarium scenery: everything in the picture that is not a fish.

Rendering cost is split by how often things change:

* **Static, built once per frame size** (:class:`StaticScene`, cached):
  water gradient, light shafts, the distant back wall, textured sand with
  pebbles, dome-shaded rocks, driftwood, the bubbler and its air line, the
  water-surface meniscus, the vignette and the IR illuminator fall-off. The
  tank layout is fixed (a real aquarium does not rearrange its rocks between
  runs); only fish, bubbles and noise depend on the seed.
* **Cheap per-frame layers**: swaying plants (a few anti-aliased polygon
  fills with sub-pixel vertices), rising bubbles (pre-rendered sprites), the
  shimmer of the surface and the LED.
* **Caustics** come from a tiny physical model: light rays refracted by a
  periodic wave height field are binned where they hit the floor, so bright
  lines appear exactly where the refracted rays focus. The wave field is
  periodic in time, so a lazily filled cache of low-resolution frames makes
  the steady-state cost a single remap per frame.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

_SHIFT = 4  # cv2 fixed-point bits for sub-pixel polygon vertices
_ONE = 1 << _SHIFT
_SCENE_SEED = 0x5EA_F15  # fixed: the tank layout is part of the product, not the seed


def _rgb(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Readable colour constants: RGB in source, BGR in memory."""
    return float(b), float(g), float(r)


@dataclass(frozen=True, slots=True)
class TankLayout:
    """Where things are, for a frame of ``width`` x ``height`` pixels."""

    width: int
    height: int
    scale: float  # frame width / reference width
    surface_y: float  # water surface line
    sand_top: F64  # per column: y of the far edge of the sand bed
    bubbler: tuple[float, float]

    @property
    def swim_bounds(self) -> tuple[float, float, float, float]:
        """Region fish centres live in (before per-fish margins)."""
        return (
            2.0 * self.scale,
            self.surface_y + 6.0 * self.scale,
            self.width - 2.0 * self.scale,
            0.9 * self.height,
        )


def make_layout(width: int, height: int, reference_width: int) -> TankLayout:
    x = np.arange(width, dtype=np.float64) / width
    sand = height * (
        0.80 + 0.022 * np.sin(2 * np.pi * 1.2 * x + 0.8) + 0.01 * np.sin(2 * np.pi * 3.1 * x + 2.0)
    )
    bx = 0.72 * width
    return TankLayout(
        width=width,
        height=height,
        scale=width / reference_width,
        surface_y=0.075 * height,
        sand_top=sand,
        bubbler=(bx, float(sand[int(bx)]) + 0.03 * height),
    )


# ---------------------------------------------------------------------------
# Static background
# ---------------------------------------------------------------------------


def _blend(dst: F32, color: tuple[float, float, float] | F32, alpha: F32) -> None:
    """In-place ``dst = dst * (1 - alpha) + color * alpha`` (float images)."""
    a = alpha[..., None]
    dst *= 1.0 - a
    dst += np.asarray(color, dtype=np.float32) * a


def _poly_mask(height: int, width: int, polys: list[F64], blur: float = 0.0) -> F32:
    m = np.zeros((height, width), np.uint8)
    cv2.fillPoly(m, [np.round(p * _ONE).astype(np.int32) for p in polys], 255, cv2.LINE_AA, _SHIFT)
    out = m.astype(np.float32) / 255.0
    if blur > 0:
        out = cast(F32, cv2.GaussianBlur(out, (0, 0), blur))
    return out


def _value_noise(rng: np.random.Generator, height: int, width: int, cell: float) -> F32:
    """Smooth noise in [0, 1]: random lattice upsampled with cubic interpolation."""
    gh, gw = max(2, int(height / cell) + 2), max(2, int(width / cell) + 2)
    lattice = rng.random((gh, gw)).astype(np.float32)
    big = cv2.resize(lattice, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
    return np.clip(big[:height, :width], 0.0, 1.0)


def _smooth01(x: F32, e0: float, e1: float) -> F32:
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (u * u * (3.0 - 2.0 * u)).astype(np.float32)


def _rock_polygon(rng: np.random.Generator, cx: float, base_y: float, rx: float, ry: float) -> F64:
    """A boulder: noisy ellipse, flattened where it sits in the sand."""
    theta = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
    bumps = sum(
        rng.normal(0.0, 0.07 / k) * np.cos(k * theta + rng.uniform(0, 6.3)) for k in range(2, 7)
    )
    r = 1.0 + bumps
    x = cx + rx * r * np.cos(theta)
    y = base_y - ry * 0.55 + ry * r * np.sin(theta)
    y = np.minimum(y, base_y + 0.08 * ry)
    return np.stack([x, y], axis=1)


@dataclass(frozen=True, slots=True)
class StaticScene:
    """Pre-rendered layers for one frame size (read-only arrays)."""

    layout: TankLayout
    bg_day: U8
    bg_night: U8
    caustic_gain: U8  # how strongly caustics light each pixel
    vignette: U8  # (H, W, 3) multiplicative lens vignette
    ir_falloff: U8  # (H, W) IR illuminator hot spot
    water_bgr: F32  # (H, 3) water colour per row, for depth haze


def _paint_water(layout: TankLayout) -> F32:
    h, w = layout.height, layout.width
    img = np.zeros((h, w, 3), np.float32)
    y = np.arange(h, dtype=np.float32)
    depth = np.clip((y - layout.surface_y) / (0.82 * h - layout.surface_y), 0.0, 1.0)[:, None]
    top, mid, deep = (
        np.array(_rgb(70, 170, 190)),
        np.array(_rgb(28, 112, 145)),
        np.array(_rgb(14, 60, 92)),
    )
    upper = top * (1 - depth / 0.5) + mid * (depth / 0.5)
    lower = mid * (1 - (depth - 0.5) / 0.5) + deep * ((depth - 0.5) / 0.5)
    column = np.where(depth < 0.5, upper, lower).astype(np.float32)
    img[:] = column[:, None, :]
    # Light falls from the lamp above the middle of the tank.
    xs = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :]
    img *= (1.0 - 0.22 * xs * xs * 4.0)[..., None]
    # Air gap above the surface: dark lid with a dim reflection.
    air = int(layout.surface_y)
    gy = np.linspace(0.0, 1.0, air, dtype=np.float32)[:, None, None]
    img[:air] = (
        np.array(_rgb(26, 34, 38), np.float32) * (1 - gy)
        + np.array(_rgb(48, 70, 76), np.float32) * gy
    )
    return img


def _paint_light_shafts(layout: TankLayout, rng: np.random.Generator) -> F32:
    h, w, s = layout.height, layout.width, layout.scale
    polys = []
    for x_top in np.linspace(0.08, 0.95, 6) * w + rng.normal(0, 12 * s, 6):
        half_top = rng.uniform(6, 16) * s
        drift = rng.uniform(-0.22, -0.1) * h
        half_bottom = half_top * rng.uniform(2.2, 3.5)
        polys.append(
            np.array(
                [
                    [x_top - half_top, layout.surface_y],
                    [x_top + half_top, layout.surface_y],
                    [x_top + drift + half_bottom, h * 0.85],
                    [x_top + drift - half_bottom, h * 0.85],
                ]
            )
        )
    shafts = _poly_mask(h, w, polys, blur=10 * s)
    fade = (
        np.clip(1.0 - (np.arange(h, dtype=np.float32) - layout.surface_y) / (0.8 * h), 0.0, 1.0)
        ** 1.5
    )
    return shafts * fade[:, None] * 0.2


def _paint_backdrop(img: F32, layout: TankLayout, rng: np.random.Generator) -> None:
    """Distant rocks and weeds, blurred and tinted by the water."""
    h, w, s = layout.height, layout.width, layout.scale
    polys: list[F64] = []
    for cx in np.linspace(-0.05, 1.05, 9) * w:
        polys.append(
            _rock_polygon(
                rng,
                cx + rng.normal(0, 20 * s),
                0.8 * h,
                rng.uniform(40, 80) * s,
                rng.uniform(25, 60) * s,
            )
        )
    for x0 in rng.uniform(0, w, 26):
        height = rng.uniform(0.15, 0.35) * h
        sway = rng.normal(0, 12 * s)
        t = np.linspace(0, 1, 12)
        cx = x0 + sway * t**2
        cy = 0.8 * h - height * t
        half = 3.0 * s * (1 - t**3)
        polys.append(
            np.concatenate([np.stack([cx - half, cy], 1), np.stack([cx + half, cy], 1)[::-1]])
        )
    mask = _poly_mask(h, w, polys, blur=2.2 * s)
    tint = img * 0.72 + np.array(_rgb(10, 40, 30), np.float32)
    _blend(img, tint, (mask * 0.6).astype(np.float32))


def _paint_sand(img: F32, layout: TankLayout, rng: np.random.Generator) -> F32:
    """Sand bed with grain, ripples and pebbles; returns the sand mask."""
    h, w, s = layout.height, layout.width, layout.scale
    ys = np.arange(h, dtype=np.float32)[:, None]
    top = layout.sand_top.astype(np.float32)[None, :]
    mask = np.clip((ys - top) / 2.0 + 0.5, 0.0, 1.0)
    near = np.clip((ys - top) / (h - top + 1e-6), 0.0, 1.0)
    far_col = np.array(_rgb(150, 160, 150), np.float32)
    near_col = np.array(_rgb(214, 186, 140), np.float32)
    sand = far_col * (1 - near[..., None]) + near_col * near[..., None]
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.6)
    ripples = np.sin(ys * (0.35 / s) + 3.0 * _value_noise(rng, h, w, 40 * s)) * 0.05
    blotch = (_value_noise(rng, h, w, 25 * s) - 0.5) * 0.18
    sand *= (1.0 + 0.07 * grain + ripples * near + blotch)[..., None]
    # The sand darkens where it meets the far water.
    sand *= (0.72 + 0.28 * np.clip((ys - top) / (14.0 * s), 0.0, 1.0))[..., None]
    _blend(img, sand, mask)

    palette = [
        _rgb(120, 115, 110),
        _rgb(160, 150, 135),
        _rgb(90, 80, 70),
        _rgb(225, 220, 205),
        _rgb(150, 110, 85),
        _rgb(70, 70, 75),
    ]
    pebbles = rng.uniform(0, 1, (140, 2))
    pebbles = pebbles[np.argsort(pebbles[:, 1])]  # far first
    shadow_layer = np.zeros((h, w), np.float32)
    for px, py in pebbles:
        x = px * w
        col_top = float(layout.sand_top[min(w - 1, int(x))])
        y = col_top + 6 * s + py**1.4 * (h - col_top)
        size = (1.5 + 4.5 * py + rng.uniform(0, 2.0)) * s
        rx, ry = size * rng.uniform(1.0, 1.6), size * rng.uniform(0.55, 0.8)
        angle = rng.uniform(-20, 20)
        center = (round(x * _ONE), round(y * _ONE))
        axes = (round(rx * _ONE), round(ry * _ONE))
        cv2.ellipse(
            shadow_layer,
            (round(x * _ONE), round((y + ry * 0.5) * _ONE)),
            axes,
            angle,
            0,
            360,
            1.0,
            -1,
            cv2.LINE_AA,
            _SHIFT,
        )
        base = np.array(palette[rng.integers(len(palette))], np.float32) * rng.uniform(0.85, 1.1)
        cv2.ellipse(
            img, center, axes, angle, 0, 360, tuple(float(c) for c in base), -1, cv2.LINE_AA, _SHIFT
        )
        hx, hy = x - rx * 0.25, y - ry * 0.3
        hi_col = tuple(float(c) for c in np.minimum(base * 1.25 + 25, 255))
        cv2.ellipse(
            img,
            (round(hx * _ONE), round(hy * _ONE)),
            (round(rx * 0.45 * _ONE), round(ry * 0.35 * _ONE)),
            angle,
            0,
            360,
            hi_col,
            -1,
            cv2.LINE_AA,
            _SHIFT,
        )
    shadow = cv2.GaussianBlur(shadow_layer, (0, 0), 1.5 * s) * mask * 0.35
    img *= (1.0 - shadow)[..., None]
    return mask


def _dome_shading(
    mask: F32, radius: float, bumps: F32, light: tuple[float, float, float]
) -> tuple[F32, F32]:
    """Lambert shading of a rounded object from its silhouette.

    The distance to the silhouette edge, square-rooted, is a dome-shaped
    height field; adding bump noise and taking its gradient gives surface
    normals. Returns (diffuse term, upward-facing factor).
    """
    inside = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
    height = np.sqrt(np.clip(dist / radius, 0.0, 1.0)) * radius * 0.9 + bumps * inside
    height = cv2.GaussianBlur(height, (0, 0), 1.2)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    norm = np.sqrt(gx * gx + gy * gy + 1.0)
    lx, ly, lz = light
    diffuse = np.clip((-gx * lx - gy * ly + lz) / norm, 0.0, 1.0)
    upward = np.clip(-gy / norm, 0.0, 1.0)
    return diffuse.astype(np.float32), upward.astype(np.float32)


_LIGHT = (-0.38, -0.72, 0.58)  # from the lamp above and slightly left, towards the camera


def _paint_rocks(img: F32, layout: TankLayout, rng: np.random.Generator) -> F32:
    """Shaded boulders; returns how much each rock pixel faces the light above."""
    h, w, s = layout.height, layout.width, layout.scale
    top_light = np.zeros((h, w), np.float32)
    specs = [  # cx, base offset below the sand edge, rx, ry, colour
        (0.2, 18, 78, 92, _rgb(128, 120, 110)),
        (0.31, 26, 50, 50, _rgb(140, 126, 112)),
        (0.09, 30, 44, 40, _rgb(112, 110, 106)),
        (0.84, 16, 58, 52, _rgb(132, 116, 120)),
        (0.93, 30, 36, 30, _rgb(120, 118, 110)),
        (0.5, 34, 24, 15, _rgb(146, 134, 120)),
    ]
    for cx_f, base_off, rx, ry, color in specs:
        cx = cx_f * w
        base_y = float(layout.sand_top[min(w - 1, int(cx))]) + base_off * s
        poly = _rock_polygon(rng, cx, base_y, rx * s, ry * s)
        # Soft contact shadow on the sand first.
        shadow = np.zeros((h, w), np.float32)
        center = (round(cx * _ONE), round(base_y * _ONE))
        axes = (round(rx * 1.15 * s * _ONE), round(ry * 0.22 * s * _ONE))
        cv2.ellipse(shadow, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA, _SHIFT)
        img *= (1.0 - cv2.GaussianBlur(shadow, (0, 0), 6 * s) * 0.45)[..., None]

        mask = _poly_mask(h, w, [poly])
        bumps = (_value_noise(rng, h, w, 7 * s) - 0.5) * 7.0 * s + (
            _value_noise(rng, h, w, 2.5 * s) - 0.5
        ) * 2.0 * s
        diffuse, upward = _dome_shading(
            mask, 0.6 * min(rx, ry) * s, bumps.astype(np.float32), _LIGHT
        )
        texture = (
            0.88 + 0.14 * _value_noise(rng, h, w, 16 * s) + 0.08 * _value_noise(rng, h, w, 3 * s)
        )
        shade = (0.3 + 0.85 * diffuse) * texture
        rock = np.array(color, np.float32) * shade[..., None]
        # A film of algae where the surface faces the light.
        moss = np.clip(upward * 1.6 - 0.25, 0.0, 1.0) * _smooth01(
            _value_noise(rng, h, w, 14 * s), 0.3, 0.7
        )
        algae = np.array(_rgb(74, 102, 50), np.float32) * shade[..., None]
        rock = rock * (1 - 0.65 * moss[..., None]) + algae * 0.65 * moss[..., None]
        _blend(img, rock, mask)
        top_light = np.maximum(top_light, mask * upward)
    return top_light


def _paint_driftwood(img: F32, layout: TankLayout, rng: np.random.Generator) -> None:
    """A gnarled branch of driftwood rising from the sand."""
    h, w, s = layout.height, layout.width, layout.scale
    root_x = 0.39 * w
    root_y = float(layout.sand_top[int(root_x)]) + 10 * s
    branches = [  # control points (fractions of the frame), base half-width
        ([(root_x / w, root_y / h), (0.42, 0.72), (0.47, 0.62), (0.54, 0.5), (0.58, 0.45)], 13.0),
        ([(0.435, 0.69), (0.44, 0.6), (0.42, 0.5), (0.4, 0.46)], 6.5),
        ([(0.5, 0.58), (0.55, 0.585), (0.6, 0.56)], 5.0),
        ([(root_x / w + 0.005, root_y / h - 0.01), (0.36, 0.75), (0.33, 0.73)], 6.0),
    ]
    polys = []
    for pts, half in branches:
        ctrl = np.array(pts) * np.array([w, h])
        t = np.linspace(0, 1, 40)
        idx = np.linspace(0, len(ctrl) - 1, 40)
        x = np.interp(idx, np.arange(len(ctrl)), ctrl[:, 0])
        y = np.interp(idx, np.arange(len(ctrl)), ctrl[:, 1])
        x = cv2.GaussianBlur(x.reshape(1, -1), (1, 0), 3).ravel() if len(ctrl) > 2 else x
        y = cv2.GaussianBlur(y.reshape(1, -1), (1, 0), 3).ravel() if len(ctrl) > 2 else y
        dx, dy = np.gradient(x), np.gradient(y)
        norm = np.hypot(dx, dy) + 1e-9
        nx, ny = -dy / norm, dx / norm
        width = half * s * (1.0 - 0.75 * t) * (1.0 + 0.12 * np.sin(t * 17.0))
        left = np.stack([x + nx * width, y + ny * width], axis=1)
        right = np.stack([x - nx * width, y - ny * width], axis=1)[::-1]
        polys.append(np.concatenate([left, right]))
    mask = _poly_mask(h, w, polys)
    grain = _value_noise(rng, h, w, 3 * s)
    streaks = cv2.GaussianBlur(
        rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=4.0
    )
    diffuse, _ = _dome_shading(mask, 7.0 * s, (grain * 1.5 * s).astype(np.float32), _LIGHT)
    shade = (0.3 + 0.9 * diffuse) * (0.9 + 0.12 * grain + 0.08 * streaks)
    wood = np.array(_rgb(104, 72, 46), np.float32) * shade[..., None]
    _blend(img, wood, mask)


def _paint_bubbler(img: F32, layout: TankLayout) -> None:
    s = layout.scale
    bx, by = layout.bubbler
    # Air line running up the back glass.
    tube = np.zeros(img.shape[:2], np.float32)
    pts = np.array(
        [
            [bx + 3 * s, by - 4 * s],
            [bx + 10 * s, by - 60 * s],
            [bx + 14 * s, layout.surface_y - 10 * s],
        ]
    )
    cv2.polylines(
        tube,
        [np.round(pts * _ONE).astype(np.int32)],
        False,
        1.0,
        max(1, round(2.5 * s)),
        cv2.LINE_AA,
        _SHIFT,
    )
    _blend(img, _rgb(150, 175, 170), tube * 0.35)
    cv2.ellipse(
        img,
        (round(bx * _ONE), round(by * _ONE)),
        (round(11 * s * _ONE), round(4.5 * s * _ONE)),
        0,
        0,
        360,
        _rgb(70, 70, 72),
        -1,
        cv2.LINE_AA,
        _SHIFT,
    )
    cv2.ellipse(
        img,
        (round(bx * _ONE), round((by - 1.5 * s) * _ONE)),
        (round(9 * s * _ONE), round(2.5 * s * _ONE)),
        0,
        0,
        360,
        _rgb(115, 115, 118),
        -1,
        cv2.LINE_AA,
        _SHIFT,
    )


@functools.lru_cache(maxsize=8)
def build_static_scene(width: int, height: int, reference_width: int) -> StaticScene:
    """Render the static layers for a frame size (cached; arrays read-only)."""
    layout = make_layout(width, height, reference_width)
    rng = np.random.default_rng(_SCENE_SEED)
    h, w = height, width
    img = _paint_water(layout)
    _paint_backdrop(img, layout, rng)
    sand_mask = _paint_sand(img, layout, rng)
    _paint_driftwood(img, layout, rng)
    rock_top = _paint_rocks(img, layout, rng)
    _paint_bubbler(img, layout)
    _paint_surface(img, layout)
    night = img.copy()
    img += _paint_light_shafts(layout, rng)[..., None] * np.array(_rgb(220, 255, 240), np.float32)

    ys = np.arange(h, dtype=np.float32)[:, None]
    water = (ys >= layout.surface_y) * np.float32(0.05)
    gain = np.maximum(water * (1 - sand_mask), sand_mask * 0.42)
    gain = np.maximum(gain, rock_top * 0.3)
    gain = cv2.GaussianBlur(gain.astype(np.float32), (0, 0), 1.5)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vignette = np.clip(1.0 - 0.2 * r2**1.2, 0.0, 1.0)
    falloff = np.clip(1.0 - 0.32 * r2, 0.0, 1.0)

    def u8(a: npt.NDArray[np.floating]) -> U8:
        out = np.clip(a + 0.5, 0, 255).astype(np.uint8)
        out.setflags(write=False)
        return out

    water_rows = _paint_water(layout)[:, w // 2, :].copy()
    water_rows.setflags(write=False)
    return StaticScene(
        layout=layout,
        bg_day=u8(img),
        bg_night=u8(night),
        caustic_gain=u8(gain * 255.0),
        vignette=u8(np.repeat(vignette[..., None] * 255.0, 3, axis=2)),
        ir_falloff=u8(falloff * 255.0),
        water_bgr=water_rows,
    )


# ---------------------------------------------------------------------------
# Plants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Blade:
    """One leaf or grass blade of the planting plan."""

    color: tuple[float, float, float]
    base: tuple[float, float]
    length: float
    width: float
    lean: float
    curl: float
    sway: float
    phase: float
    lance: bool


class PlantLayer:
    """All blades of one depth layer, animated in one vectorised pass.

    Each blade is a ribbon whose segment angles follow a travelling wave
    ``lean + curl * f + sway * f^1.5 * sin(omega t + phase - 2 f)`` (``f`` is
    the fraction of the way up the blade), so tips move more than bases and
    the motion runs up the plant like a current pushing it. Geometry for all
    blades is computed at once; drawing costs two anti-aliased ``fillPoly``
    calls per colour (a darker margin under a narrower lit blade, which reads
    as a curved leaf) plus the midribs.
    """

    segments = 10

    def __init__(self, blades: list[_Blade]) -> None:
        self.n = len(blades)
        self.base = np.array([b.base for b in blades], dtype=np.float64).reshape(-1, 2)
        self.length = np.array([b.length for b in blades], dtype=np.float64)
        self.lean = np.array([b.lean for b in blades], dtype=np.float64)
        self.curl = np.array([b.curl for b in blades], dtype=np.float64)
        self.sway = np.array([b.sway for b in blades], dtype=np.float64)
        self.phase = np.array([b.phase for b in blades], dtype=np.float64)
        f = np.linspace(0.0, 1.0, self.segments + 1)
        self.f = f[None, :]
        width = np.array([b.width for b in blades], dtype=np.float64)[:, None]
        lance = np.array([b.lance for b in blades], dtype=bool)[:, None]
        ribbon_profile = (1.0 - self.f**6) ** 0.5
        lance_profile = np.sin(np.pi * np.clip(self.f, 0.02, 1.0)) ** 0.8
        self.half_width = 0.5 * width * np.where(lance, lance_profile, ribbon_profile)
        colors = sorted({b.color for b in blades})
        self.groups = [
            (
                tuple(v * 0.72 for v in c),  # leaf margin, in shadow
                c,  # leaf blade
                tuple(min(255.0, v * 1.3 + 14.0) for v in c),  # midrib catching light
                np.array([i for i, b in enumerate(blades) if b.color == c]),
            )
            for c in colors
        ]

    def draw(self, img: U8, t: float, sway: float = 1.0) -> None:
        """Draw every blade at time ``t``; ``sway`` scales the current's strength."""
        if self.n == 0:
            return
        angle = self.lean[:, None] + self.curl[:, None] * self.f
        angle = angle + sway * self.sway[:, None] * self.f**1.5 * np.sin(
            1.3 * t + self.phase[:, None] - 2.0 * self.f
        )
        seg = (self.length / self.segments)[:, None]
        steps_x = np.sin(angle[:, :-1]) * seg
        steps_y = -np.cos(angle[:, :-1]) * seg
        zero = np.zeros((self.n, 1))
        cx = self.base[:, :1] + np.concatenate([zero, np.cumsum(steps_x, axis=1)], axis=1)
        cy = self.base[:, 1:] + np.concatenate([zero, np.cumsum(steps_y, axis=1)], axis=1)
        nx, ny = np.cos(angle) * self.half_width, np.sin(angle) * self.half_width
        outer = self._outline(cx, cy, nx, ny)
        inner = self._outline(cx, cy, 0.55 * nx, 0.55 * ny)
        ribs = np.round(np.stack([cx, cy], axis=2)[:, :-2] * _ONE).astype(np.int32)
        for margin, blade, rib, idx in self.groups:
            cv2.fillPoly(img, list(outer[idx]), margin, cv2.LINE_AA, _SHIFT)
            cv2.fillPoly(img, list(inner[idx]), blade, cv2.LINE_AA, _SHIFT)
            cv2.polylines(img, list(ribs[idx]), False, rib, 1, cv2.LINE_AA, _SHIFT)

    @staticmethod
    def _outline(cx: F64, cy: F64, nx: F64, ny: F64) -> npt.NDArray[np.int32]:
        left = np.stack([cx + nx, cy + ny], axis=2)
        right = np.stack([cx - nx, cy - ny], axis=2)[:, ::-1]
        return np.round(np.concatenate([left, right], axis=1) * _ONE).astype(np.int32)


@dataclass(frozen=True, slots=True)
class Plants:
    """The planted tank: a layer behind the fish and one in front of them."""

    back: PlantLayer
    front: PlantLayer

    def draw(self, img: U8, t: float, *, front: bool, sway: float = 1.0) -> None:
        (self.front if front else self.back).draw(img, t, sway)


@functools.lru_cache(maxsize=8)
def build_plants(width: int, height: int, reference_width: int) -> Plants:
    """The fixed planting plan of the tank (deterministic, cached)."""
    layout = make_layout(width, height, reference_width)
    rng = np.random.default_rng(_SCENE_SEED + 1)
    w, h, s = layout.width, layout.height, layout.scale
    greens = [
        _rgb(46, 120, 52),
        _rgb(62, 140, 58),
        _rgb(38, 98, 48),
        _rgb(86, 156, 66),
        _rgb(52, 128, 80),
    ]

    def blades(
        count: int,
        x_range: tuple[float, float],
        length: tuple[float, float],
        width: float,
        *,
        lean: float = 0.12,
        curl: float = 0.35,
        sway: float = 0.05,
        lance: bool = False,
        colors: list[tuple[float, float, float]] = greens,
        base_y: float | None = None,
    ) -> list[_Blade]:
        out = []
        for _ in range(count):
            x = rng.uniform(*x_range) * w
            y = (
                base_y
                if base_y is not None
                else float(layout.sand_top[min(w - 1, max(0, int(x)))]) + rng.uniform(2, 20) * s
            )
            out.append(
                _Blade(
                    color=colors[int(rng.integers(len(colors)))],
                    base=(x, y),
                    length=rng.uniform(*length) * h,
                    width=rng.uniform(0.75, 1.25) * width * s,
                    lean=rng.normal(0.0, lean),
                    curl=rng.uniform(-curl, curl),
                    sway=rng.uniform(0.6, 1.0) * sway,
                    phase=rng.uniform(0.0, 2 * np.pi),
                    lance=lance,
                )
            )
        return out

    reds = [_rgb(170, 60, 50), _rgb(196, 86, 52), _rgb(150, 50, 60)]
    sword = [_rgb(58, 132, 50), _rgb(78, 150, 60), _rgb(48, 116, 46)]
    shade = [_rgb(30, 78, 40), _rgb(36, 90, 44)]
    back = (
        blades(12, (0.06, 0.2), (0.12, 0.2), 4.0, curl=0.25)
        + blades(16, (0.0, 0.16), (0.38, 0.62), 7.0)
        + blades(14, (0.84, 1.0), (0.3, 0.56), 7.0)
        + blades(10, (0.42, 0.56), (0.1, 0.2), 4.0, curl=0.2)
        + blades(
            11,
            (0.33, 0.35),
            (0.08, 0.15),
            12.0,
            lean=0.42,
            curl=0.25,
            sway=0.03,
            lance=True,
            colors=reds,
        )
        + blades(
            10,
            (0.6, 0.61),
            (0.17, 0.3),
            19.0,
            lean=0.6,
            curl=0.3,
            sway=0.035,
            lance=True,
            colors=sword,
        )
    )
    front = blades(5, (0.0, 0.07), (0.4, 0.55), 9.0, colors=shade, base_y=h + 8 * s)
    front += blades(3, (0.95, 1.0), (0.3, 0.45), 9.0, colors=shade, base_y=h + 8 * s)
    return Plants(PlantLayer(back), PlantLayer(front))


# ---------------------------------------------------------------------------
# Caustics
# ---------------------------------------------------------------------------


class Caustics:
    """Animated caustics from refracted rays focusing on the floor.

    A periodic height field ``h = sum_i a_i cos(k_i . p - w_i t + phi_i)``
    tilts the light rays; each ray lands displaced by ``depth * grad h``.
    Counting landed rays per cell gives the light density, which is bright
    along the fold lines of that mapping: real caustics, not a texture.

    Wave numbers are integers per grid period and angular frequencies are
    integers per ``period_s``, so the pattern tiles in space (rays wrap
    around seamlessly) and loops in time. Frames are computed lazily at
    ``slot_hz`` and cached, so after one loop the per-frame cost is one
    ``cv2.remap`` (which also squashes the pattern on the sand floor for
    perspective).
    """

    def __init__(
        self,
        layout: TankLayout,
        *,
        grid_div: int = 4,
        period_s: float = 10.0,
        slot_hz: float = 30.0,
    ) -> None:
        self.layout = layout
        self.gw, self.gh = max(16, layout.width // grid_div), max(12, layout.height // grid_div)
        self.period_s = period_s
        self.slot_hz = slot_hz
        self.n_slots = round(period_s * slot_hz)
        self._cache: dict[int, U8] = {}
        rng = np.random.default_rng(_SCENE_SEED + 2)
        n = 7
        ang = rng.uniform(0.0, 2.0 * np.pi, n)
        mag = rng.uniform(6.0, 12.0, n)
        ix = np.round(np.cos(ang) * mag)
        iy = np.round(np.sin(ang) * mag * self.gh / self.gw)
        self.kx = (2 * np.pi * ix / self.gw).astype(np.float32)
        self.ky = (2 * np.pi * iy / self.gh).astype(np.float32)
        k2 = np.maximum(self.kx**2 + self.ky**2, 1e-6)
        self.omega = (2 * np.pi * rng.integers(3, 9, n) / period_s * rng.choice([-1, 1], n)).astype(
            np.float32
        )
        self.amp = (rng.uniform(0.7, 1.0, n) / k2).astype(np.float32)
        self.phi = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
        rays = 2
        ys, xs = np.mgrid[0 : self.gh * rays, 0 : self.gw * rays].astype(np.float32) / rays
        self.rx, self.ry = xs.ravel(), ys.ravel()
        self.ray_weight = np.float32(1.0 / (rays * rays))
        self.proj = (self.kx[:, None] * self.rx[None] + self.ky[:, None] * self.ry[None]).astype(
            np.float32
        )
        self.map_x, self.map_y = self._maps()

    def _maps(self) -> tuple[F32, F32]:
        lay = self.layout
        h, w = lay.height, lay.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        top = lay.sand_top.astype(np.float32)[None, :]
        floor_y = np.where(yy > top, top + (yy - top) * 2.4, yy)
        return (xx + 0.5) * self.gw / w - 0.5, (floor_y + 0.5) * self.gh / h - 0.5

    def _render(self, t: float) -> U8:
        # Accumulate wave by wave: a (waves x rays) matrix product would hand
        # this tiny job to a multi-threaded BLAS, whose spinning threads cost
        # far more than the arithmetic on a busy machine.
        shift_x = np.zeros_like(self.rx)
        shift_y = np.zeros_like(self.ry)
        offsets = self.phi - self.omega * np.float32(t)
        for i in range(len(self.kx)):
            c = np.cos(self.proj[i] + offsets[i])
            c *= 0.75 * self.amp[i]
            shift_x += self.kx[i] * c
            shift_y += self.ky[i] * c
        px = (self.rx + shift_x).astype(np.int32) % self.gw
        py = (self.ry + shift_y).astype(np.int32) % self.gh
        density = np.bincount(py * self.gw + px, minlength=self.gw * self.gh).reshape(
            self.gh, self.gw
        )
        light = cv2.GaussianBlur(density.astype(np.float32) * self.ray_weight, (0, 0), 0.7)
        return np.clip((light - 0.7) * (255.0 / 1.8), 0, 255).astype(np.uint8)

    def frame(self, t: float) -> U8:
        """Low-resolution caustic intensity at time ``t`` (cached by slot)."""
        slot = round((t % self.period_s) * self.slot_hz) % self.n_slots
        cached = self._cache.get(slot)
        if cached is None:
            cached = self._render(slot / self.slot_hz)
            self._cache[slot] = cached
        return cached

    def apply(self, img: U8, t: float, gain: U8) -> None:
        """Add the caustic light at time ``t`` to ``img`` in place."""
        full = cv2.remap(
            self.frame(t), self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
        )
        lit = cv2.multiply(full, gain, scale=1.0 / 255.0)
        cv2.add(img, cv2.cvtColor(lit, cv2.COLOR_GRAY2BGR), dst=img)


@functools.lru_cache(maxsize=8)
def build_caustics(width: int, height: int, reference_width: int) -> Caustics:
    """Shared caustics model per frame size (its frame cache is shared too)."""
    return Caustics(make_layout(width, height, reference_width))


# ---------------------------------------------------------------------------
# Bubbles
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _bubble_sprite(radius_q: int) -> U8:
    """Additive bubble sprite: bright rim, faint body, catch-light."""
    r = radius_q / 2.0
    size = math.ceil(r) * 2 + 3
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    rim = np.clip(1.0 - np.abs(d - r * 0.85) / max(0.8, 0.3 * r), 0.0, 1.0)
    body = np.clip(r - d, 0.0, 1.0) * 0.15
    spot = np.clip(
        1.0 - np.hypot(xx - (c - 0.35 * r), yy - (c - 0.35 * r)) / max(0.6, 0.3 * r), 0.0, 1.0
    )
    v = np.clip(rim * 0.6 + body + spot * 0.9, 0.0, 1.0)
    tint = np.array([1.0, 1.0, 0.92], np.float32)
    out = np.clip(v[..., None] * tint * 150.0, 0, 255).astype(np.uint8)
    out.setflags(write=False)
    return out


class Bubbles:
    """Air bubbles from the bubbler: Poisson emission, rise with wobble.

    Rise speed grows with bubble size (bigger bubbles rise faster), bubbles
    swell slightly as the pressure drops and pop at the surface.
    """

    def __init__(self, layout: TankLayout, rng: np.random.Generator, rate_hz: float = 10.0) -> None:
        self.layout = layout
        self.rng = rng
        self.rate_hz = rate_hz
        self.x0 = np.zeros(0)
        self.y = np.zeros(0)
        self.r = np.zeros(0)
        self.vy = np.zeros(0)
        self.wobble = np.zeros(0)
        self.freq = np.zeros(0)
        self.t = 0.0

    def step(self, dt: float) -> None:
        s = self.layout.scale
        self.t += dt
        n_new = int(self.rng.poisson(self.rate_hz * dt))
        if n_new:
            bx, by = self.layout.bubbler
            r = self.rng.uniform(1.0, 3.4, n_new) * s
            self.x0 = np.concatenate([self.x0, bx + self.rng.normal(0, 2.0 * s, n_new)])
            self.y = np.concatenate([self.y, by - self.rng.uniform(0, 6 * s, n_new)])
            self.r = np.concatenate([self.r, r])
            self.vy = np.concatenate([self.vy, (45.0 + 18.0 * r / s) * s])
            self.wobble = np.concatenate([self.wobble, self.rng.uniform(0.0, 2 * np.pi, n_new)])
            self.freq = np.concatenate([self.freq, self.rng.uniform(1.5, 3.5, n_new)])
        self.y -= self.vy * dt
        self.r *= 1.0 + 0.04 * dt
        alive = self.y > self.layout.surface_y + self.r
        self.x0, self.y, self.r = self.x0[alive], self.y[alive], self.r[alive]
        self.vy, self.wobble, self.freq = self.vy[alive], self.wobble[alive], self.freq[alive]

    def draw(self, img: U8) -> None:
        h, w = img.shape[:2]
        s = self.layout.scale
        xs = self.x0 + 2.5 * s * np.sin(2 * np.pi * self.freq * self.t + self.wobble)
        for x, y, r in zip(xs, self.y, self.r, strict=True):
            sprite = _bubble_sprite(max(2, round(r * 2)))
            k = sprite.shape[0]
            x0, y0 = round(x) - k // 2, round(y) - k // 2
            if x0 < 0 or y0 < 0 or x0 + k > w or y0 + k > h:
                continue
            roi = img[y0 : y0 + k, x0 : x0 + k]
            cv2.add(roi, sprite, dst=roi)


# ---------------------------------------------------------------------------
# Water surface and LED
# ---------------------------------------------------------------------------


def _paint_surface(img: F32, layout: TankLayout) -> None:
    """The meniscus: a bright line where the camera sees the surface edge-on.

    It is static on purpose. Seen from the side, the surface of an aquarium
    is a steady line; real ripples show up as a shimmer in brightness (see
    :func:`shimmer_surface`), not as the line moving, and a moving
    high-contrast line would read as motion to any background model.
    """
    s = layout.scale
    xs = np.linspace(0, layout.width, 80)
    ys = layout.surface_y + 0.8 * s * np.sin(xs * 0.03 / s + 0.4)
    pts = np.round(np.stack([xs, ys], axis=1) * _ONE).astype(np.int32)
    glow = np.zeros(img.shape[:2], np.float32)
    cv2.polylines(
        glow,
        [pts + np.array([0, 3 * _ONE], np.int32)],
        False,
        1.0,
        max(1, round(5 * s)),
        cv2.LINE_AA,
        _SHIFT,
    )
    _blend(img, _rgb(150, 210, 215), cv2.GaussianBlur(glow, (0, 0), 1.5 * s) * 0.35)
    line = np.zeros(img.shape[:2], np.float32)
    cv2.polylines(line, [pts], False, 1.0, max(1, round(1.6 * s)), cv2.LINE_AA, _SHIFT)
    _blend(img, _rgb(215, 240, 235), line)


def shimmer_surface(img: U8, layout: TankLayout, t: float) -> None:
    """Animate the surface band with slow travelling brightness waves (+-8 %)."""
    s = layout.scale
    y0 = max(0, int(layout.surface_y - 4 * s))
    y1 = min(layout.height, int(layout.surface_y + 12 * s))
    x = np.arange(layout.width, dtype=np.float32) / s
    gain = 1.0 + 0.05 * np.sin(x * 0.05 - 1.1 * t) + 0.03 * np.sin(x * 0.013 + 0.7 * t)
    band = img[y0:y1]
    shape = (1, -1, 1) if band.ndim == 3 else (1, -1)
    band[...] = cv2.convertScaleAbs(band.astype(np.float32) * gain.reshape(shape))


class LedLamp:
    """The liveness LED as the camera sees it: a dark housing that glows.

    The LED is an 850 nm IR emitter seen by a NoIR camera, so it is visible
    day and night: by day it glows the pinkish-white NoIR cameras render IR
    as, at night it is plain white on the monochrome image. The glow is a
    pre-rendered additive sprite (core + halo scattered by the water).
    """

    def __init__(self, center: tuple[int, int], radius: int) -> None:
        self.center = center
        self.radius = radius
        reach = 4 * radius
        size = 2 * reach + 1
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - reach
        d = np.sqrt(xx * xx + yy * yy)
        core = np.clip(radius + 0.5 - d, 0.0, 1.0)
        halo = np.exp(-((d / (1.6 * radius)) ** 2)) * 0.55 + np.exp(-d / (1.2 * radius)) * 0.25
        glow = np.clip(core + halo, 0.0, 1.0)
        self.reach = reach
        self._glow_day = (glow[..., None] * np.array([235, 205, 245], np.float32)).astype(np.uint8)
        self._glow_night = (glow * 255.0).astype(np.uint8)
        body = np.clip(radius + 1.5 - d, 0.0, 1.0)
        self._body_alpha = body
        self._body_level = (
            40.0
            + 25.0
            * np.clip(1.0 - np.hypot(xx + 0.35 * radius, yy + 0.35 * radius) / radius, 0.0, 1.0)
        ).astype(np.float32)

    @property
    def roi(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) box covering the LED and its brightest halo."""
        cx, cy = self.center
        half = 2 * self.radius
        return (cx - half, cy - half, 2 * half + 1, 2 * half + 1)

    def draw(self, img: U8, on: bool) -> None:
        """Draw onto a BGR (day) or single-channel (night) image in place."""
        cx, cy = self.center
        r = self.reach
        h, w = img.shape[:2]
        y0, y1, x0, x1 = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        sy, sx = slice(y0 - (cy - r), y1 - (cy - r)), slice(x0 - (cx - r), x1 - (cx - r))
        roi = img[y0:y1, x0:x1]
        a = self._body_alpha[sy, sx]
        level = self._body_level[sy, sx]
        if roi.ndim == 3:
            housed = roi * (1.0 - a[..., None]) + (level * a)[..., None]
        else:
            housed = roi * (1.0 - a) + level * a
        roi[...] = np.clip(housed + 0.5, 0, 255).astype(np.uint8)
        if on:
            glow = self._glow_day[sy, sx] if roi.ndim == 3 else self._glow_night[sy, sx]
            cv2.add(roi, glow, dst=roi)


def ir_converter(scene: StaticScene) -> Callable[[U8], U8]:
    """Return a function mapping a lit colour frame to the IR night view.

    Near-infrared sees reflectance, not colour: water absorbs it (dark
    background), sand, plants and pale fish reflect it. A channel mix
    weighted towards red approximates that; a tone curve makes the scene dim
    and slightly flat, and the illuminator's hot spot falls off to the
    corners.
    """
    weights = np.array([[0.12, 0.38, 0.5]], np.float32)
    x = np.arange(256, dtype=np.float32) / 255.0
    tone = np.clip(14.0 + 170.0 * x**1.1, 0, 255).astype(np.uint8)
    falloff = scene.ir_falloff

    def convert(img: U8) -> U8:
        gray = cv2.transform(img, weights)
        gray = cv2.LUT(gray, tone)
        return cast(U8, cv2.multiply(gray, falloff, scale=1.0 / 255.0))

    return convert
