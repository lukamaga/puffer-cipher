"""Procedural fish sprites for the aquarium simulator.

Every fish is painted once, at start-up, into a small set of pre-rendered
sprites (one per tail-beat phase); per frame the renderer only picks a phase
and pastes it with a single affine warp. That split is what makes a
good-looking simulator cheap: all the expensive art (per-pixel shading,
species patterns, translucent fins with rays, supersampled anti-aliasing)
happens ``PHASES`` times per fish instead of 60 times per second.

Anatomy model
-------------
A fish is described in *body-length units* (``L = 1``) with the nose at
``x = +0.5`` and the tip of the tail at about ``x = -0.5``:

* the body is a closed outline from upper and lower profiles
  (half-heights as a function of ``s``, 0 at the nose and 1 at the caudal
  peduncle), interpolated with monotone cubic splines so that a handful of
  control points gives a smooth, overshoot-free silhouette;
* fins (dorsal, anal, pelvic) are polygons growing from the body edge,
  caudal fins are fans attached to the peduncle (rounded, forked,
  truncate, lyre or flowing), the pectoral fin is a flapping paddle drawn
  over the body;
* animation: a tail that sweeps (seen from the side the sweep shows up as
  foreshortening plus a slight vertical flick), rippling fin edges and
  fluttering pectorals. The body outline is static, so it is painted once
  and only the fins are re-rendered per phase.

Appearance
----------
Colours come from the :class:`~puffer.sources.synthetic.FishSpec`, species
patterns (clownfish bands, the blue tang's palette, neon stripe, ...) are
evaluated in body coordinates ``(s, t)`` where ``t`` runs from -1 (dorsal
edge) to +1 (ventral edge). Shading treats the body as a cylinder lit from
above and in front (diffuse + a soft specular streak), which is enough to
make flat polygons read as round fish. Everything is drawn at ``SUPERSAMPLE``
times the final size and area-averaged down, giving clean anti-aliasing.

Sprites are premultiplied-alpha ``float32`` BGRA, which makes the per-frame
bilinear warp and compositing exact (no dark fringes at the edges).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from puffer.sources.synthetic import FishSpec

F32 = npt.NDArray[np.float32]
Color = tuple[float, float, float]

PHASES = 16
"""Pre-rendered tail-beat phases per fish (per-frame phase is quantised)."""

SUPERSAMPLE = 4
"""Drawing resolution multiplier used for anti-aliasing at build time."""

_SHIFT = 4  # cv2 fixed-point fractional bits for sub-pixel polygon vertices
_ONE = 1 << _SHIFT
ALPHA_HULL_THRESHOLD = 0.35
"""Alpha above which a sprite pixel counts as "fish" for ground-truth boxes."""


# ---------------------------------------------------------------------------
# Anatomy descriptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinShape:
    """A fin growing from the dorsal (``side=-1``) or ventral (``+1``) edge.

    ``s0``/``s1`` bound its base along the body; ``outline`` gives control
    points ``(u, height, sweep)`` along the base (``u`` in [0, 1]) with the
    fin edge at ``height`` away from the body and ``sweep`` towards the tail
    (both in body lengths).
    """

    side: int
    s0: float
    s1: float
    outline: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class Caudal:
    """Tail fin: ``shape`` in round/fork/truncate/lyre/flowing."""

    shape: str
    length: float
    spread: float  # half-angle of the fan, radians


@dataclass(frozen=True, slots=True)
class Anatomy:
    """Geometry and drawing style of one species (body-length units)."""

    body: float  # nose-to-peduncle length as a fraction of L
    top: tuple[tuple[float, float], ...]  # (s, half-height) dorsal profile
    bottom: tuple[tuple[float, float], ...]  # (s, half-height) ventral profile
    caudal: Caudal
    fins: tuple[FinShape, ...]
    pectoral: tuple[float, float, float, float]  # s, t, length, width
    eye: tuple[float, float, float]  # s, t, radius
    iris: Color
    fin_alpha: float
    ray_spacing: float  # distance between fin rays (L units); 0 = none
    outline_alpha: float = 0.55


def _fin(side: int, s0: float, s1: float, *pts: tuple[float, float, float]) -> FinShape:
    return FinShape(side, s0, s1, tuple(pts))


ANATOMY: dict[str, Anatomy] = {
    "clownfish": Anatomy(
        body=0.80,
        top=(
            (0, 0),
            (0.02, 0.07),
            (0.1, 0.14),
            (0.3, 0.185),
            (0.55, 0.17),
            (0.82, 0.105),
            (1, 0.07),
        ),
        bottom=(
            (0, 0),
            (0.02, 0.05),
            (0.12, 0.12),
            (0.35, 0.165),
            (0.62, 0.135),
            (0.86, 0.085),
            (1, 0.065),
        ),
        caudal=Caudal("round", 0.22, 0.75),
        fins=(
            _fin(-1, 0.22, 0.52, (0, 0, 0), (0.3, 0.075, 0.02), (0.7, 0.07, 0.03), (1, 0.03, 0.02)),
            _fin(-1, 0.5, 0.92, (0, 0.03, 0), (0.3, 0.11, 0.05), (0.7, 0.10, 0.07), (1, 0, 0.02)),
            _fin(+1, 0.58, 0.88, (0, 0, 0), (0.35, 0.09, 0.05), (0.75, 0.08, 0.06), (1, 0, 0.02)),
            _fin(+1, 0.22, 0.34, (0, 0, 0), (0.5, 0.08, 0.06), (1, 0, 0.02)),
        ),
        pectoral=(0.27, 0.25, 0.13, 0.065),
        eye=(0.11, -0.25, 0.045),
        iris=(30, 110, 230),
        fin_alpha=0.95,
        ray_spacing=0.025,
    ),
    "blue_tang": Anatomy(
        body=0.78,
        top=(
            (0, 0),
            (0.02, 0.06),
            (0.1, 0.16),
            (0.3, 0.235),
            (0.55, 0.22),
            (0.82, 0.14),
            (1, 0.06),
        ),
        bottom=(
            (0, 0),
            (0.02, 0.06),
            (0.14, 0.16),
            (0.38, 0.22),
            (0.65, 0.18),
            (0.88, 0.1),
            (1, 0.055),
        ),
        caudal=Caudal("truncate", 0.22, 0.85),
        fins=(
            _fin(
                -1,
                0.18,
                0.96,
                (0, 0, 0),
                (0.15, 0.06, 0.02),
                (0.6, 0.065, 0.03),
                (0.92, 0.05, 0.05),
                (1, 0, 0.02),
            ),
            _fin(+1, 0.45, 0.95, (0, 0, 0), (0.2, 0.06, 0.02), (0.8, 0.06, 0.05), (1, 0, 0.02)),
            _fin(+1, 0.26, 0.36, (0, 0, 0), (0.5, 0.07, 0.05), (1, 0, 0.02)),
        ),
        pectoral=(0.26, 0.15, 0.15, 0.07),
        eye=(0.11, -0.2, 0.045),
        iris=(40, 40, 40),
        fin_alpha=0.95,
        ray_spacing=0.02,
    ),
    "pufferfish": Anatomy(
        body=0.84,
        top=(
            (0, 0),
            (0.015, 0.09),
            (0.08, 0.18),
            (0.3, 0.245),
            (0.6, 0.21),
            (0.86, 0.11),
            (1, 0.06),
        ),
        bottom=(
            (0, 0),
            (0.015, 0.08),
            (0.1, 0.2),
            (0.38, 0.27),
            (0.68, 0.2),
            (0.9, 0.1),
            (1, 0.06),
        ),
        caudal=Caudal("round", 0.17, 0.7),
        fins=(
            _fin(-1, 0.72, 0.86, (0, 0, 0), (0.4, 0.09, 0.06), (1, 0, 0.02)),
            _fin(+1, 0.74, 0.88, (0, 0, 0), (0.4, 0.08, 0.06), (1, 0, 0.02)),
        ),
        pectoral=(0.3, 0.05, 0.11, 0.08),
        eye=(0.14, -0.3, 0.07),
        iris=(60, 170, 200),
        fin_alpha=0.6,
        ray_spacing=0.02,
    ),
    "yellow_tang": Anatomy(
        body=0.80,
        top=(
            (0, 0.01),
            (0.03, 0.04),
            (0.1, 0.12),
            (0.3, 0.25),
            (0.5, 0.25),
            (0.8, 0.16),
            (1, 0.06),
        ),
        bottom=(
            (0, 0.01),
            (0.03, 0.03),
            (0.12, 0.13),
            (0.38, 0.24),
            (0.62, 0.21),
            (0.88, 0.11),
            (1, 0.055),
        ),
        caudal=Caudal("truncate", 0.2, 0.8),
        fins=(
            _fin(
                -1,
                0.2,
                0.95,
                (0, 0, 0),
                (0.2, 0.15, 0.05),
                (0.55, 0.17, 0.08),
                (0.88, 0.1, 0.07),
                (1, 0, 0.02),
            ),
            _fin(+1, 0.38, 0.93, (0, 0, 0), (0.25, 0.14, 0.05), (0.7, 0.12, 0.07), (1, 0, 0.02)),
        ),
        pectoral=(0.28, 0.15, 0.13, 0.06),
        eye=(0.16, -0.22, 0.045),
        iris=(20, 20, 20),
        fin_alpha=0.92,
        ray_spacing=0.022,
    ),
    "betta": Anatomy(
        body=0.58,
        top=((0, 0), (0.03, 0.06), (0.15, 0.1), (0.4, 0.11), (0.75, 0.095), (1, 0.08)),
        bottom=((0, 0), (0.03, 0.05), (0.18, 0.1), (0.45, 0.115), (0.8, 0.1), (1, 0.08)),
        caudal=Caudal("flowing", 0.42, 1.05),
        fins=(
            _fin(
                -1,
                0.5,
                0.98,
                (0, 0, 0),
                (0.2, 0.14, 0.1),
                (0.6, 0.2, 0.2),
                (0.9, 0.17, 0.25),
                (1, 0, 0.05),
            ),
            _fin(
                +1,
                0.26,
                1.0,
                (0, 0, 0),
                (0.15, 0.17, 0.08),
                (0.5, 0.26, 0.18),
                (0.85, 0.27, 0.28),
                (1, 0, 0.06),
            ),
            _fin(+1, 0.2, 0.28, (0, 0, 0), (0.5, 0.2, 0.08), (1, 0, 0.02)),
        ),
        pectoral=(0.28, 0.2, 0.09, 0.05),
        eye=(0.1, -0.2, 0.05),
        iris=(30, 30, 90),
        fin_alpha=0.82,
        ray_spacing=0.03,
        outline_alpha=0.4,
    ),
    "neon_tetra": Anatomy(
        body=0.82,
        top=(
            (0, 0),
            (0.02, 0.05),
            (0.12, 0.1),
            (0.35, 0.125),
            (0.65, 0.11),
            (0.9, 0.07),
            (1, 0.05),
        ),
        bottom=(
            (0, 0),
            (0.02, 0.04),
            (0.15, 0.1),
            (0.4, 0.125),
            (0.7, 0.11),
            (0.9, 0.07),
            (1, 0.05),
        ),
        caudal=Caudal("fork", 0.2, 0.62),
        fins=(
            _fin(-1, 0.48, 0.6, (0, 0, 0), (0.3, 0.1, 0.05), (1, 0, 0.03)),
            _fin(+1, 0.6, 0.85, (0, 0, 0), (0.25, 0.07, 0.04), (1, 0, 0.02)),
        ),
        pectoral=(0.25, 0.25, 0.1, 0.04),
        eye=(0.11, -0.12, 0.06),
        iris=(120, 120, 110),
        fin_alpha=0.4,
        ray_spacing=0.0,
    ),
    "guppy": Anatomy(
        body=0.62,
        top=((0, 0), (0.03, 0.05), (0.15, 0.095), (0.4, 0.105), (0.75, 0.085), (1, 0.06)),
        bottom=((0, 0), (0.03, 0.04), (0.2, 0.1), (0.5, 0.12), (0.8, 0.09), (1, 0.06)),
        caudal=Caudal("flowing", 0.4, 0.62),
        fins=(_fin(-1, 0.55, 0.85, (0, 0, 0), (0.3, 0.12, 0.08), (0.8, 0.13, 0.16), (1, 0, 0.05)),),
        pectoral=(0.25, 0.2, 0.08, 0.04),
        eye=(0.1, -0.15, 0.055),
        iris=(140, 150, 150),
        fin_alpha=0.85,
        ray_spacing=0.025,
        outline_alpha=0.4,
    ),
    "angelfish": Anatomy(
        body=0.70,
        top=(
            (0, 0),
            (0.03, 0.06),
            (0.15, 0.17),
            (0.35, 0.25),
            (0.6, 0.22),
            (0.85, 0.12),
            (1, 0.06),
        ),
        bottom=(
            (0, 0),
            (0.03, 0.05),
            (0.15, 0.17),
            (0.4, 0.26),
            (0.65, 0.22),
            (0.88, 0.11),
            (1, 0.06),
        ),
        caudal=Caudal("lyre", 0.28, 0.8),
        fins=(
            _fin(
                -1,
                0.28,
                0.92,
                (0, 0, 0),
                (0.25, 0.19, 0.12),
                (0.5, 0.3, 0.3),
                (0.6, 0.38, 0.56),
                (0.66, 0.27, 0.44),
                (0.85, 0.12, 0.24),
                (1, 0, 0.05),
            ),
            _fin(
                +1,
                0.3,
                0.92,
                (0, 0, 0),
                (0.25, 0.21, 0.12),
                (0.5, 0.33, 0.3),
                (0.6, 0.42, 0.58),
                (0.66, 0.3, 0.45),
                (0.85, 0.13, 0.25),
                (1, 0, 0.05),
            ),
            _fin(+1, 0.18, 0.21, (0, 0, 0), (0.5, 0.36, 0.2), (1, 0, 0.01)),
        ),
        pectoral=(0.3, 0.1, 0.11, 0.05),
        eye=(0.12, -0.15, 0.05),
        iris=(40, 50, 170),
        fin_alpha=0.6,
        ray_spacing=0.03,
    ),
}
"""Built-in species. Keys are valid values of ``FishSpec.species``."""


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _smoothstep(e0: float | F32, e1: float | F32, x: F32) -> F32:
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (u * u * (3.0 - 2.0 * u)).astype(np.float32)


def _mix(a: Color | F32, b: Color | F32, w: F32) -> F32:
    """Per-pixel linear blend of two colours (or colour fields)."""
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    ww = w[..., None]
    return (aa * (1.0 - ww) + bb * ww).astype(np.float32)


def _scale(c: Color, k: float, add: float = 0.0) -> Color:
    return (
        float(min(255.0, c[0] * k + add)),
        float(min(255.0, c[1] * k + add)),
        float(min(255.0, c[2] * k + add)),
    )


def _profile(points: Sequence[tuple[float, float]]) -> Callable[[F32], F32]:
    s = np.array([p[0] for p in points], dtype=np.float64)
    h = np.array([p[1] for p in points], dtype=np.float64)
    spline = PchipInterpolator(s, h)
    table_s = np.linspace(0.0, 1.0, 512)
    table_h = np.maximum(spline(table_s), 0.0)

    def evaluate(x: F32) -> F32:
        return np.interp(x, table_s, table_h).astype(np.float32)

    return evaluate


# ---------------------------------------------------------------------------
# Species patterns: body colour as a function of body coordinates (s, t)
# ---------------------------------------------------------------------------

Pattern = Callable[[F32, F32, "FishSpec"], F32]


def _belly(base: Color, t: F32, lift: float = 50.0, start: float = 0.15) -> F32:
    light = _scale(base, 1.05, lift)
    return _mix(base, light, _smoothstep(start, 1.0, t))


def _pattern_clownfish(s: F32, t: F32, spec: FishSpec) -> F32:
    col = _belly(spec.body_bgr, t, 35.0)
    white = (245.0, 248.0, 250.0)
    black = (25.0, 20.0, 20.0)
    bands = (
        (0.2 + 0.035 * t * t, 0.05),
        (0.5 - 0.05 * (1.0 - t * t), 0.06),
        (0.92, 0.04),
    )
    for center, half in bands:
        d = np.abs(s - center)
        edge = _smoothstep(half + 0.022, half + 0.012, d)
        col = _mix(col, black, edge)
        col = _mix(col, white, _smoothstep(half + 0.004, half - 0.004, d))
    return col


def _pattern_blue_tang(s: F32, t: F32, spec: FishSpec) -> F32:
    col = _mix(spec.body_bgr, _scale(spec.body_bgr, 1.1, 45.0), _smoothstep(0.3, 1.0, t))
    dark = (35.0, 18.0, 12.0)
    center = -0.5 + 0.45 * s + 0.25 * s * s
    width = 0.28 - 0.1 * s
    palette = _smoothstep(width + 0.04, width - 0.02, np.abs(t - center))
    palette *= _smoothstep(0.08, 0.16, s)
    # The blue "window" inside the black palette marking.
    hole = ((s - 0.47) / 0.15) ** 2 + ((t + 0.08) / 0.2) ** 2
    palette *= _smoothstep(0.8, 1.2, hole)
    dorsal_band = _smoothstep(-0.78, -0.88, t) * _smoothstep(0.1, 0.25, s)
    col = _mix(col, dark, np.maximum(palette, dorsal_band))
    yellow = spec.accent_bgr
    return _mix(col, yellow, _smoothstep(0.86, 0.94, s) * _smoothstep(-0.7, -0.2, t))


def _pattern_pufferfish(s: F32, t: F32, spec: FishSpec) -> F32:
    belly = (215.0, 235.0, 242.0)
    col = _mix(spec.body_bgr, belly, _smoothstep(-0.05, 0.45, t))
    rng = np.random.default_rng(1729)  # fixed: the same fish looks the same every run
    spots = np.zeros_like(s)
    for cs, ct, r in zip(
        rng.uniform(0.08, 0.98, 46),
        rng.uniform(-1.0, 0.25, 46),
        rng.uniform(0.022, 0.045, 46),
        strict=True,
    ):
        d = np.sqrt(((s - cs) * 1.0) ** 2 + ((t - ct) * 0.28) ** 2)
        spots = np.maximum(spots, _smoothstep(r, r * 0.55, d))
    spots *= _smoothstep(0.35, 0.05, t)
    return _mix(col, spec.accent_bgr, spots * 0.9)


def _pattern_yellow_tang(s: F32, t: F32, spec: FishSpec) -> F32:
    col = _mix(spec.body_bgr, _scale(spec.body_bgr, 1.0, 40.0), _smoothstep(0.2, 1.0, t))
    # Faint vertical striations and the white scalpel near the tail.
    col = (col * (0.98 + 0.02 * np.cos(s * 90.0))[..., None]).astype(np.float32)
    spine = ((s - 0.9) / 0.045) ** 2 + ((t - 0.05) / 0.22) ** 2
    return _mix(col, (240.0, 245.0, 245.0), _smoothstep(1.1, 0.7, spine))


def _pattern_betta(s: F32, t: F32, spec: FishSpec) -> F32:
    sheen = (0.5 + 0.5 * np.sin(s * 14.0 + t * 3.0)).astype(np.float32)
    col = _mix(spec.body_bgr, spec.accent_bgr, 0.45 * sheen * _smoothstep(0.1, 0.5, s))
    return _mix(col, _scale(spec.body_bgr, 1.1, 25.0), _smoothstep(0.4, 1.0, t) * 0.6)


def _pattern_neon_tetra(s: F32, t: F32, spec: FishSpec) -> F32:
    back = (110.0, 135.0, 120.0)
    belly = (225.0, 228.0, 230.0)
    col = _mix(back, belly, _smoothstep(-0.2, 0.5, t))
    red = spec.accent_bgr
    col = _mix(col, red, _smoothstep(0.42, 0.52, s) * _smoothstep(-0.05, 0.2, t))
    stripe_c = -0.18 + 0.08 * s
    stripe = _smoothstep(0.24, 0.12, np.abs(t - stripe_c)) * _smoothstep(0.1, 0.16, s)
    stripe *= _smoothstep(0.97, 0.88, s)
    col = _mix(col, spec.body_bgr, stripe)
    glow = _smoothstep(0.1, 0.0, np.abs(t - stripe_c)) * stripe
    return _mix(col, (255.0, 255.0, 220.0), (glow * 0.45).astype(np.float32))


def _pattern_guppy(s: F32, t: F32, spec: FishSpec) -> F32:
    col = _mix(spec.body_bgr, _scale(spec.body_bgr, 1.05, 50.0), _smoothstep(-0.2, 0.9, t))
    patch = np.exp(-(((s - 0.75) / 0.12) ** 2) - ((t + 0.1) / 0.5) ** 2).astype(np.float32)
    col = _mix(col, spec.accent_bgr, patch * 0.8)
    blue_patch = np.exp(-(((s - 0.5) / 0.08) ** 2) - ((t - 0.2) / 0.4) ** 2).astype(np.float32)
    return _mix(col, (220.0, 120.0, 40.0), blue_patch * 0.55)


def _pattern_angelfish(s: F32, t: F32, spec: FishSpec) -> F32:
    col = _mix(spec.body_bgr, _scale(spec.body_bgr, 1.03, 30.0), _smoothstep(-0.3, 1.0, t))
    bars = ((0.12, 0.035), (0.42 - 0.04 * t, 0.06), (0.74, 0.05), (0.97, 0.03))
    for center, half in bars:
        col = _mix(
            col, spec.accent_bgr, 0.88 * _smoothstep(half + 0.015, half - 0.01, np.abs(s - center))
        )
    return col


PATTERNS: dict[str, Pattern] = {
    "clownfish": _pattern_clownfish,
    "blue_tang": _pattern_blue_tang,
    "pufferfish": _pattern_pufferfish,
    "yellow_tang": _pattern_yellow_tang,
    "betta": _pattern_betta,
    "neon_tetra": _pattern_neon_tetra,
    "guppy": _pattern_guppy,
    "angelfish": _pattern_angelfish,
}


@dataclass(frozen=True, slots=True)
class FinStyle:
    """Fin colours: gradient from ``base`` (at the body) to ``tip``, plus edge."""

    base: Color
    tip: Color
    edge: Color


def fin_style(spec: FishSpec, kind: str) -> FinStyle:
    """Colours of one fin (``kind`` in caudal/dorsal/ventral/pectoral)."""
    body, accent = spec.body_bgr, spec.accent_bgr
    match spec.species:
        case "clownfish":
            return FinStyle(_scale(body, 1.05), _scale(body, 1.1, 10), (20.0, 15.0, 15.0))
        case "blue_tang":
            if kind in ("caudal", "pectoral"):
                return FinStyle(_scale(accent, 1.0), _scale(accent, 0.95), (60.0, 30.0, 15.0))
            return FinStyle(_scale(body, 0.85), _scale(body, 0.7), (25.0, 12.0, 8.0))
        case "pufferfish":
            return FinStyle((185.0, 215.0, 222.0), (210.0, 230.0, 236.0), (150.0, 180.0, 190.0))
        case "yellow_tang":
            return FinStyle(_scale(body, 0.97), _scale(body, 1.0, 15), _scale(body, 0.8))
        case "betta":
            return FinStyle(_scale(body, 1.0), _scale(accent, 1.0), _scale(accent, 1.1, 40))
        case "guppy":
            return FinStyle(_scale(accent, 1.0), (215.0, 110.0, 40.0), (235.0, 160.0, 90.0))
        case "angelfish":
            return FinStyle((205.0, 212.0, 212.0), (175.0, 185.0, 185.0), _scale(accent, 1.2))
        case _:
            return FinStyle((215.0, 225.0, 230.0), (230.0, 235.0, 235.0), (200.0, 210.0, 215.0))


FinPattern = Callable[[F32, F32, "FishSpec", str], tuple[Color, F32] | None]


def _fin_pattern_angelfish(x: F32, y: F32, spec: FishSpec, kind: str) -> tuple[Color, F32] | None:
    """The body bars continue into the tall dorsal and anal fins."""
    if kind not in ("dorsal", "ventral"):
        return None
    s = (0.5 - x) / ANATOMY["angelfish"].body
    bars = np.zeros_like(s)
    for center, half in ((0.42, 0.05), (0.74, 0.045)):
        bars = np.maximum(bars, _smoothstep(half + 0.03, half - 0.01, np.abs(s - center)))
    return _scale(spec.accent_bgr, 1.0), (bars * 0.75).astype(np.float32)


def _fin_pattern_guppy(x: F32, y: F32, spec: FishSpec, kind: str) -> tuple[Color, F32] | None:
    """Black spots sprinkled over the fancy tail and dorsal fin."""
    if kind not in ("caudal", "dorsal"):
        return None
    rng = np.random.default_rng(31337)  # fixed: the same fish looks the same every run
    spots = np.zeros_like(x)
    for cx, cy, r in zip(
        rng.uniform(-0.5, -0.1, 40),
        rng.uniform(-0.3, 0.3, 40),
        rng.uniform(0.012, 0.022, 40),
        strict=True,
    ):
        spots = np.maximum(spots, _smoothstep(r, r * 0.5, np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    return (25.0, 20.0, 30.0), spots * 0.8


FIN_PATTERNS: dict[str, FinPattern] = {
    "angelfish": _fin_pattern_angelfish,
    "guppy": _fin_pattern_guppy,
}


# ---------------------------------------------------------------------------
# Sprite construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FishSprite:
    """Pre-rendered phases of one fish facing +x.

    ``frames[k]`` is an ``(h, w, 4)`` float32 image of tail-beat phase
    ``2*pi*k/PHASES``: premultiplied BGR in 0..255 and alpha in 0..1, so
    compositing is ``dst * (1 - alpha) + bgr``. ``origin`` is the pixel of
    the fish's reference point (its centre) in those images; ``hulls[k]`` is the convex
    hull of the opaque part relative to ``origin`` (for ground-truth boxes).
    """

    frames: tuple[F32, ...]
    origin: tuple[float, float]
    hulls: tuple[npt.NDArray[np.float64], ...]
    length_px: float


class _Geometry:
    """Outline polygons of one species (body-length units).

    The body outline itself is static; animation lives in the fins, which
    is both cheaper to pre-render and closer to what a side-on camera sees
    (a fish's lateral undulation happens in the depth direction).
    """

    def __init__(self, anatomy: Anatomy) -> None:
        self.a = anatomy
        self.top = _profile(anatomy.top)
        self.bottom = _profile(anatomy.bottom)

    def x_of(self, s: F32) -> F32:
        return (0.5 - self.a.body * s).astype(np.float32)

    def point(self, s: float, t: float) -> tuple[float, float]:
        """Body coordinates ``(s, t)`` to L-unit ``(x, y)``."""
        ss = np.array([s], dtype=np.float32)
        half = self.top(ss)[0] if t < 0 else self.bottom(ss)[0]
        return float(self.x_of(ss)[0]), float(t * half)

    def body(self) -> npt.NDArray[np.float64]:
        s = np.linspace(0.0, 1.0, 72, dtype=np.float32)
        x = self.x_of(s)
        upper = np.stack([x, -self.top(s)], axis=1)
        lower = np.stack([x, self.bottom(s)], axis=1)[::-1]
        return np.concatenate([upper, lower]).astype(np.float64)

    def fin(
        self, fin: FinShape, phase: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (base points, outer points) of a dorsal/ventral fin."""
        n = 40
        u = np.linspace(0.0, 1.0, n, dtype=np.float32)
        us = np.array([p[0] for p in fin.outline])
        height = np.interp(u, us, [p[1] for p in fin.outline]).astype(np.float32)
        sweep = np.interp(u, us, [p[2] for p in fin.outline]).astype(np.float32)
        s = (fin.s0 + (fin.s1 - fin.s0) * u).astype(np.float32)
        x = self.x_of(s)
        edge = self.top(s) if fin.side < 0 else self.bottom(s)
        # Start the fin slightly inside the body so no seam shows.
        base_y = fin.side * edge * 0.9
        ripple = 1.0 + 0.07 * np.sin(phase + 6.0 * u)
        outer_x = x - sweep * (0.85 + 0.15 * np.cos(phase - 2.0 * u))
        outer_y = base_y + fin.side * height * ripple
        base = np.stack([x, base_y], axis=1).astype(np.float64)
        outer = np.stack([outer_x, outer_y], axis=1).astype(np.float64)
        return base, outer

    def caudal(self, phase: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (base points, outer points) of the tail fan.

        The sweep is drawn as foreshortening of the fan (it swings towards
        and away from the camera) plus a slight vertical flick.
        """
        c = self.a.caudal
        n = 48
        u = np.linspace(-1.0, 1.0, n)
        theta = u * c.spread
        absu = np.abs(u)
        if c.shape == "round":
            r = c.length * (0.72 + 0.28 * np.cos(u * np.pi / 2.0))
        elif c.shape == "fork":
            r = c.length * (0.5 + 0.5 * absu**1.3)
        elif c.shape == "truncate":
            r = c.length * (0.82 + 0.18 * absu**2) / np.cos(theta * 0.75)
        elif c.shape == "lyre":
            r = c.length * (0.55 + 0.45 * absu**2.5 + 0.35 * np.clip(absu - 0.85, 0.0, None) / 0.15)
        else:  # flowing
            r = c.length * (0.88 + 0.12 * np.cos(u * np.pi / 2.0) + 0.05 * np.sin(9.0 * u + phase))
        x_p, _ = self.point(1.0, 0.0)
        x_p += 0.025
        h_p = float(self.top(np.array([1.0], dtype=np.float32))[0])
        sweep = 0.74 + 0.26 * math.cos(phase)
        flick = 0.04 * math.sin(phase)
        base_y = 0.85 * h_p * u
        outer_x = x_p - r * np.cos(theta) * sweep
        outer_y = base_y + r * np.sin(theta) + flick * (r / c.length) ** 2
        base = np.stack([np.full(n, x_p), base_y], axis=1)
        outer = np.stack([outer_x, outer_y], axis=1)
        return base, outer

    def pectoral(self, phase: float) -> npt.NDArray[np.float64]:
        s, t, length, width = self.a.pectoral
        x0, y0 = self.point(s, t)
        angle = math.pi - 0.45 + 0.35 * math.sin(2.0 * phase)
        flap = 0.7 + 0.3 * math.cos(2.0 * phase)
        k = np.linspace(0.0, 1.0, 24)
        along = np.concatenate([k, k[::-1]]) * length * flap
        across = (
            np.concatenate([np.sin(np.pi * k**0.7), -np.sin(np.pi * k[::-1] ** 0.7)]) * width * 0.5
        )
        ca, sa = math.cos(angle), math.sin(angle)
        x = x0 + along * ca - across * sa
        y = y0 + along * sa + across * ca
        return np.stack([x, y], axis=1)


Roi = tuple[slice, slice]


class _Canvas:
    """Premultiplied BGR + alpha accumulation buffer (geometry in L units).

    Every layer is composited only inside the bounding box of its polygon,
    which keeps sprite construction fast (most fins cover a small part of
    the canvas).
    """

    def __init__(self, width: int, height: int, scale: float, origin: tuple[float, float]) -> None:
        self.w, self.h = width, height
        self.scale = scale  # pixels per body length (supersampled)
        self.origin = origin  # pixel position of (0, 0)
        self.color = np.zeros((height, width, 3), np.float32)
        self.alpha = np.zeros((height, width), np.float32)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self.xg = (xs - origin[0]) / scale
        self.yg = (ys - origin[1]) / scale

    def to_px(self, pts: npt.NDArray[np.float64], roi: Roi) -> npt.NDArray[np.int32]:
        offset = np.array([self.origin[0] - roi[1].start, self.origin[1] - roi[0].start])
        return np.round((pts * self.scale + offset) * _ONE).astype(np.int32)

    def roi(self, polys: Sequence[npt.NDArray[np.float64]], pad: float = 0.03) -> Roi:
        pts = np.concatenate(polys) * self.scale + np.array(self.origin)
        x0, y0 = np.floor(pts.min(axis=0) - pad * self.scale).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0) + pad * self.scale).astype(int)
        return slice(max(0, y0), min(self.h, y1 + 1)), slice(max(0, x0), min(self.w, x1 + 1))

    def mask(self, polys: Sequence[npt.NDArray[np.float64]], roi: Roi) -> F32:
        m = np.zeros((roi[0].stop - roi[0].start, roi[1].stop - roi[1].start), np.uint8)
        cv2.fillPoly(m, [self.to_px(p, roi) for p in polys], 255, cv2.LINE_AA, _SHIFT)
        return m.astype(np.float32) * (1.0 / 255.0)

    def lines(self, lines: Sequence[npt.NDArray[np.float64]], thickness_l: float, roi: Roi) -> F32:
        m = np.zeros((roi[0].stop - roi[0].start, roi[1].stop - roi[1].start), np.uint8)
        thickness = max(1, round(thickness_l * self.scale))
        cv2.polylines(
            m, [self.to_px(p, roi) for p in lines], False, 255, thickness, cv2.LINE_AA, _SHIFT
        )
        return m.astype(np.float32) * (1.0 / 255.0)

    def over(self, roi: Roi, coverage: F32, color: Color | F32, opacity: float = 1.0) -> None:
        """Composite ``color`` with per-pixel ``coverage`` over the buffer."""
        a = coverage * opacity
        col = np.asarray(color, dtype=np.float32)
        dst = self.color[roi]
        dst *= (1.0 - a)[..., None]
        dst += col * a[..., None]
        self.alpha[roi] = a + self.alpha[roi] * (1.0 - a)

    def disc(
        self, center: tuple[float, float], radius: float, color: Color, opacity: float = 1.0
    ) -> None:
        ring = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        poly = np.array(center) + radius * np.stack([np.cos(ring), np.sin(ring)], axis=1)
        roi = self.roi([poly])
        self.over(roi, self.mask([poly], roi), color, opacity)


def _rays(
    base: npt.NDArray[np.float64], outer: npt.NDArray[np.float64], spacing: float
) -> list[npt.NDArray[np.float64]]:
    """Fin rays: segments from the fin base towards its edge."""
    if spacing <= 0:
        return []
    along = outer if np.allclose(base[0], base[-1]) else base
    pos = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(along, axis=0), axis=1))])
    count = max(2, int(pos[-1] / spacing))
    idx = np.clip(
        np.searchsorted(pos, np.linspace(0.0, pos[-1], count + 2)[1:-1]), 0, len(base) - 1
    )
    return [np.stack([base[i], base[i] + 0.92 * (outer[i] - base[i])]) for i in idx]


def _paint_fin(
    spec: FishSpec,
    canvas: _Canvas,
    kind: str,
    base: npt.NDArray[np.float64],
    outer: npt.NDArray[np.float64],
) -> None:
    a = ANATOMY[spec.species]
    style = fin_style(spec, kind)
    poly = np.concatenate([base, outer[::-1]])
    roi = canvas.roi([poly])
    cover = canvas.mask([poly], roi)
    xg, yg = canvas.xg[roi], canvas.yg[roi]
    # Colour gradient from the root of the fin to its edge.
    root = base.mean(axis=0)
    reach = float(np.max(np.linalg.norm(outer - root, axis=1))) or 1.0
    dist = np.clip(np.sqrt((xg - root[0]) ** 2 + (yg - root[1]) ** 2) / reach, 0.0, 1.0)
    color = _mix(style.base, style.tip, dist.astype(np.float32))
    pattern = FIN_PATTERNS.get(spec.species)
    overlay = pattern(xg, yg, spec, kind) if pattern is not None else None
    if overlay is not None:
        color = _mix(color, overlay[0], overlay[1])
    canvas.over(roi, cover, color, a.fin_alpha)
    rays = _rays(base, outer, a.ray_spacing)
    if rays:
        canvas.over(roi, canvas.lines(rays, 0.006, roi) * cover, _scale(style.base, 0.7), 0.35)
    canvas.over(roi, canvas.lines([outer], 0.012, roi), style.edge, 0.7 * a.fin_alpha)


def _paint_body(spec: FishSpec, geo: _Geometry, canvas: _Canvas) -> None:
    """Body with pattern, shading, gill, outline and eye (phase independent)."""
    a = geo.a
    body_poly = geo.body()
    roi = canvas.roi([body_poly])
    cover = canvas.mask([body_poly], roi)
    xg, yg = canvas.xg[roi], canvas.yg[roi]
    s = np.clip((0.5 - xg) / a.body, 0.0, 1.0).astype(np.float32)
    half = np.where(yg < 0, geo.top(s), geo.bottom(s))
    t = np.clip(yg / np.maximum(half, 1e-4), -1.0, 1.0).astype(np.float32)
    color = PATTERNS[spec.species](s, t, spec)
    # Cylinder lit from above and in front: diffuse term plus a soft
    # specular streak along the upper flank.
    nz = np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))
    diffuse = np.clip(-0.42 * t + 0.9 * nz, 0.0, 1.0)
    shade = (0.5 + 0.62 * diffuse) * (0.8 + 0.2 * _smoothstep(0.0, 0.1, s))
    gloss = (
        0.3
        * np.exp(-(((t + 0.5) / 0.2) ** 2))
        * _smoothstep(0.06, 0.3, s)
        * _smoothstep(0.95, 0.55, s)
    )
    shaded = np.clip(color * shade[..., None] + (255.0 * gloss)[..., None], 0.0, 255.0)
    canvas.over(roi, cover, shaded.astype(np.float32))

    # Gill cover arc and a darker outline give the silhouette definition.
    gill_t = np.linspace(-0.75, 0.8, 20)
    gill = np.array(
        [geo.point(0.2 + 0.02 * a.eye[2] / 0.05 + 0.05 * (1 - tt * tt), tt) for tt in gill_t]
    )
    canvas.over(roi, canvas.lines([gill], 0.008, roi) * cover, _scale(spec.body_bgr, 0.45), 0.35)
    closed = np.concatenate([body_poly, body_poly[:1]])
    canvas.over(
        roi, canvas.lines([closed], 0.01, roi), _scale(spec.body_bgr, 0.35), a.outline_alpha
    )

    # Eye: dark rim, iris, pupil and a catch-light.
    es, et, er = a.eye
    ex, ey = geo.point(es, et)
    canvas.disc((ex, ey), er * 1.15, _scale(spec.body_bgr, 0.4), 0.6)
    canvas.disc((ex, ey), er, a.iris)
    canvas.disc((ex + 0.08 * er, ey), er * 0.6, (12.0, 10.0, 10.0))
    canvas.disc((ex + 0.35 * er, ey - 0.35 * er), er * 0.25, (255.0, 255.0, 255.0), 0.9)


def _bounds(geo: _Geometry) -> tuple[float, float, float, float]:
    pts: list[npt.NDArray[np.float64]] = [geo.body()]
    for k in range(PHASES):
        phase = 2.0 * math.pi * k / PHASES
        for base, outer in [geo.caudal(phase), *(geo.fin(f, phase) for f in geo.a.fins)]:
            pts += [base, outer]
        pts.append(geo.pectoral(phase))
    allp = np.concatenate(pts)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


@functools.lru_cache(maxsize=64)
def build_sprite(spec: FishSpec) -> FishSprite:
    """Render all animation phases of ``spec`` (cached per spec)."""
    anatomy = ANATOMY[spec.species]
    geo = _Geometry(anatomy)
    x0, y0, x1, y1 = _bounds(geo)
    pad = 2.0 / spec.length_px  # two output pixels of margin
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    out_w = math.ceil((x1 - x0) * spec.length_px)
    out_h = math.ceil((y1 - y0) * spec.length_px)
    origin = (-x0 * spec.length_px, -y0 * spec.length_px)
    scale = spec.length_px * SUPERSAMPLE
    size = (out_w * SUPERSAMPLE, out_h * SUPERSAMPLE)
    origin_ss = (origin[0] * SUPERSAMPLE, origin[1] * SUPERSAMPLE)

    body = _Canvas(*size, scale, origin_ss)
    _paint_body(spec, geo, body)
    frames: list[F32] = []
    hulls: list[npt.NDArray[np.float64]] = []
    for k in range(PHASES):
        phase = 2.0 * math.pi * k / PHASES
        canvas = _Canvas(*size, scale, origin_ss)
        _paint_fin(spec, canvas, "caudal", *geo.caudal(phase))
        for fin in anatomy.fins:
            _paint_fin(spec, canvas, "dorsal" if fin.side < 0 else "ventral", *geo.fin(fin, phase))
        # Static body over the animated fins, then the fluttering pectoral.
        canvas.color = body.color + canvas.color * (1.0 - body.alpha)[..., None]
        canvas.alpha = body.alpha + canvas.alpha * (1.0 - body.alpha)
        pect = geo.pectoral(phase)
        style = fin_style(spec, "pectoral")
        roi = canvas.roi([pect])
        canvas.over(
            roi,
            canvas.mask([pect], roi),
            _scale(style.base, 1.0, 25.0),
            0.5 * max(anatomy.fin_alpha, 0.6),
        )
        canvas.over(roi, canvas.lines([pect], 0.006, roi), style.edge, 0.3)

        premult = np.dstack([canvas.color, canvas.alpha])
        small = cv2.resize(premult, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        small.setflags(write=False)
        frames.append(small)
        ys, xs = np.nonzero(small[..., 3] > ALPHA_HULL_THRESHOLD)
        pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64) - np.array(origin)
        hull.setflags(write=False)
        hulls.append(hull)
    return FishSprite(tuple(frames), origin, tuple(hulls), spec.length_px)


MIN_FORESHORTEN = 0.22
"""Narrowest apparent width (fraction of length) of a fish turning edge-on."""


def draw_sprite(
    img: npt.NDArray[np.uint8],
    sprite: FishSprite,
    *,
    phase: float,
    center: tuple[float, float],
    facing: float,
    pitch: float,
    scale: float,
    haze: float,
    haze_bgr: F32,
) -> tuple[float, float, float, float] | None:
    """Composite one fish onto a BGR frame; return its visible bounding box.

    ``facing`` in [-1, 1] is the cosine of the fish's yaw (1 = facing
    right): its sign mirrors the sprite, its magnitude foreshortens it, so a
    turning fish narrows to edge-on and opens up the other way. ``pitch``
    tilts the nose (positive = down) without ever turning the fish upside
    down. ``scale`` (depth) and ``haze`` (blend towards the water colour)
    place the fish nearer or farther from the glass.
    """
    k = int(phase / (2.0 * math.pi) * PHASES) % PHASES
    frame = sprite.frames[k]
    sign = 1.0 if facing >= 0 else -1.0
    sx = max(abs(facing), MIN_FORESHORTEN)
    ca, sa = math.cos(pitch), math.sin(pitch)
    forward = (sign * ca, sa)
    up = (sign * sa, -ca)
    a = scale * np.array([[sx * forward[0], -up[0]], [sx * forward[1], -up[1]]])
    c = np.array(center)
    tvec = c - a @ np.array(sprite.origin)
    fh, fw = frame.shape[:2]
    corners = a @ np.array([[0.0, fw, 0.0, fw], [0.0, 0.0, fh, fh]]) + tvec[:, None]
    ih, iw = img.shape[:2]
    x0, y0 = max(0, math.floor(corners[0].min())), max(0, math.floor(corners[1].min()))
    x1, y1 = min(iw, math.ceil(corners[0].max()) + 1), min(ih, math.ceil(corners[1].max()) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    m = np.hstack([a, (tvec - np.array([x0, y0]))[:, None]])
    warped = cv2.warpAffine(
        frame,
        m,
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = warped[..., 3:4]
    color = warped[..., :3]
    if haze > 0.0:
        color = color * (1.0 - haze) + (haze * haze_bgr) * alpha
    roi = img[y0:y1, x0:x1]
    roi[...] = cv2.convertScaleAbs(roi.astype(np.float32) * (1.0 - alpha) + color)
    hull = a @ sprite.hulls[k].T + c[:, None]
    bx0, by0 = max(0.0, float(hull[0].min())), max(0.0, float(hull[1].min()))
    bx1, by1 = min(float(iw), float(hull[0].max())), min(float(ih), float(hull[1].max()))
    if bx0 >= bx1 or by0 >= by1:
        return None
    return bx0, by0, bx1, by1
