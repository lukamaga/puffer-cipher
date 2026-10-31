"""Synthetic aquarium video for liveness tests.

Self-contained on purpose (the real simulator lives in ``puffer.sources`` and
is developed independently): a textured "water" background, a fish swimming
around, sensor noise, an LED disc that shows *any* state function of time
with a configurable camera latency and exposure blur, a decoy light blinking
on its own random pattern, and the usual failure modes - occlusions, glare,
noisy frames, dropped frames, timestamp jitter, a covered LED, slow lighting
drift and grayscale night (IR) frames.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from puffer.liveness import ChallengeSchedule, Encoding, LivenessVerifier, Roi
from puffer.types import Frame, LightMode, LivenessState, LivenessStatus

StateFn = Callable[[float], bool]

_FISH = np.array([30.0, 110.0, 200.0], dtype=np.float32)  # orange fish (BGR)
_COVER = np.array([40.0, 90.0, 40.0], dtype=np.float32)  # a plant leaf in front of the LED


@dataclass(frozen=True)
class Scene:
    width: int = 128
    height: int = 96
    led_center: tuple[int, int] = (100, 22)
    led_radius: int = 5
    decoy_center: tuple[int, int] | None = (26, 72)
    mode: LightMode = LightMode.DAY
    noise_sigma: float = 3.0
    drift_amplitude: float = 0.0  # +- grey levels of ambient light
    drift_period_s: float = 40.0
    fish: bool = True

    @property
    def led_roi(self) -> Roi:
        """Largest square fully inside the LED disc."""
        s = math.floor(self.led_radius / math.sqrt(2))
        cx, cy = self.led_center
        return (cx - s, cy - s, 2 * s + 1, 2 * s + 1)

    def colours(self) -> tuple[npt.NDArray[np.float32], ...]:
        """(water, led_on, led_off, decoy_on, decoy_off) as BGR float32."""
        if self.mode is LightMode.NIGHT:  # IR: everything is grey, the IR LED is bright
            grey = [35.0, 210.0, 50.0, 200.0, 45.0]
            return tuple(np.full(3, g, dtype=np.float32) for g in grey)
        bgr = [(110, 90, 50), (60, 70, 250), (60, 60, 80), (40, 240, 60), (40, 70, 50)]
        return tuple(np.array(c, dtype=np.float32) for c in bgr)


def seeded_schedule(
    seed: int, *, symbol_s: float = 0.2, encoding: Encoding = "manchester", t0: float = 0.0
) -> ChallengeSchedule:
    """A schedule whose bits come from a seeded generator (reproducible, never secret)."""
    gen = np.random.default_rng(seed)
    return ChallengeSchedule(symbol_s, encoding, t0, random_bytes=gen.bytes)


def render(
    led: StateFn,
    *,
    rng: np.random.Generator,
    fps: float = 30.0,
    duration_s: float = 20.0,
    start_s: float = 0.0,
    latency_s: float = 0.0,
    scene: Scene | None = None,
    exposure_s: float = 0.004,
    decoy: StateFn | None = None,
    corrupt_prob: float = 0.0,
    drop_prob: float = 0.0,
    capture_jitter_s: float = 0.0,
    timestamp_noise_s: float = 0.0,
    covered: Callable[[float], bool] | None = None,
) -> Iterator[Frame]:
    """Yield frames captured at ``fps`` showing ``led(t - latency_s)``.

    ``corrupt_prob`` corrupts that fraction of frames with an occlusion (fish in
    front of the LED), glare (LED region saturated) or heavy sensor noise.
    ``covered(t)`` hides the LED behind something constant at capture time ``t``.
    """
    sc = scene or Scene()
    water, led_on, led_off, decoy_on, decoy_off = sc.colours()
    h, w = sc.height, sc.width
    yy, xx = np.mgrid[0:h, 0:w]
    background = _texture(rng, h, w, sc.mode) + water
    led_mask = _disc(xx, yy, sc.led_center, sc.led_radius)
    decoy_mask = None if sc.decoy_center is None else _disc(xx, yy, sc.decoy_center, 4)

    def shown(state: StateFn, t_capture: float) -> float:
        """Fraction of the exposure during which ``state`` was on (motion blur in time)."""
        t_end = t_capture - latency_s
        subs = [state(t_end - exposure_s * (j + 0.5) / 4) for j in range(4)]
        return sum(subs) / 4.0

    n_frames = round(duration_s * fps)
    for n in range(n_frames):
        if drop_prob and rng.random() < drop_prob:
            continue
        t_capture = start_s + n / fps
        if capture_jitter_s:
            t_capture += float(rng.uniform(-capture_jitter_s, capture_jitter_s))
        img = background.copy()
        if sc.fish:
            cx = w * (0.45 + 0.25 * math.sin(2 * math.pi * t_capture / 7.0))
            cy = h * (0.6 + 0.2 * math.sin(2 * math.pi * t_capture / 5.3))
            img[((xx - cx) / 10.0) ** 2 + ((yy - cy) / 5.0) ** 2 <= 1.0] = _FISH
        if decoy is not None and decoy_mask is not None:
            f = shown(decoy, t_capture)
            img[decoy_mask] = decoy_off + f * (decoy_on - decoy_off)
        if covered is not None and covered(t_capture):
            img[led_mask] = _COVER if sc.mode is LightMode.DAY else _COVER.mean()
        else:
            f = shown(led, t_capture)
            img[led_mask] = led_off + f * (led_on - led_off)

        sigma = sc.noise_sigma
        if corrupt_prob and rng.random() < corrupt_prob:
            kind = int(rng.integers(3))
            if kind == 0:
                img[led_mask] = _FISH
            elif kind == 1:
                img[led_mask] = 255.0
            else:
                sigma = 40.0
        if sc.drift_amplitude:
            img += sc.drift_amplitude * math.sin(2 * math.pi * t_capture / sc.drift_period_s)
        if sc.mode is LightMode.NIGHT:
            grey = img[..., 0] + rng.standard_normal((h, w), dtype=np.float32) * sigma
            img = np.repeat(grey[..., None], 3, axis=2)
        else:
            img += rng.standard_normal((h, w, 3), dtype=np.float32) * sigma
        image = np.clip(img, 0, 255).astype(np.uint8)

        timestamp = t_capture
        if timestamp_noise_s:
            timestamp += float(rng.normal(0.0, timestamp_noise_s))
        yield Frame(
            index=n,
            timestamp=timestamp,
            wall_time=1.79e9 + timestamp,
            image=image,
            mode=sc.mode,
            source_id="test-sim",
        )


def run(verifier: LivenessVerifier, frames: Iterable[Frame]) -> list[LivenessStatus]:
    return [verifier.process(frame) for frame in frames]


def fraction(statuses: list[LivenessStatus], state: LivenessState) -> float:
    return sum(s.state is state for s in statuses) / max(1, len(statuses))


def overlap(a: Roi, b: Roi) -> float:
    """Intersection area divided by the area of ``a``."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3])


def _disc(
    xx: npt.NDArray[np.int_], yy: npt.NDArray[np.int_], center: tuple[int, int], radius: int
) -> npt.NDArray[np.bool_]:
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


def _texture(rng: np.random.Generator, h: int, w: int, mode: LightMode) -> npt.NDArray[np.float32]:
    """Smooth static texture (gravel, plants, glass reflections)."""
    amplitude = 10.0 if mode is LightMode.NIGHT else 20.0
    coarse = rng.uniform(-amplitude, amplitude, size=(max(2, h // 8), max(2, w // 8), 3))
    smooth = cv2.resize(coarse.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.asarray(smooth, dtype=np.float32)
