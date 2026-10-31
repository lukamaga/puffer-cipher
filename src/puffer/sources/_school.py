"""Fish behaviour model for the aquarium simulator.

The goal is motion that a tracker finds *realistic* and an observer finds
*alive*: smooth most of the time, occasionally surprising. Each fish is a
point with a velocity driven by a few interacting stochastic processes
(all fish are updated together with numpy, units are pixels and seconds):

* **Ornstein-Uhlenbeck velocity.** ``dv = theta (v_target - v) dt + F dt +
  sigma dW``: the velocity relaxes towards a target velocity with
  correlation time ``1/theta`` while being kicked by white noise. That gives
  the smooth-but-unpredictable trajectories real fish have (and that make the
  tracker's Kalman innovations a meaningful entropy source).
* **Wandering preferred heading.** The direction of ``v_target`` is a
  random walk (``wander`` rad/sqrt(s)) with a gentle pull towards horizontal,
  because fish mostly cruise level and turn rather than climb.
* **Darts and pauses.** Poisson-timed events: a dart (rate grows with
  ``burstiness``) briefly multiplies the target speed by 2.5-4.5 with a
  sudden turn and a stiffer response; a pause drops it to ~10 % (hovering).
* **Soft constraints as forces** ``F``: quadratic repulsion from the walls,
  surface and sand, keep-out disks (the LED housing), a weak spring towards
  the species' preferred depth, and gentle mutual avoidance between fish at
  similar distance from the camera. The preferred heading is continuously
  re-aligned with ``v_target + F``, so fish *turn away* from walls instead of
  pressing against them.
* **Sleep.** An ``awake`` level follows the light with a lag of a few
  seconds; activity scales with it (about 80 % slower at night), darts all
  but stop, and fish sink towards a resting depth and mostly hover.

Rendering-only state is integrated here too: depth ``z`` (0 = near the
glass, 1 = far; drives scale and haze), a smoothed facing direction (fish
turn around through a foreshortened "edge-on" pose), smoothed pitch, and
the tail-beat phase whose frequency grows with speed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from puffer.sources.synthetic import FishSpec

F64 = npt.NDArray[np.float64]

_THETA_CRUISE = 1.6  # 1/s, velocity correlation rate while cruising
_THETA_DART = 6.0  # 1/s, stiffer response during a dart
_WALL_ZONE = 48.0  # px at reference scale: width of the soft wall zone
_WALL_ACCEL = 260.0  # px/s^2 at reference scale, at the wall itself
_NEIGHBOUR_ACCEL = 140.0  # px/s^2 at reference scale
_DEPTH_ACCEL = 24.0  # px/s^2 at reference scale, preferred-depth spring
_ALIGN_RATE = 2.5  # 1/s, how fast the preferred heading follows the steering
_LEVEL_RATE = 0.5  # 1/s, pull of the heading towards horizontal
_TURN_RATE = 5.0  # facing change per second (1 -> -1 takes ~0.4 s)
_PITCH_TAU = 0.18  # s, pitch smoothing
_SLEEP_TAU = 6.0  # s, lag of the awake level behind the light
_NIGHT_ACTIVITY = 0.2  # activity multiplier when fully asleep


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned region, pixels."""

    x0: float
    y0: float
    x1: float
    y1: float


class School:
    """State and dynamics of every fish in the tank.

    ``bounds`` is the region fish *centres* may occupy (the visible water
    column); per-fish margins proportional to body length are applied on
    top. ``keep_out`` lists ``(x, y, radius)`` disks fish steer around.
    ``scale`` converts reference-resolution constants to this frame size.
    """

    def __init__(
        self,
        specs: Sequence[FishSpec],
        bounds: Box,
        *,
        scale: float,
        rng: np.random.Generator,
        keep_out: Sequence[tuple[float, float, float]] = (),
        awake: float = 1.0,
    ) -> None:
        self.rng = rng
        self.scale = scale
        self.n = n = len(specs)
        self.length = np.array([s.length_px for s in specs], dtype=np.float64) * scale
        self.cruise = np.array([s.speed_px_s for s in specs], dtype=np.float64) * scale
        self.burstiness = np.array([s.burstiness for s in specs], dtype=np.float64)
        self.depth_pref = np.array([s.depth_pref for s in specs], dtype=np.float64)
        self.wander = np.array([s.wander for s in specs], dtype=np.float64)
        self.keep_out = np.array(keep_out, dtype=np.float64).reshape(-1, 3)

        # Per-fish box for the centre: half a body length (plus a little)
        # from every edge, so the whole fish stays in the water.
        margin_x = 0.5 * self.length + 4.0 * scale
        margin_y = 0.3 * self.length + 4.0 * scale
        self.lo = np.stack([bounds.x0 + margin_x, bounds.y0 + margin_y], axis=1)
        self.hi = np.stack([bounds.x1 - margin_x, bounds.y1 - margin_y], axis=1)
        self.hi = np.maximum(self.hi, self.lo + 1.0)

        # Initial state: spread over the tank near each species' depth.
        span = self.hi - self.lo
        self.pos = self.lo + span * np.stack(
            [
                rng.uniform(0.1, 0.9, n),
                np.clip(self.depth_pref + rng.normal(0.0, 0.12, n), 0.05, 0.95),
            ],
            axis=1,
        )
        self.psi = rng.choice([0.0, math.pi], n) + rng.normal(0.0, 0.25, n)
        self.vel = (
            0.5 * self.cruise[:, None] * np.stack([np.cos(self.psi), np.sin(self.psi)], axis=1)
        )
        self.z = rng.uniform(0.1, 0.9, n)
        self.z_home = self.z.copy()
        self.facing = np.sign(np.cos(self.psi))
        self.facing[self.facing == 0] = 1.0
        self.pitch = np.zeros(n)
        self.phase = rng.uniform(0.0, 2.0 * math.pi, n)
        self.mood = rng.normal(0.0, 0.5, n)
        self.depth_offset = rng.normal(0.0, 0.08, n)
        self.dart_t = np.zeros(n)
        self.pause_t = np.zeros(n)
        self.awake = float(awake)

    # ------------------------------------------------------------------ forces

    def _wall_force(self) -> F64:
        zone = _WALL_ZONE * self.scale
        accel = _WALL_ACCEL * self.scale
        d_lo = np.clip((zone - (self.pos - self.lo)) / zone, 0.0, None)
        d_hi = np.clip((zone - (self.hi - self.pos)) / zone, 0.0, None)
        force: F64 = accel * (d_lo**2 - d_hi**2)
        return force

    def _keep_out_force(self) -> F64:
        force = np.zeros((self.n, 2))
        zone = _WALL_ZONE * self.scale
        for cx, cy, radius in self.keep_out:
            delta = self.pos - np.array([cx, cy])
            dist = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-6)
            reach = radius + 0.5 * self.length + zone
            push = np.clip((reach - dist) / zone, 0.0, None) ** 2
            force += (_WALL_ACCEL * self.scale * push / dist)[:, None] * delta
        return force

    def _neighbour_force(self) -> F64:
        if self.n < 2:
            return np.zeros((self.n, 2))
        delta = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.maximum(np.hypot(delta[..., 0], delta[..., 1]), 1e-6)
        reach = 0.6 * (self.length[:, None] + self.length[None, :])
        # Fish at clearly different depths may overlap on screen (occlusion).
        same_depth = np.clip(1.0 - np.abs(self.z[:, None] - self.z[None, :]) / 0.35, 0.0, 1.0)
        push = np.clip(1.0 - dist / reach, 0.0, None) * same_depth
        np.fill_diagonal(push, 0.0)
        force: F64 = _NEIGHBOUR_ACCEL * self.scale * ((push / dist)[..., None] * delta).sum(axis=1)
        return force

    def _depth_force(self, awake: float) -> F64:
        # Asleep, fish sink towards a resting depth in the lower water.
        pref = self.depth_pref + self.depth_offset
        pref = awake * pref + (1.0 - awake) * (0.72 + 0.2 * self.depth_pref)
        target_y = self.lo[:, 1] + np.clip(pref, 0.0, 1.0) * (self.hi[:, 1] - self.lo[:, 1])
        band = 0.2 * (self.hi[:, 1] - self.lo[:, 1])
        force = np.zeros((self.n, 2))
        force[:, 1] = (
            _DEPTH_ACCEL * self.scale * np.clip((target_y - self.pos[:, 1]) / band, -1.0, 1.0)
        )
        return force

    # -------------------------------------------------------------------- step

    def step(self, dt: float, light: float) -> None:
        """Advance the simulation by ``dt`` seconds at light level ``light``."""
        if self.n == 0 or dt <= 0.0:
            return
        # Integrate in sub-steps of at most 1/30 s so low frame rates stay stable.
        sub = max(1, math.ceil(dt * 30.0))
        for _ in range(sub):
            self._substep(dt / sub, light)

    def _substep(self, dt: float, light: float) -> None:
        rng, n = self.rng, self.n
        self.awake += (light - self.awake) * (1.0 - math.exp(-dt / _SLEEP_TAU))
        activity = _NIGHT_ACTIVITY + (1.0 - _NIGHT_ACTIVITY) * self.awake

        # Poisson-timed darts and pauses.
        idle = (self.dart_t <= 0.0) & (self.pause_t <= 0.0)
        dart_rate = (0.04 + 0.45 * self.burstiness) * activity**3
        darts = idle & (rng.random(n) < -np.expm1(-dart_rate * dt))
        k = int(darts.sum())
        if k:
            self.dart_t[darts] = rng.uniform(0.25, 0.7, k)
            self.psi[darts] += rng.normal(0.0, 1.1, k)
        pause_rate = 0.05 + 0.25 * (1.0 - self.awake)
        pauses = idle & ~darts & (rng.random(n) < -np.expm1(-pause_rate * dt))
        k = int(pauses.sum())
        if k:
            self.pause_t[pauses] = rng.uniform(0.8, 3.5, k)
        self.dart_t -= dt
        self.pause_t -= dt
        darting = self.dart_t > 0.0
        pausing = self.pause_t > 0.0

        # Target speed: cruise modulated by a slow "mood", darts and pauses.
        self.mood += -self.mood * dt / 5.0 + 0.3 * math.sqrt(dt) * rng.standard_normal(n)
        speed = self.cruise * activity * (1.0 + 0.3 * np.tanh(self.mood))
        speed = np.where(darting, self.cruise * activity * (2.5 + 2.0 * self.burstiness), speed)
        speed = np.where(pausing, 0.1 * speed, speed)

        # Preferred heading: random walk plus a pull towards horizontal.
        self.psi += self.wander * math.sqrt(activity * dt) * rng.standard_normal(n)
        self.psi -= _LEVEL_RATE * np.sin(2.0 * self.psi) * dt

        force = self._wall_force() + self._keep_out_force() + self._neighbour_force()
        force += self._depth_force(self.awake)
        target = speed[:, None] * np.stack([np.cos(self.psi), np.sin(self.psi)], axis=1)

        # Turn the preferred heading towards where the forces steer.
        steer = target + 0.6 * force
        desired = np.arctan2(steer[:, 1], steer[:, 0])
        turn = (desired - self.psi + math.pi) % (2.0 * math.pi) - math.pi
        self.psi += np.clip(_ALIGN_RATE * dt * (1.0 + 2.0 * darting), 0.0, 1.0) * turn

        # Ornstein-Uhlenbeck velocity update.
        theta = np.where(darting, _THETA_DART, _THETA_CRUISE)
        sigma = 0.35 * self.cruise * activity * np.sqrt(theta)
        noise = rng.standard_normal((n, 2)) * (sigma * math.sqrt(dt))[:, None]
        self.vel += (theta * dt)[:, None] * (target - self.vel) + force * dt + noise
        max_speed = self.cruise * (3.0 + 2.5 * self.burstiness)
        norm = np.hypot(self.vel[:, 0], self.vel[:, 1])
        too_fast = norm > max_speed
        self.vel[too_fast] *= (max_speed[too_fast] / norm[too_fast])[:, None]
        self.pos += self.vel * dt

        # Hard limits as a last resort: bounce off the box.
        below, above = self.pos < self.lo, self.pos > self.hi
        self.vel[below | above] *= -0.5
        np.clip(self.pos, self.lo, self.hi, out=self.pos)

        self._update_pose(dt, activity)

    def _update_pose(self, dt: float, activity: float) -> None:
        rng, n = self.rng, self.n
        # Depth drifts slowly around each fish's favourite distance.
        self.z += 0.15 * (self.z_home - self.z) * dt + 0.1 * math.sqrt(dt) * rng.standard_normal(n)
        np.clip(self.z, 0.0, 1.0, out=self.z)

        # Facing turns through edge-on; hysteresis avoids flicker at low vx.
        vx, vy = self.vel[:, 0], self.vel[:, 1]
        want = np.where(
            np.abs(vx) > 0.15 * self.cruise * activity, np.sign(vx), np.sign(self.facing)
        )
        step = np.clip(want - self.facing, -_TURN_RATE * dt, _TURN_RATE * dt)
        self.facing = np.clip(self.facing + step, -1.0, 1.0)

        speed = np.hypot(vx, vy)
        level = speed > 0.2 * self.cruise
        target_pitch = np.where(level, np.clip(np.arctan2(vy, np.abs(vx)), -0.6, 0.6), 0.0)
        self.pitch += (target_pitch - self.pitch) * (1.0 - math.exp(-dt / _PITCH_TAU))

        beat_hz = 1.0 + 1.6 * speed / self.length
        self.phase = (self.phase + 2.0 * math.pi * beat_hz * dt) % (2.0 * math.pi)

    # ------------------------------------------------------------- read-outs

    @property
    def speed(self) -> F64:
        """Current speed of every fish, px/s."""
        result: F64 = np.hypot(self.vel[:, 0], self.vel[:, 1])
        return result

    @property
    def heading(self) -> F64:
        """Direction of motion, radians (``atan2(vy, vx)``)."""
        result: F64 = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        return result
