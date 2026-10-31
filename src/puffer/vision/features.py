"""Motion features per track and long-running "character" statistics per fish.

:class:`MotionFeatures` turns the tracker's Kalman state into the physical
quantities of :class:`puffer.types.FishObservation`: velocity in pixels per
second (the filter works per frame; we scale by the *measured* frame
period), speed, heading, acceleration magnitude, turn rate and the latest
centre innovation. These observations feed the fish entropy channel.

:class:`FishCharacter` aggregates observations per stable fish identity for
the dashboard's fish cards and leaderboard: how fast, how active, how
bursty, how far - and how *chaotic*. The chaos score deliberately reuses
the quantity the entropy channel credits (Kalman innovation, normalised by
body length so a big fish is not "more chaotic" just by being big) plus
turn-rate variability, so the leaderboard ranks fish by roughly how
unpredictable - and therefore how useful - they are as an entropy source.

Heading is undefined for a fish at rest (``atan2`` of filter noise), so
turn rates are only computed, and only enter the statistics, while the fish
moves faster than a small fraction of its body length per second.
"""

from __future__ import annotations

import math
from typing import Protocol

from puffer.types import BBox, FishObservation, Point

_CHAOS_INNOVATION_WEIGHT = 20.0  # 5% of a body length per frame -> 1.0


class MotionTrack(Protocol):
    """What :class:`MotionFeatures` needs from a track (satisfied by ``bytetrack.Track``)."""

    @property
    def track_id(self) -> int: ...

    @property
    def center(self) -> Point: ...

    @property
    def velocity(self) -> Point: ...  # pixels per frame

    @property
    def bbox(self) -> BBox: ...

    @property
    def innovation(self) -> Point | None: ...


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def body_length(bbox: BBox) -> float:
    """Proxy for fish body length in pixels: the longer side of the box."""
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)


class MotionFeatures:
    """Feature extractor for one track; keeps the previous observation for derivatives.

    ``heading_min_speed_bl`` is the speed, in body lengths per second, below
    which the heading is considered undefined and ``turn_rate`` is 0.
    """

    def __init__(self, track_id: int, heading_min_speed_bl: float = 0.1) -> None:
        self.track_id = track_id
        self.heading_min_speed_bl = float(heading_min_speed_bl)
        self._prev: tuple[float, float, float, float, bool] | None = None  # t, vx, vy, hdg, valid

    def observe(
        self, track: MotionTrack, t: float, frame_period: float, fish_id: str | None
    ) -> FishObservation:
        """Compute the observation of ``track`` at time ``t`` (seconds).

        Raises ``ValueError`` if the track has not had a Kalman update yet:
        an innovation of exactly zero would be a fabricated, perfectly
        predictable sample for the entropy channel.
        """
        if track.innovation is None:
            raise ValueError(f"track {track.track_id} has no measurement update yet")
        if frame_period <= 0.0:
            raise ValueError("frame_period must be positive")
        vfx, vfy = track.velocity
        vx, vy = vfx / frame_period, vfy / frame_period
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx)
        heading_valid = speed >= self.heading_min_speed_bl * body_length(track.bbox)
        accel = 0.0
        turn_rate = 0.0
        if self._prev is not None and t > self._prev[0]:
            pt, pvx, pvy, pheading, pvalid = self._prev
            dt = t - pt
            accel = math.hypot(vx - pvx, vy - pvy) / dt
            if heading_valid and pvalid:
                turn_rate = wrap_angle(heading - pheading) / dt
        self._prev = (t, vx, vy, heading, heading_valid)
        cx, cy = track.center
        return FishObservation(
            track_id=track.track_id,
            fish_id=fish_id,
            t=t,
            cx=cx,
            cy=cy,
            vx=vx,
            vy=vy,
            speed=speed,
            heading=heading,
            accel=accel,
            turn_rate=turn_rate,
            innovation=track.innovation,
        )


class _Running:
    """Welford running mean / population variance."""

    __slots__ = ("_m2", "mean", "n")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / self.n) if self.n else 0.0


class FishCharacter:
    """Running behaviour statistics for one fish identity.

    Parameters
    ----------
    fish_id:
        The identity the statistics belong to.
    moving_speed_bl:
        Speed (body lengths per second) above which the fish counts as
        moving: drives ``activity``, ``distance_px`` (Kalman jitter of a
        resting fish is not distance travelled) and which turn rates count.
    max_gap_s:
        Consecutive observations further apart than this (or from different
        tracks) start a new segment: time and distance are not bridged.
    """

    def __init__(self, fish_id: str, moving_speed_bl: float = 0.25, max_gap_s: float = 1.0) -> None:
        self.fish_id = fish_id
        self.moving_speed_bl = float(moving_speed_bl)
        self.max_gap_s = float(max_gap_s)
        self.n_obs = 0
        self.seen_s = 0.0
        self.distance_px = 0.0
        self.speed_max = 0.0
        self._moving_s = 0.0
        self._speed = _Running()
        self._innovation = _Running()
        self._turn = _Running()
        self._last: tuple[int, float, float, float] | None = None  # track_id, t, cx, cy

    def update(self, obs: FishObservation, body_length_px: float) -> None:
        """Fold one observation in; ``body_length_px`` normalises speeds and innovations."""
        body = max(float(body_length_px), 1.0)
        moving = obs.speed >= self.moving_speed_bl * body
        self.n_obs += 1
        self._speed.add(obs.speed)
        self.speed_max = max(self.speed_max, obs.speed)
        self._innovation.add(math.hypot(*obs.innovation) / body)
        if moving:
            self._turn.add(obs.turn_rate)
        if self._last is not None:
            track_id, t, cx, cy = self._last
            dt = obs.t - t
            if track_id == obs.track_id and 0.0 < dt <= self.max_gap_s:
                self.seen_s += dt
                if moving:
                    self._moving_s += dt
                    self.distance_px += math.hypot(obs.cx - cx, obs.cy - cy)
        self._last = (obs.track_id, obs.t, obs.cx, obs.cy)

    @property
    def speed_avg(self) -> float:
        """Mean speed in pixels per second."""
        return self._speed.mean

    @property
    def activity(self) -> float:
        """Fraction of the observed time spent moving, in ``[0, 1]``."""
        return self._moving_s / self.seen_s if self.seen_s > 0.0 else 0.0

    @property
    def burstiness(self) -> float:
        """Goh-Barabasi burstiness of the speed series, ``(sigma - mu) / (sigma + mu)``.

        -1 for perfectly steady cruising, 0 for Poisson-like variation,
        towards +1 for long rests punctuated by sprints.
        """
        mu, sigma = self._speed.mean, self._speed.std
        return (sigma - mu) / (sigma + mu) if sigma + mu > 0.0 else 0.0

    @property
    def innovation_norm(self) -> float:
        """Mean centre innovation in body lengths per frame."""
        return self._innovation.mean

    @property
    def turn_rate_std(self) -> float:
        """Standard deviation of the turn rate while moving (rad/s)."""
        return self._turn.std

    @property
    def chaos_score(self) -> float:
        """Dimensionless unpredictability: ``20 * innovation_norm + turn_rate_std / pi``.

        0 for a perfectly predictable swimmer; a calm cruiser scores well
        below 1, an erratic darting fish several.
        """
        return _CHAOS_INNOVATION_WEIGHT * self.innovation_norm + self.turn_rate_std / math.pi

    def to_dict(self) -> dict[str, str | int | float]:
        """JSON-ready statistics (feeds ``FishCard.stats`` and the leaderboard)."""
        return {
            "fish_id": self.fish_id,
            "n_obs": self.n_obs,
            "seen_s": self.seen_s,
            "distance_px": self.distance_px,
            "speed_avg": self.speed_avg,
            "speed_max": self.speed_max,
            "activity": self.activity,
            "burstiness": self.burstiness,
            "innovation_norm": self.innovation_norm,
            "turn_rate_std": self.turn_rate_std,
            "chaos_score": self.chaos_score,
        }
