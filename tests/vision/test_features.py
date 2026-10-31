"""Motion features and fish character statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from puffer.types import BBox, FishObservation, Point
from puffer.vision.features import FishCharacter, MotionFeatures, body_length, wrap_angle

PERIOD = 1.0 / 30.0


@dataclass
class FakeTrack:
    velocity: Point  # px / frame
    center: Point = (100.0, 100.0)
    bbox: BBox = (85.0, 94.0, 115.0, 106.0)  # 30 px long
    innovation: Point | None = (0.5, -0.25)
    track_id: int = 1


def obs(
    t: float,
    cx: float,
    speed: float,
    *,
    track_id: int = 1,
    innovation: Point = (0.0, 0.0),
    turn_rate: float = 0.0,
) -> FishObservation:
    return FishObservation(
        track_id=track_id,
        fish_id="f1",
        t=t,
        cx=cx,
        cy=50.0,
        vx=speed,
        vy=0.0,
        speed=speed,
        heading=0.0,
        accel=0.0,
        turn_rate=turn_rate,
        innovation=innovation,
    )


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (math.pi / 2, math.pi / 2), (3 * math.pi / 2, -math.pi / 2), (-3.0, -3.0)],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected)
    assert -math.pi <= wrap_angle(angle + 20 * math.pi) < math.pi


def test_body_length() -> None:
    assert body_length((0.0, 0.0, 30.0, 12.0)) == 30.0
    assert body_length((0.0, 0.0, 0.5, 0.5)) == 1.0


def test_velocity_scaled_by_measured_frame_period() -> None:
    mf = MotionFeatures(1)
    o = mf.observe(FakeTrack(velocity=(2.0, -1.0)), 0.0, 1 / 25, "f7")
    assert (o.vx, o.vy) == pytest.approx((50.0, -25.0))
    assert o.speed == pytest.approx(math.hypot(50.0, 25.0))
    assert o.heading == pytest.approx(math.atan2(-25.0, 50.0))
    assert o.accel == 0.0  # no previous observation
    assert o.turn_rate == 0.0
    assert o.innovation == (0.5, -0.25)
    assert o.fish_id == "f7"
    assert (o.cx, o.cy) == (100.0, 100.0)


def test_acceleration_and_turn_rate() -> None:
    mf = MotionFeatures(1)
    mf.observe(FakeTrack(velocity=(1.0, 0.0)), 0.0, PERIOD, None)  # 30 px/s east
    o = mf.observe(FakeTrack(velocity=(0.0, 1.0)), 0.1, PERIOD, None)  # 30 px/s south
    assert o.accel == pytest.approx(math.hypot(30.0, 30.0) / 0.1)
    assert o.turn_rate == pytest.approx((math.pi / 2) / 0.1)


def test_turn_rate_wraps_across_pi() -> None:
    mf = MotionFeatures(1)
    a, b = math.radians(175), math.radians(-175)
    mf.observe(FakeTrack(velocity=(math.cos(a), math.sin(a))), 0.0, PERIOD, None)
    o = mf.observe(FakeTrack(velocity=(math.cos(b), math.sin(b))), 0.5, PERIOD, None)
    assert o.turn_rate == pytest.approx(math.radians(10) / 0.5)


def test_turn_rate_suppressed_at_rest() -> None:
    mf = MotionFeatures(1, heading_min_speed_bl=0.1)  # 3 px/s for a 30 px fish
    mf.observe(FakeTrack(velocity=(0.05, 0.0)), 0.0, PERIOD, None)  # 1.5 px/s
    o = mf.observe(FakeTrack(velocity=(-0.05, 0.0)), 0.1, PERIOD, None)
    assert o.turn_rate == 0.0
    assert o.accel == pytest.approx(3.0 / 0.1)


def test_observation_requires_measurement_update() -> None:
    mf = MotionFeatures(1)
    with pytest.raises(ValueError, match="no measurement"):
        mf.observe(FakeTrack(velocity=(1.0, 0.0), innovation=None), 0.0, PERIOD, None)
    with pytest.raises(ValueError, match="frame_period"):
        mf.observe(FakeTrack(velocity=(1.0, 0.0)), 0.0, 0.0, None)


def test_character_steady_cruiser() -> None:
    ch = FishCharacter("f1")
    for i in range(61):
        t = i * PERIOD
        ch.update(obs(t, 30.0 * t, 30.0), body_length_px=30.0)
    assert ch.n_obs == 61
    assert ch.seen_s == pytest.approx(2.0)
    assert ch.distance_px == pytest.approx(60.0)
    assert ch.speed_avg == pytest.approx(30.0)
    assert ch.speed_max == pytest.approx(30.0)
    assert ch.activity == pytest.approx(1.0)
    assert ch.burstiness == pytest.approx(-1.0)  # perfectly regular
    assert ch.chaos_score == pytest.approx(0.0)


def test_character_resting_fish() -> None:
    ch = FishCharacter("f1", moving_speed_bl=0.25)
    for i in range(30):
        # filter jitter: tiny speeds and centre wobble are not distance travelled
        ch.update(obs(i * PERIOD, 100.0 + 0.2 * (-1) ** i, 1.0), body_length_px=30.0)
    assert ch.activity == 0.0
    assert ch.distance_px == 0.0
    assert ch.seen_s == pytest.approx(29 * PERIOD)


def test_character_bursty_and_chaotic() -> None:
    calm, wild = FishCharacter("calm"), FishCharacter("wild")
    for i in range(120):
        t = i * PERIOD
        calm.update(obs(t, 20.0 * t, 20.0, innovation=(0.1, 0.0)), 30.0)
        sprint = i % 30 < 4  # four frames of sprinting, alternating turns, per second
        wild.update(
            obs(
                t,
                20.0 * t,
                200.0 if sprint else 0.5,
                innovation=(2.0, -1.5),
                turn_rate=6.0 if i % 2 else -6.0,
            ),
            30.0,
        )
    assert wild.burstiness > 0.3 > calm.burstiness
    assert wild.chaos_score > 5 * calm.chaos_score
    assert wild.activity < 0.2
    assert wild.speed_max == pytest.approx(200.0)
    assert wild.turn_rate_std == pytest.approx(6.0)
    assert wild.innovation_norm == pytest.approx(2.5 / 30.0)


def test_character_does_not_bridge_gaps_or_track_changes() -> None:
    ch = FishCharacter("f1", max_gap_s=1.0)
    ch.update(obs(0.0, 0.0, 30.0), 30.0)
    ch.update(obs(0.5, 15.0, 30.0), 30.0)
    ch.update(obs(5.0, 200.0, 30.0), 30.0)  # 4.5 s gap
    ch.update(obs(5.5, 215.0, 30.0, track_id=2), 30.0)  # same fish, new track
    ch.update(obs(6.0, 230.0, 30.0, track_id=2), 30.0)
    assert ch.seen_s == pytest.approx(1.0)
    assert ch.distance_px == pytest.approx(30.0)


def test_character_to_dict() -> None:
    ch = FishCharacter("f3")
    ch.update(obs(0.0, 0.0, 10.0, innovation=(3.0, 4.0)), 50.0)
    d = ch.to_dict()
    assert d["fish_id"] == "f3"
    for key in ("speed_avg", "speed_max", "activity", "chaos_score", "distance_px", "seen_s"):
        assert key in d  # FishCard.stats fields owned by vision
    assert d["innovation_norm"] == pytest.approx(0.1)
    assert d["chaos_score"] == pytest.approx(2.0)
    empty = FishCharacter("f9").to_dict()
    assert empty["activity"] == 0.0
    assert empty["burstiness"] == 0.0
