"""Factories for hand-made frames and tracking results used by the entropy tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from puffer.types import FishObservation, Frame, TrackingResult, TrackSnapshot

FrameFactory = Callable[..., Frame]
TrackingFactory = Callable[..., TrackingResult]


@pytest.fixture
def make_frame() -> FrameFactory:
    """``make_frame(image, index=0, t=None)``; ``t`` defaults to ``index / 30``."""

    def factory(image: npt.NDArray[np.uint8], index: int = 0, t: float | None = None) -> Frame:
        ts = index / 30.0 if t is None else t
        return Frame(index=index, timestamp=ts, wall_time=1_793_454_000.0 + ts, image=image)

    return factory


def observation(
    track_id: int,
    innovation: tuple[float, float],
    fish_id: str | None = None,
    t: float = 0.0,
) -> FishObservation:
    return FishObservation(
        track_id=track_id,
        fish_id=fish_id,
        t=t,
        cx=100.0 + track_id,
        cy=50.5,
        vx=3.25,
        vy=-1.5,
        speed=3.57,
        heading=-0.43,
        accel=0.75,
        turn_rate=0.125,
        innovation=innovation,
    )


def snapshot(track_id: int, confirmed: bool = True, fish_id: str | None = None) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        fish_id=fish_id,
        bbox=(10.0, 10.0, 30.0, 20.0),
        score=0.9,
        center=(20.0, 15.0),
        velocity=(1.0, 0.0),
        trail=((19.0, 15.0), (20.0, 15.0)),
        confirmed=confirmed,
        age_frames=10,
    )


@pytest.fixture
def make_tracking() -> TrackingFactory:
    """``make_tracking(t, n_confirmed=0, observations=(), frame_index=0)``."""

    def factory(
        t: float,
        n_confirmed: int = 0,
        observations: Sequence[FishObservation] = (),
        frame_index: int = 0,
    ) -> TrackingResult:
        return TrackingResult(
            frame_index=frame_index,
            t=t,
            detections=(),
            tracks=tuple(snapshot(i) for i in range(n_confirmed)),
            observations=tuple(observations),
        )

    return factory


@pytest.fixture
def make_observation() -> Callable[..., FishObservation]:
    """``make_observation(track_id, innovation, fish_id=None, t=0.0)``."""
    return observation
