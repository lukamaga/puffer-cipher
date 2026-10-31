"""Session-wide simulator runs shared by the engine integration tests.

The expensive part of an end-to-end test is simulating and tracking enough
frames for the LED challenge to verify (one 64-bit window = 12.8 s of tank
time), the 5 s health cooldown to pass and the fish channel to finish its
1024-sample start-up. The day and night runs are therefore made once per
session and inspected by several tests. Everything is seeded (simulator,
LED challenge), so the runs are identical on every machine.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from puffer.runtime import build_engine

from .helpers import Run, config, run_frames, seeded_schedule, sim_spec

DAY_FRAMES = 900  # 30 s: verified 12.9 s, sensor OK 17.9 s, fish OK 23.2 s, then credit
NIGHT_FRAMES = 1080  # 36 s: fewer, sleepier fish fill the start-up window later (28.8 s)


@pytest.fixture(scope="session")
def day_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """30 s of a seeded day tank with liveness required (the default policy)."""
    cfg = config(tmp_path_factory.mktemp("day"), source=sim_spec(seed=1, mode="day"))
    engine = build_engine(cfg, schedule=seeded_schedule(11))
    try:
        yield run_frames(engine, DAY_FRAMES)
    finally:
        engine.close()


@pytest.fixture(scope="session")
def night_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """36 s of the same tank at night (IR camera, sleeping fish), liveness required."""
    cfg = config(tmp_path_factory.mktemp("night"), source=sim_spec(seed=1, mode="night"))
    engine = build_engine(cfg, schedule=seeded_schedule(12))
    try:
        yield run_frames(engine, NIGHT_FRAMES)
    finally:
        engine.close()
