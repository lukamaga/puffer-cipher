"""Full-resolution benchmark: the default 640x480 simulator with liveness and tracking.

Marked ``slow``: it renders and tracks ~15 s of tank at the production
resolution. It doubles as the full-size end-to-end check (LED verified,
sensor credited) and prints the measured throughput.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from puffer.config import PufferConfig
from puffer.runtime import build_engine
from puffer.types import CHANNEL_SENSOR, HealthState, LivenessState

from .helpers import seeded_schedule

FRAMES = 600  # 20 s of tank: verified at 12.9 s, sensor credited after the 5 s cooldown


@pytest.mark.slow
def test_default_resolution_throughput(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = PufferConfig(source="synthetic:seed=1", data_dir=tmp_path)  # 640x480, 6 fish
    engine = build_engine(cfg, schedule=seeded_schedule(51))
    try:
        start = time.perf_counter()
        snap = None
        for _ in range(FRAMES):
            snap = engine.step()
        elapsed = time.perf_counter() - start
        assert snap is not None
        assert (snap.width, snap.height) == (640, 480)
        assert snap.liveness.state is LivenessState.VERIFIED
        assert snap.channels[CHANNEL_SENSOR].health is HealthState.OK
        fps = FRAMES / elapsed
        with capsys.disabled():
            print(
                f"\nengine throughput 640x480 (sim render + liveness + tracking + entropy): "
                f"{fps:.1f} frames/s (processing_fps EMA {snap.processing_fps:.1f})"
            )
        assert fps > 2.0  # a loose floor; real numbers are reported, not asserted
    finally:
        engine.close()
