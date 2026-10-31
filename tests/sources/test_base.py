"""Tests for the shared source helpers."""

from __future__ import annotations

import numpy as np
import pytest

from puffer.sources.base import RealtimePacer, is_monochrome, iter_frames, to_bgr
from puffer.types import Frame


class _CountingSource:
    source_id = "count"
    fps = 10.0

    def __init__(self, n: int) -> None:
        self.n = n
        self.i = 0
        self.closed = False

    def read(self) -> Frame | None:
        if self.i >= self.n:
            return None
        self.i += 1
        return Frame(self.i - 1, self.i / 10.0, 0.0, np.zeros((2, 2, 3), np.uint8))

    def close(self) -> None:
        self.closed = True


def test_iter_frames_until_exhausted_or_limit() -> None:
    assert [f.index for f in iter_frames(_CountingSource(4))] == [0, 1, 2, 3]
    source = _CountingSource(10)
    assert len(list(iter_frames(source, max_frames=3))) == 3
    assert source.i == 3  # no frame read beyond the limit
    assert not source.closed  # the caller owns the source
    assert list(iter_frames(_CountingSource(5), max_frames=0)) == []
    with pytest.raises(ValueError, match="max_frames"):
        list(iter_frames(_CountingSource(1), max_frames=-1))


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacer_uses_absolute_deadlines() -> None:
    clock = _FakeClock()
    pacer = RealtimePacer(10.0, clock=clock, sleep=clock.sleep)
    assert pacer.wait() == 100.0  # first frame immediately
    clock.now += 0.03  # work took 30 ms
    pacer.wait()
    assert clock.sleeps == [pytest.approx(0.07)]
    clock.now += 0.15  # a slow frame: 50 ms late, no sleep, no drift
    pacer.wait()
    assert len(clock.sleeps) == 1
    pacer.wait()
    assert clock.sleeps[-1] == pytest.approx(0.05)  # back on the original grid
    assert clock.now == pytest.approx(100.3)


def test_pacer_reanchors_after_a_long_stall() -> None:
    clock = _FakeClock()
    pacer = RealtimePacer(10.0, max_lag_s=0.25, clock=clock, sleep=clock.sleep)
    pacer.wait()
    clock.now += 5.0
    pacer.wait()  # re-anchored: returns at once
    pacer.wait()
    assert clock.sleeps == [pytest.approx(0.1)]  # one period, not a burst
    pacer.reset()
    assert pacer.wait() == clock.now
    with pytest.raises(ValueError, match="fps"):
        RealtimePacer(0.0)


def test_monochrome_detection_and_bgr_conversion() -> None:
    gray = np.random.default_rng(1).integers(0, 256, (40, 60), dtype=np.uint8)
    bgr = to_bgr(gray)
    assert bgr.shape == (40, 60, 3)
    assert is_monochrome(bgr)
    assert is_monochrome(gray)
    noisy = bgr.astype(np.int16) + np.random.default_rng(2).integers(-1, 2, bgr.shape)
    assert is_monochrome(np.clip(noisy, 0, 255).astype(np.uint8))  # compression chroma noise
    colour = bgr.copy()
    colour[..., 2] = 255 - colour[..., 2]
    assert not is_monochrome(colour)
    assert to_bgr(gray[..., None]).shape == (40, 60, 3)
    bgra = np.dstack([bgr, np.full((40, 60), 255, np.uint8)])
    assert np.array_equal(to_bgr(bgra), bgr)
    assert to_bgr(bgr) is not None
    assert to_bgr(bgr).flags.c_contiguous
    with pytest.raises(TypeError):
        to_bgr(gray.astype(np.float32))
    with pytest.raises(ValueError, match="unsupported image shape"):
        to_bgr(np.zeros((4, 4, 2), np.uint8))
