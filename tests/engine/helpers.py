"""Shared helpers for the engine tests: seeded specs, schedules, fake sources, runners."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from puffer.config import PufferConfig
from puffer.engine import EngineSnapshot, EntropyEngine
from puffer.liveness import ChallengeSchedule
from puffer.types import Frame, HealthState, LivenessState

WIDTH, HEIGHT = 320, 240
"""Test resolution: 4x fewer pixels than the default, same tank (sizes scale with width)."""

FPS = 30.0


def sim_spec(seed: int = 1, *, mode: str = "day", width: int = WIDTH, **options: object) -> str:
    """A seeded simulator spec with the LED inside a ``width``-wide frame."""
    height = width * 3 // 4
    parts = [f"seed={seed}", f"width={width}", f"height={height}", f"led={width - 32}:32"]
    parts.append(f"mode={mode}")
    parts.extend(f"{key}={value}" for key, value in options.items())
    return "synthetic:" + ",".join(parts)


def seeded_schedule(seed: int, symbol_s: float = 0.2) -> ChallengeSchedule:
    """A reproducible LED challenge (production schedules draw from ``os.urandom``)."""
    generator = np.random.default_rng(seed)
    return ChallengeSchedule(symbol_s=symbol_s, random_bytes=lambda n: generator.bytes(n))


def config(data_dir: Path, **overrides: object) -> PufferConfig:
    """Test config: seeded day simulator, ``data_dir`` under ``tmp_path``."""
    values: dict[str, object] = {"source": sim_spec(), "data_dir": data_dir}
    values.update(overrides)
    return PufferConfig(**values)  # type: ignore[arg-type]


@dataclass
class FrameRecord:
    """The few numbers of one snapshot a test needs later (snapshots hold whole reports)."""

    index: int
    t: float
    liveness: LivenessState
    states: dict[str, HealthState]
    credited: dict[str, float]
    bits_per_s: dict[str, float]
    reasons: dict[str, tuple[str, ...]]
    checks: dict[str, dict[str, str]]


def record_of(snap: EngineSnapshot) -> FrameRecord:
    return FrameRecord(
        index=snap.frame_index,
        t=snap.t,
        liveness=snap.liveness.state,
        states={name: c.health for name, c in snap.channels.items()},
        credited={name: c.credited_total for name, c in snap.channels.items()},
        bits_per_s={name: c.bits_per_s for name, c in snap.channels.items()},
        reasons={name: r.reasons for name, r in snap.health.items()},
        checks={name: dict(r.checks) for name, r in snap.health.items()},
    )


@dataclass
class Run:
    """An engine and the per-frame records of a synchronous run."""

    engine: EntropyEngine
    records: list[FrameRecord] = field(default_factory=list)

    @property
    def last(self) -> FrameRecord:
        return self.records[-1]

    def first(self, predicate: Callable[[FrameRecord], bool]) -> FrameRecord | None:
        return next((r for r in self.records if predicate(r)), None)


def run_frames(
    engine: EntropyEngine, n: int, *, until: Callable[[FrameRecord], bool] | None = None
) -> Run:
    """Step ``engine`` up to ``n`` frames (or until the source ends / ``until`` holds)."""
    run = Run(engine)
    for _ in range(n):
        snap = engine.step()
        if snap is None:
            break
        run.records.append(record_of(snap))
        if until is not None and until(run.records[-1]):
            break
    return run


class FrozenSource:
    """A source that shows one image forever: a stuck driver, or a photo held up to the lens."""

    source_id = "frozen"
    fps = FPS

    def __init__(self, image: np.ndarray, *, frames: int | None = None) -> None:
        self.image = image
        self.frames = frames
        self.index = 0
        self.closed = False

    def read(self) -> Frame | None:
        if self.closed or (self.frames is not None and self.index >= self.frames):
            return None
        i = self.index
        self.index += 1
        return Frame(i, i / FPS, 1_793_454_000.0 + i / FPS, self.image, source_id=self.source_id)

    def close(self) -> None:
        self.closed = True


class FreezeWindow:
    """Wrap a source and repeat one frame's picture for frames ``[start, stop)``.

    Indices and timestamps keep advancing, exactly like a driver that keeps
    delivering its last buffer during a transient hiccup.
    """

    def __init__(self, inner: object, start: int, stop: int) -> None:
        self.inner = inner
        self.start, self.stop = start, stop
        self.source_id = "freeze-window"
        self.fps = FPS
        self._held: np.ndarray | None = None

    def read(self) -> Frame | None:
        frame = self.inner.read()  # type: ignore[attr-defined]
        if frame is None:
            return None
        if self.start <= frame.index < self.stop:
            if self._held is None:
                self._held = frame.image
            return replace(frame, image=self._held)
        return frame

    def close(self) -> None:
        self.inner.close()  # type: ignore[attr-defined]


class FakeMonotonic:
    """A monotonic clock the test advances by hand (thread-safe)."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    def advance(self, dt: float) -> None:
        with self._lock:
            self._t += dt


class ScriptedSource:
    """Frames from an image generator, with scripted failures and stalls.

    ``fail(i)`` decides whether read number ``i`` raises; ``stall`` (an Event)
    blocks reads while it is clear, like a camera stuck inside its driver.
    """

    source_id = "scripted"
    fps = FPS

    def __init__(
        self,
        *,
        seed: int = 0,
        fail: Callable[[int], bool] = lambda i: False,
        stall_after: int | None = None,
        clock: FakeMonotonic | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._base = self._rng.integers(30, 220, size=(HEIGHT, WIDTH, 3)).astype(np.float64)
        self.fail = fail
        self.stall_after = stall_after
        self.unstall = threading.Event()
        self.stalled = threading.Event()
        self.clock = clock
        self.reads = 0
        self.frames = 0
        self.closed = False

    def read(self) -> Frame | None:
        i = self.reads
        self.reads += 1
        if self.stall_after is not None and self.frames >= self.stall_after:
            self.stalled.set()
            self.unstall.wait()
            if self.closed:
                return None
        if self.fail(i):
            raise OSError(f"scripted read failure {i}")
        noise = self._rng.normal(0.0, 3.0, self._base.shape)
        image = np.clip(self._base + noise, 0, 255).astype(np.uint8)
        k = self.frames
        self.frames += 1
        if self.clock is not None:
            self.clock.advance(1.0 / FPS)  # a live camera: frames stamped with the clock
        t = self.clock() if self.clock is not None else k / FPS
        return Frame(k, t, 1_793_454_000.0 + k / FPS, image, source_id=self.source_id)

    def close(self) -> None:
        self.closed = True
        self.unstall.set()


def wait_for(predicate: Callable[[], bool], timeout: float = 10.0, step: float = 0.01) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` (wall seconds) passes."""
    event = threading.Event()
    deadline = timeout / step
    for _ in range(int(deadline)):
        if predicate():
            return True
        event.wait(step)
    return predicate()


def states_sequence(records: Iterator[FrameRecord] | list[FrameRecord], channel: str) -> list[str]:
    """Run-length compressed sequence of a channel's health states."""
    out: list[str] = []
    for record in records:
        state = record.states[channel].value
        if not out or out[-1] != state:
            out.append(state)
    return out
