"""Deterministic stand-ins for the engine, configuration and fish statistics.

The fake engine implements the DESIGN.md 9.2 surface the CLI uses
(:class:`puffer.cli.protocols.EngineLike`) on top of the *real*
:class:`~puffer.entropy.EntropyPool` and :class:`~puffer.entropy.PufferRNG`,
so receipts, fish attribution and key material come from production code;
only frames, tracking and health are scripted. Each frame credits
``sensor_bits`` to the sensor channel and ``fish_bits`` split evenly between
the fish - unless liveness is failing, in which case nothing is absorbed,
exactly like a FAILED channel in the real engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from puffer.entropy import EntropyPool, PufferRNG, ReceiptContext
from puffer.types import (
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    ChannelOutput,
    Frame,
    HealthReport,
    HealthState,
    LightMode,
    LivenessState,
    LivenessStatus,
    TrackSnapshot,
)

FISH: tuple[tuple[str, str, str], ...] = (
    ("f1", "Biscuit", "#ff7014"),
    ("f2", "Captain Fin", "#1965e6"),
    ("f3", "Professor Puff", "#c4b260"),
)


@dataclass
class FakeConfig:
    """The ``PufferConfig`` fields the CLI reads (a dataclass, so overrides work)."""

    source: str = "synthetic:seed=7"
    liveness: str = "auto"
    data_dir: Path = Path("data")
    pool_capacity_bits: float = 4096.0
    host: str = "127.0.0.1"
    port: int = 8000
    beacon_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "liveness": self.liveness,
            "data_dir": str(self.data_dir),
            "pool_capacity_bits": self.pool_capacity_bits,
            "host": self.host,
            "port": self.port,
            "beacon_enabled": self.beacon_enabled,
        }


@dataclass(frozen=True)
class FakeChannelStats:
    name: str
    bits_per_s: float
    h_per_sample: float | None
    credited_total: float
    health: HealthState


@dataclass(frozen=True)
class FakeSnapshot:
    frame_index: int
    t: float
    fps: float
    mode: LightMode
    tracks: tuple[TrackSnapshot, ...]
    channels: dict[str, FakeChannelStats]
    pool: dict[str, float]
    health: dict[str, HealthReport]
    overall: HealthState
    liveness: LivenessStatus
    liveness_policy: str = "required"


class RecordingRng:
    """Wraps a real :class:`PufferRNG` and records every call.

    ``with_salt=False`` hides ``salt`` to exercise the CLI's fallback for
    RNGs that predate public-value helpers.
    """

    def __init__(self, rng: PufferRNG, *, with_salt: bool = True) -> None:
        self._rng = rng
        self.calls: list[tuple[str, int, str]] = []
        if with_salt:
            self.salt = self._salt

    def generate(self, n: int, purpose: str = "bytes") -> Any:
        self.calls.append(("generate", n, purpose))
        return self._rng.generate(n, purpose)

    def _salt(self, n: int = 16) -> bytes:
        self.calls.append(("salt", n, "salt"))
        return self._rng.salt(n)


def _track(i: int, fish_id: str | None, index: int) -> TrackSnapshot:
    x = 50.0 + 40 * i + (index % 7)
    return TrackSnapshot(
        track_id=i + 1,
        fish_id=fish_id,
        bbox=(x, 60.0, x + 30, 80.0),
        score=0.9,
        center=(x + 15, 70.0),
        velocity=(3.0, 0.5),
        trail=((x, 70.0), (x + 15, 70.0)),
        confirmed=True,
        age_frames=index + 1,
    )


class FakeEngine:
    """A scripted engine; see the module docstring.

    Parameters mirror situations the CLI must handle: ``liveness`` (a state
    other than VERIFIED/DISABLED blocks all credit), ``frames`` (a finite
    source that ends), ``unidentified`` (a track without a re-ID identity),
    ``interrupt_after`` (Ctrl-C during collection), ``frame_delay_s`` (keeps
    timeout tests from spinning).
    """

    def __init__(
        self,
        config: Any,
        *,
        sensor_bits: float = 100.0,
        fish_bits: float = 12.0,
        liveness: LivenessState = LivenessState.VERIFIED,
        frames: int | None = None,
        source_id: str = "synthetic",
        unidentified: bool = False,
        with_salt: bool = True,
        interrupt_after: int | None = None,
        frame_delay_s: float = 0.0,
        fish: tuple[tuple[str, str, str], ...] = FISH,
    ) -> None:
        self.config = config
        self.sensor_bits = sensor_bits
        self.fish_bits = fish_bits
        self.liveness_state = liveness
        self.frames = frames
        self.source_id = source_id
        self.unidentified = unidentified
        self.interrupt_after = interrupt_after
        self.frame_delay_s = frame_delay_s
        self.fish = fish
        self.pool = EntropyPool(config.pool_capacity_bits)
        self.rng = RecordingRng(PufferRNG(self.pool, context=self._context), with_salt=with_salt)
        self.index = -1
        self.started = 0
        self.stopped = 0
        self.closed = False
        self.collect_calls: list[tuple[float, float, int | None]] = []
        self._snapshot: FakeSnapshot | None = None
        self._frame: Frame | None = None

    # -- DESIGN 9.2 surface ---------------------------------------------------

    def collect(
        self, min_credit_bits: float, *, timeout_s: float, max_frames: int | None = None
    ) -> bool:
        """Same contract as ``EntropyEngine.collect`` (synchronous mode)."""
        self.collect_calls.append((min_credit_bits, timeout_s, max_frames))
        deadline = time.monotonic() + timeout_s
        done = 0
        while self.pool.camera_credit() < min_credit_bits:
            if (max_frames is not None and done >= max_frames) or time.monotonic() >= deadline:
                return False
            if self._step() is None:
                return False
            done += 1
        return True

    def snapshot(self) -> FakeSnapshot | None:
        return self._snapshot

    def latest_frame(self) -> Frame | None:
        return self._frame

    def fish_cards(self) -> list[dict[str, Any]]:
        return [
            {"fish_id": fid, "name": name, "color": color, "present": True}
            for fid, name, color in self.fish
        ]

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed = True

    # -- internals ----------------------------------------------------------------

    def _context(self) -> ReceiptContext:
        healthy = self._healthy()
        state = HealthState.OK if healthy else HealthState.FAILED
        return ReceiptContext(
            mode="day",
            liveness=self._liveness(),
            health={CHANNEL_SENSOR: state, CHANNEL_FISH: state},
            fish_names={fid: name for fid, name, _ in self.fish},
        )

    def _healthy(self) -> bool:
        return self.liveness_state in (LivenessState.VERIFIED, LivenessState.DISABLED)

    def _liveness(self) -> LivenessStatus:
        verified = self.liveness_state is LivenessState.VERIFIED
        return LivenessStatus(
            self.liveness_state,
            match_ratio=1.0 if verified else 0.5,
            lag_s=0.05,
            confidence_bits=31.5 if verified else 2.0,
        )

    def _step(self) -> FakeSnapshot | None:
        if self.frames is not None and self.index + 1 >= self.frames:
            return None
        if self.interrupt_after is not None and self.index + 1 >= self.interrupt_after:
            raise KeyboardInterrupt
        if self.frame_delay_s:
            time.sleep(self.frame_delay_s)
        self.index += 1
        i = self.index
        t = i / 30.0
        owners: list[str | None] = [fid for fid, _, _ in self.fish]
        if self.unidentified:
            owners.append("track-99")
        if self._healthy():
            self.pool.absorb(
                ChannelOutput(CHANNEL_SENSOR, i, t, b"s%d" % i, np.zeros(16, np.uint8)),
                self.sensor_bits,
            )
            self.pool.absorb(
                ChannelOutput(
                    CHANNEL_FISH,
                    i,
                    t,
                    b"f%d" % i,
                    np.zeros(len(owners), np.uint8),
                    tuple(owners),
                ),
                self.fish_bits,
            )
        state = HealthState.OK if self._healthy() else HealthState.FAILED
        reasons = () if self._healthy() else ("liveness failed",)
        tracks = tuple(_track(n, fid, i) for n, (fid, _, _) in enumerate(self.fish))
        if self.unidentified:
            tracks += (_track(len(self.fish), None, i),)
        credits = self.pool.credits()
        self._frame = Frame(
            index=i,
            timestamp=t,
            wall_time=1_793_454_000.0 + t,
            image=np.zeros((4, 4, 3), np.uint8),
            source_id=self.source_id,
        )
        self._snapshot = FakeSnapshot(
            frame_index=i,
            t=t,
            fps=30.0,
            mode=LightMode.DAY,
            tracks=tracks,
            channels={
                name: FakeChannelStats(name, 0.0, 2.0, 0.0, state)
                for name in (CHANNEL_SENSOR, CHANNEL_FISH)
            }
            | {CHANNEL_SYSTEM: FakeChannelStats(CHANNEL_SYSTEM, 0.0, None, 0.0, HealthState.OK)},
            pool={
                CHANNEL_SENSOR: credits.get(CHANNEL_SENSOR, 0.0),
                CHANNEL_FISH: credits.get(CHANNEL_FISH, 0.0),
            },
            health={
                name: HealthReport(name, state, reasons) for name in (CHANNEL_SENSOR, CHANNEL_FISH)
            },
            overall=state,
            liveness=self._liveness(),
        )
        return self._snapshot


@dataclass
class FakeFishStats:
    rows: list[dict[str, Any]] = field(default_factory=list)
    days: list[int] = field(default_factory=list)
    closed: bool = False

    def leaderboard(self, days: int = 7) -> list[dict[str, Any]]:
        self.days.append(days)
        return [dict(row) for row in self.rows]

    def close(self) -> None:
        self.closed = True
