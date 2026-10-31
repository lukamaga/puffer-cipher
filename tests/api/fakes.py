"""Deterministic stand-ins for the engine (DESIGN.md 9.2) as the API sees it.

The RNG is *real*: a ``PufferRNG`` over a real ``EntropyPool``, with seeded
"system randomness" so outputs are reproducible. That way receipts, key ids
and AEAD round trips in the API tests exercise the production code paths;
only the camera, vision and threads are faked.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from puffer.entropy import EntropyPool, PufferRNG, ReceiptContext
from puffer.types import (
    ChannelOutput,
    Frame,
    HealthReport,
    HealthState,
    LightMode,
    LivenessState,
    LivenessStatus,
    TrackSnapshot,
)

T0_S = 1_793_454_000.0
FAKE_JPEG = b"\xff\xd8\xff\xe0 not really a jpeg \xff\xd9"


@dataclass(frozen=True)
class FakeConfig:
    source: str = "synthetic:seed=7"
    pool_capacity_bits: float = 4096.0
    stream_fps: float = 10.0
    stream_width: int = 640
    jpeg_quality: int = 70


@dataclass(frozen=True)
class FakeChannelStats:
    name: str
    bits_per_s: float
    h_per_sample: float
    credited_total: float
    health: HealthState


@dataclass(frozen=True)
class FakeSnapshot:
    frame_index: int
    t: float
    wall_time: float
    fps: float
    mode: LightMode
    width: int
    height: int
    tracks: tuple[TrackSnapshot, ...]
    channels: dict[str, FakeChannelStats]
    pool: dict[str, float]
    health: dict[str, HealthReport]
    overall: HealthState
    liveness: LivenessStatus


def default_tracks() -> tuple[TrackSnapshot, ...]:
    """An identified fish, an unidentified one and a tentative (unconfirmed) track."""
    return (
        TrackSnapshot(
            track_id=3,
            fish_id="f1",
            bbox=(100.04, 50.0, 160.0, 90.0),
            score=0.9,
            center=(130.0, 70.0),
            velocity=(30.0, 40.0),
            trail=((120.0, 60.0), (125.0, 65.0), (130.0, 70.0)),
            confirmed=True,
            age_frames=40,
        ),
        TrackSnapshot(
            track_id=7,
            fish_id=None,
            bbox=(300.0, 200.0, 340.0, 230.0),
            score=0.7,
            center=(320.0, 215.0),
            velocity=(-5.0, 0.0),
            trail=((322.0, 215.0),),
            confirmed=True,
            age_frames=8,
        ),
        TrackSnapshot(
            track_id=9,
            fish_id=None,
            bbox=(10.0, 10.0, 20.0, 20.0),
            score=0.3,
            center=(15.0, 15.0),
            velocity=(0.0, 0.0),
            trail=(),
            confirmed=False,
            age_frames=1,
        ),
    )


def default_cards() -> list[dict[str, Any]]:
    return [
        {
            "fish_id": "f1",
            "name": "Nemo",
            "color": "#ff8800",
            "present": True,
            "track_id": 3,
            "stats": {
                "speed_avg": 41.5,
                "speed_max": 120.0,
                "activity": 0.62,
                "chaos_score": 0.71,
                "bits_total": 5120.5,
                "bits_today": 812.25,
                "distance_px": 15000.0,
                "seen_s": 3600.0,
            },
        },
        {
            "fish_id": "f2",
            "name": "Dory",
            "color": "#1f77b4",
            "present": False,
            "track_id": None,
            "stats": {
                "speed_avg": 12.0,
                "speed_max": 40.0,
                "activity": 0.2,
                "chaos_score": 0.33,
                "bits_total": 900.0,
                "bits_today": 0.0,
                "distance_px": 1200.0,
                "seen_s": 60.0,
            },
        },
    ]


def default_health() -> dict[str, HealthReport]:
    return {
        "sensor": HealthReport("sensor", HealthState.OK, (), {"rct": "ok", "apt": "ok"}),
        "fish": HealthReport(
            "fish",
            HealthState.DEGRADED,
            ("fewer fish than expected",),
            {"rct": "ok", "fish_presence": "degraded"},
        ),
        "system": HealthReport("system", HealthState.OK),
    }


def default_liveness() -> LivenessStatus:
    # One undecodable bit ("?") under a commanded 1, as the verifier reports it.
    return LivenessStatus(
        state=LivenessState.VERIFIED,
        match_ratio=0.953,
        lag_s=0.12,
        confidence_bits=41.7,
        expected="01101",
        observed="01?01",
        window_start=10.0,
        window_end=22.8,
    )


class SeededBytes:
    """Reproducible stand-in for ``os.urandom`` (tests only, never secrets)."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self._lock = threading.Lock()

    def __call__(self, n: int) -> bytes:
        with self._lock:
            return self._random.randbytes(n)


class RecordingRng:
    """Wraps a real PufferRNG and records which methods the API called."""

    def __init__(self, inner: PufferRNG) -> None:
        self.inner = inner
        self.calls: list[tuple[str, int, str]] = []
        self._lock = threading.Lock()

    def _record(self, method: str, n: int, purpose: str) -> None:
        with self._lock:
            self.calls.append((method, n, purpose))

    def generate(self, n: int, purpose: str = "bytes") -> tuple[bytes, Any]:
        self._record("generate", n, purpose)
        return self.inner.generate(n, purpose)

    def generate_key(self) -> tuple[bytes, Any]:
        self._record("generate_key", 32, "key")
        return self.inner.generate_key()

    def nonce(self) -> bytes:
        self._record("nonce", 24, "nonce")
        return self.inner.nonce()

    def display_bits(self, n_bits: int) -> str:
        return self.inner.display_bits(n_bits)

    def methods(self) -> list[str]:
        with self._lock:
            return [c[0] for c in self.calls]


@dataclass
class FakeEngine:
    """Snapshots on demand; ``auto_advance`` makes every ``snapshot()`` a new frame.

    Auto-advancing keeps the live stream flowing without a background
    thread: the hub only sends a frame when the index changed.
    """

    ready: bool = True
    auto_advance: bool = True
    width: int = 640
    height: int = 360
    tracks: tuple[TrackSnapshot, ...] = field(default_factory=default_tracks)
    cards: list[dict[str, Any]] = field(default_factory=default_cards)
    health: dict[str, HealthReport] = field(default_factory=default_health)
    liveness: LivenessStatus = field(default_factory=default_liveness)
    jpeg_bytes: bytes | None = FAKE_JPEG
    source_id: str | None = "synthetic"
    system_h: float = math.nan
    seed: int = 7
    config: FakeConfig = field(default_factory=FakeConfig)

    def __post_init__(self) -> None:
        self.pool = EntropyPool(capacity_bits=self.config.pool_capacity_bits)
        self.rng = RecordingRng(
            PufferRNG(
                self.pool,
                system_random=SeededBytes(self.seed),
                clock=lambda: T0_S,
                context=self.receipt_context,
            )
        )
        self._lock = threading.Lock()
        self._index = 0
        self._subscribers: list[Callable[[Any], None]] = []
        self.jpeg_calls = 0
        self.jpeg_args: list[tuple[int, int]] = []
        self.fish_calls = 0

    # -- test controls ------------------------------------------------------

    def advance(self, frames: int = 1) -> FakeSnapshot:
        """Produce a new frame and notify subscribers (like the engine thread would)."""
        with self._lock:
            self.ready = True
            self._index += frames
            snapshot = self._build(self._index)
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(snapshot)
        return snapshot

    def credit(self, sensor_bits: float = 800.0, fish_bits: float = 300.0) -> None:
        """Absorb camera credit so the next consuming draw has something to report."""
        self.pool.absorb(
            ChannelOutput("sensor", 10, 1.0, b"sensor bytes", np.zeros(4, np.uint8)), sensor_bits
        )
        self.pool.absorb(
            ChannelOutput(
                "fish",
                12,
                1.2,
                b"fish bytes",
                np.arange(3, dtype=np.uint8),
                sample_owner=("f1", "f1", "f2"),
            ),
            fish_bits,
        )

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # -- EngineLike ---------------------------------------------------------

    def receipt_context(self) -> ReceiptContext:
        return ReceiptContext(
            mode="day",
            liveness=self.liveness,
            health={"sensor": HealthState.OK, "fish": HealthState.DEGRADED},
            fish_names={card["fish_id"]: card["name"] for card in self.cards},
        )

    def snapshot(self) -> FakeSnapshot | None:
        with self._lock:
            if not self.ready:
                return None
            if self.auto_advance:
                self._index += 1
            return self._build(self._index)

    def latest_frame(self) -> Frame | None:
        if not self.ready or self.source_id is None:
            return None
        return Frame(
            index=self._index,
            timestamp=self._index / 10,
            wall_time=T0_S,
            image=np.zeros((4, 4, 3), np.uint8),
            source_id=self.source_id,
        )

    def jpeg(self, *, max_width: int = 640, quality: int = 70) -> bytes | None:
        with self._lock:
            self.jpeg_calls += 1
            self.jpeg_args.append((max_width, quality))
        return self.jpeg_bytes

    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def fish_cards(self) -> list[dict[str, Any]]:
        with self._lock:
            self.fish_calls += 1
        return [dict(card) for card in self.cards]

    def leaderboard(self, days: int = 7) -> list[dict[str, Any]]:
        ranked = sorted(self.cards, key=lambda c: -c["stats"]["chaos_score"])
        return [
            {
                "rank": i,
                "fish_id": c["fish_id"],
                "name": c["name"],
                "chaos_score": c["stats"]["chaos_score"],
                "bits": c["stats"]["bits_total"] * days / 7,
            }
            for i, c in enumerate(ranked, start=1)
        ]

    # -- internals ----------------------------------------------------------

    def _build(self, index: int) -> FakeSnapshot:
        t = index / 10
        return FakeSnapshot(
            frame_index=index,
            t=t,
            wall_time=T0_S + t,
            fps=9.97,
            mode=LightMode.DAY,
            width=self.width,
            height=self.height,
            tracks=self.tracks,
            channels={
                "sensor": FakeChannelStats("sensor", 9876.54321, 3.2, 1.0e6, HealthState.OK),
                "fish": FakeChannelStats("fish", 123.4, 1.9, 4.2e4, HealthState.DEGRADED),
                "system": FakeChannelStats("system", 2560.0, self.system_h, 2.56e5, HealthState.OK),
            },
            pool={"sensor": 2048.0, "fish": 77.5},
            health=self.health,
            overall=HealthState.DEGRADED,
            liveness=self.liveness,
        )


def without(engine: FakeEngine, **changes: Any) -> FakeEngine:
    """A copy of ``engine`` with dataclass fields changed (fresh RNG and pool)."""
    return replace(engine, **changes)
