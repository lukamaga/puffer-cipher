"""What the CLI needs from the rest of puffer, as structural types.

The CLI is a thin, user-facing layer over the engine of DESIGN.md 9.2. It
depends on these :class:`typing.Protocol` classes instead of the concrete
``EntropyEngine`` / ``PufferConfig`` / ``FishStats`` for two reasons:

* **Testability.** Every command and every error path is unit-tested with
  small fakes (``tests/cli/fakes.py``) that run in milliseconds and can
  simulate situations a real tank produces only rarely - liveness that never
  verifies, a source that dies mid-collection, a pool that never fills.
* **A narrow contract.** Listing only the members the CLI reads documents
  exactly what it relies on. mypy checks the real engine against these
  protocols where :mod:`puffer.cli.wiring` returns it, so drift between the
  engine and the CLI is a type error, not a runtime surprise.

Read-only members are declared as properties so frozen dataclasses (the
engine's snapshot types) satisfy them, and so containers can be declared
covariantly (``Mapping``/``Sequence``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from puffer.types import (
    Frame,
    HealthReport,
    HealthState,
    LightMode,
    LivenessStatus,
    TrackSnapshot,
)

if TYPE_CHECKING:
    from puffer.entropy.receipt import EntropyReceipt

__all__ = [
    "ChannelStatsLike",
    "ConfigLike",
    "EngineLike",
    "FishStatsLike",
    "RngLike",
    "SnapshotLike",
]


class ChannelStatsLike(Protocol):
    """One entropy channel in a snapshot (``engine.ChannelStats``)."""

    @property
    def name(self) -> str: ...

    @property
    def bits_per_s(self) -> float: ...

    @property
    def h_per_sample(self) -> float | None: ...

    @property
    def credited_total(self) -> float: ...

    @property
    def health(self) -> HealthState: ...


class SnapshotLike(Protocol):
    """The engine state after one frame (``engine.EngineSnapshot``)."""

    @property
    def frame_index(self) -> int: ...

    @property
    def t(self) -> float: ...

    @property
    def fps(self) -> float: ...

    @property
    def mode(self) -> LightMode: ...

    @property
    def tracks(self) -> Sequence[TrackSnapshot]: ...

    @property
    def channels(self) -> Mapping[str, ChannelStatsLike]: ...

    @property
    def pool(self) -> Mapping[str, float]:
        """Current (undrawn) credit per camera channel."""
        ...

    @property
    def health(self) -> Mapping[str, HealthReport]: ...

    @property
    def overall(self) -> HealthState: ...

    @property
    def liveness(self) -> LivenessStatus: ...


class RngLike(Protocol):
    """The engine's ``PufferRNG``: the only way the CLI obtains secret bytes.

    Public-value helpers (``salt()``) are optional and looked up with
    ``getattr`` (see :func:`puffer.cli.collect.public_random`), so the CLI
    keeps working against an RNG that predates them.
    """

    def generate(self, n: int, purpose: str = ...) -> tuple[bytes, EntropyReceipt]: ...


class EngineLike(Protocol):
    """The subset of ``EntropyEngine`` (DESIGN.md 9.2) the CLI drives."""

    @property
    def rng(self) -> RngLike: ...

    def collect(
        self, min_credit_bits: float, *, timeout_s: float, max_frames: int | None = None
    ) -> bool:
        """Process frames until the camera channels hold ``min_credit_bits`` of credit.

        ``False`` when ``timeout_s`` or ``max_frames`` is reached, or a finite
        source ends, first (DESIGN.md 9.2).
        """
        ...

    def snapshot(self) -> SnapshotLike | None: ...

    def latest_frame(self) -> Frame | None: ...

    def fish_cards(self) -> list[dict[str, Any]]:
        """DESIGN.md 7.1 ``FishCard`` dicts (``fish_id``, ``name``, ``color``...)."""
        ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class ConfigLike(Protocol):
    """The ``PufferConfig`` fields (DESIGN.md 9.1) the CLI reads or overrides."""

    @property
    def source(self) -> str: ...

    @property
    def liveness(self) -> str: ...

    @property
    def data_dir(self) -> str | os.PathLike[str]: ...

    @property
    def pool_capacity_bits(self) -> float: ...

    @property
    def host(self) -> str: ...

    @property
    def port(self) -> int: ...


class FishStatsLike(Protocol):
    """``puffer.fishstats.FishStats``: the persisted leaderboard."""

    def leaderboard(self, days: int = ...) -> Sequence[Mapping[str, Any]]:
        """Rows ``{rank, fish_id, name, chaos_score, bits}`` (DESIGN.md 7.1)."""
        ...

    def close(self) -> None: ...
