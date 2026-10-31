"""What the HTTP layer needs from the rest of puffer, as structural types.

The API is written against these Protocols instead of the concrete
``puffer.engine.EntropyEngine`` (DESIGN.md 9.2) for two reasons:

* Unit tests drive every endpoint and the WebSocket hub with a small fake
  engine: no camera, no vision pipeline, no background thread.
* The surface is explicit. The API can only touch what is listed here, and
  the only way it can obtain secret bytes is :class:`RngLike` - there is no
  ``os.urandom`` or ``secrets`` call anywhere in :mod:`puffer.api`.

Members are declared as read-only properties so that frozen dataclasses (the
engine's ``EngineSnapshot`` and ``ChannelStats``) satisfy them structurally.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from puffer.types import Frame, HealthReport, LivenessStatus, TrackSnapshot


class ReceiptLike(Protocol):
    """An entropy receipt (``puffer.entropy.EntropyReceipt``): accounting, never secret."""

    def to_dict(self) -> dict[str, Any]: ...


class RngLike(Protocol):
    """The fish-seeded generator (``puffer.entropy.PufferRNG``).

    ``nonce()`` may or may not consume camera credit depending on the
    PufferRNG version; either way it is OS-keyed and never weaker than
    ``os.urandom``.
    """

    def generate(self, n: int, purpose: str = ...) -> tuple[bytes, ReceiptLike]: ...

    def generate_key(self) -> tuple[bytes, ReceiptLike]: ...

    def nonce(self) -> bytes: ...

    def display_bits(self, n_bits: int) -> str: ...


class ChannelStatsLike(Protocol):
    """``puffer.engine.ChannelStats``: one entropy channel's live numbers."""

    @property
    def name(self) -> str: ...

    @property
    def bits_per_s(self) -> float: ...

    @property
    def h_per_sample(self) -> float | None: ...

    @property
    def credited_total(self) -> float: ...

    @property
    def health(self) -> str: ...


class SnapshotLike(Protocol):
    """``puffer.engine.EngineSnapshot``: the engine state after one frame."""

    @property
    def frame_index(self) -> int: ...

    @property
    def t(self) -> float: ...

    @property
    def wall_time(self) -> float: ...

    @property
    def fps(self) -> float: ...

    @property
    def mode(self) -> str: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def tracks(self) -> Sequence[TrackSnapshot]: ...

    @property
    def channels(self) -> Mapping[str, ChannelStatsLike]: ...

    @property
    def pool(self) -> Mapping[str, float]: ...

    @property
    def health(self) -> Mapping[str, HealthReport]: ...

    @property
    def overall(self) -> str: ...

    @property
    def liveness(self) -> LivenessStatus: ...


class ConfigLike(Protocol):
    """The subset of ``puffer.config.PufferConfig`` the API reads."""

    @property
    def source(self) -> str: ...

    @property
    def pool_capacity_bits(self) -> float: ...

    @property
    def stream_fps(self) -> float: ...

    @property
    def stream_width(self) -> int: ...

    @property
    def jpeg_quality(self) -> int: ...


class EngineLike(Protocol):
    """``puffer.engine.EntropyEngine`` as seen by the API (DESIGN.md 9.2).

    Every method may be called from any thread. ``jpeg`` and ``fish_cards``
    may block (image encoding, SQLite) and are always called off the event
    loop.
    """

    @property
    def config(self) -> ConfigLike: ...

    @property
    def rng(self) -> RngLike: ...

    def snapshot(self) -> SnapshotLike | None: ...

    def latest_frame(self) -> Frame | None: ...

    def jpeg(self, *, max_width: int = 640, quality: int = 70) -> bytes | None: ...

    def subscribe(self, callback: Callable[[SnapshotLike], None]) -> Callable[[], None]: ...

    def fish_cards(self) -> list[dict[str, Any]]: ...

    def leaderboard(self, days: int = 7) -> list[dict[str, Any]]: ...
