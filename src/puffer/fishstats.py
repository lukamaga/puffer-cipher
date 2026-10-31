"""Per-fish daily statistics: credited bits, time seen, distance, chaos - and the leaderboard.

The dashboard tells a story about individual fish ("Professor Puff earned 9 kbit
today and is the most chaotic swimmer this week"). The numbers behind it must
survive restarts, so :class:`FishStats` keeps one aggregate row per fish per UTC
day in SQLite (``data_dir/puffer.db``, DESIGN.md section 9.2):

======== ==============================================================
bits     credited fish-channel bits attributed to the fish (see below)
seen_s   seconds the fish was tracked
distance pixels swum (while moving; resting jitter is not distance)
chaos    sum and count of chaos-score samples, so the mean is exact
======== ==============================================================

Design decisions
----------------
* **Write batching on the frame clock.** :meth:`FishStats.record` is called from
  the engine thread for every fish on every frame (~200 calls/s). It only
  updates an in-memory accumulator; the accumulated deltas are upserted in one
  transaction every ``flush_every_s`` seconds of *frame* time (so recorded
  video and the simulator batch exactly like a live camera) and on
  :meth:`FishStats.close`. A crash loses at most that many seconds of
  statistics - nothing here is security relevant.
* **Reads see everything.** :meth:`leaderboard` and :meth:`totals` flush first,
  so an API caller never sees numbers older than the last frame.
* **UTC days from wall time.** The day bucket comes from the frame's
  ``wall_time`` (frame timestamps are monotonic and have no calendar), in UTC
  so a leaderboard does not shift with daylight saving time.
* **What "bits" means.** The engine records the credit the fish channel *earned*
  (health-gated, estimator-capped, safety-factored, split between fish by
  sample count), before the pool's saturating ledger. That is the fish's
  contribution rate; how much of it a particular key consumed is on that key's
  receipt.
* **Thread-safe.** One connection (``check_same_thread=False``) behind a lock:
  the engine thread records while API threads read. ``PRAGMA synchronous =
  NORMAL``: statistics do not deserve an fsync per batch on an SD card.
* **Versioned schema.** ``PRAGMA user_version`` holds the schema version;
  migrations are append-only and a database from a newer version is refused
  rather than misread.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Self

__all__ = ["MEMORY", "SCHEMA_VERSION", "FishStats", "FishStatsError", "FishTotals"]

MEMORY: Final = ":memory:"
SCHEMA_VERSION: Final = 1

# _MIGRATIONS[v - 1] upgrades schema version v - 1 to v. Append only.
_MIGRATIONS: Final[tuple[tuple[str, ...], ...]] = (
    (
        """
        CREATE TABLE fish_daily (
            fish_id     TEXT    NOT NULL,
            day         TEXT    NOT NULL,                  -- YYYY-MM-DD, UTC
            name        TEXT    NOT NULL,
            bits        REAL    NOT NULL DEFAULT 0 CHECK (bits >= 0),
            seen_s      REAL    NOT NULL DEFAULT 0 CHECK (seen_s >= 0),
            distance_px REAL    NOT NULL DEFAULT 0 CHECK (distance_px >= 0),
            chaos_sum   REAL    NOT NULL DEFAULT 0,
            chaos_count INTEGER NOT NULL DEFAULT 0 CHECK (chaos_count >= 0),
            PRIMARY KEY (fish_id, day)
        ) WITHOUT ROWID
        """,
        "CREATE INDEX fish_daily_day ON fish_daily (day)",
    ),
)
assert len(_MIGRATIONS) == SCHEMA_VERSION

_UPSERT: Final = """
    INSERT INTO fish_daily (fish_id, day, name, bits, seen_s, distance_px, chaos_sum, chaos_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (fish_id, day) DO UPDATE SET
        name        = excluded.name,
        bits        = bits + excluded.bits,
        seen_s      = seen_s + excluded.seen_s,
        distance_px = distance_px + excluded.distance_px,
        chaos_sum   = chaos_sum + excluded.chaos_sum,
        chaos_count = chaos_count + excluded.chaos_count
"""

# SQLite returns the bare ``name`` column from the row holding MAX(day): the name
# a fish had most recently, which is the one to display.
_TOTALS: Final = """
    SELECT fish_id, name, MAX(day), SUM(bits), SUM(seen_s), SUM(distance_px),
           SUM(chaos_sum), SUM(chaos_count),
           SUM(CASE WHEN day = ? THEN bits ELSE 0 END)
    FROM fish_daily
    WHERE day >= ?
    GROUP BY fish_id
"""


class FishStatsError(RuntimeError):
    """The statistics database cannot be used (e.g. written by a newer version)."""


@dataclass(frozen=True, slots=True)
class FishTotals:
    """Aggregates for one fish: all-time (or over a window) plus today's bits.

    ``chaos_score`` is the mean of the recorded chaos samples, ``None`` if
    there were none.
    """

    fish_id: str
    name: str
    bits_total: float
    bits_today: float
    seen_s: float
    distance_px: float
    chaos_score: float | None


class _Pending:
    """Deltas for one (fish, day) not yet written."""

    __slots__ = ("bits", "chaos_count", "chaos_sum", "distance_px", "name", "seen_s")

    def __init__(self, name: str) -> None:
        self.name = name
        self.bits = 0.0
        self.seen_s = 0.0
        self.distance_px = 0.0
        self.chaos_sum = 0.0
        self.chaos_count = 0


def utc_day(wall_time: float) -> str:
    """``YYYY-MM-DD`` of a Unix timestamp, in UTC."""
    return dt.datetime.fromtimestamp(wall_time, dt.UTC).date().isoformat()


def _amount(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return v


class FishStats:
    """Daily per-fish aggregates in SQLite; ``path`` may be :data:`MEMORY`.

    ``flush_every_s`` is in frame time (the ``t`` passed to :meth:`record`);
    ``clock`` (Unix seconds) decides what "today" is for reads.

    Raises:
        FishStatsError: the database uses an unknown (newer) schema version.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] = MEMORY,
        *,
        flush_every_s: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not (math.isfinite(flush_every_s) and flush_every_s >= 0):
            raise ValueError("flush_every_s must be finite and non-negative")
        self.path = os.fspath(path)
        self.flush_every_s = float(flush_every_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._pending: dict[tuple[str, str], _Pending] = {}
        self._last_flush_t: float | None = None
        self._closed = False
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Flush pending deltas and close the database (idempotent)."""
        with self._lock:
            if self._closed:
                return
            try:
                self._flush_locked()
            finally:
                self._closed = True
                self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # -- writing --------------------------------------------------------------

    def record(
        self,
        fish_id: str,
        *,
        t: float,
        wall_time: float,
        name: str | None = None,
        bits: float = 0.0,
        seen_s: float = 0.0,
        distance_px: float = 0.0,
        chaos_score: float | None = None,
    ) -> None:
        """Add one observation's deltas for ``fish_id`` at frame time ``t``.

        ``bits``, ``seen_s`` and ``distance_px`` are increments; ``chaos_score``
        is one sample of the fish's current score (averaged on read). Flushes
        when ``flush_every_s`` of frame time has passed since the last flush.

        Raises:
            ValueError: an amount is negative or not finite.
        """
        if not fish_id:
            raise ValueError("fish_id must be non-empty")
        increments = (
            _amount("bits", bits),
            _amount("seen_s", seen_s),
            _amount("distance_px", distance_px),
        )
        chaos = None if chaos_score is None else _amount("chaos_score", chaos_score)
        day = utc_day(wall_time)
        with self._lock:
            self._check_open()
            entry = self._pending.get((fish_id, day))
            if entry is None:
                entry = self._pending[(fish_id, day)] = _Pending(name or fish_id)
            elif name:
                entry.name = name
            entry.bits += increments[0]
            entry.seen_s += increments[1]
            entry.distance_px += increments[2]
            if chaos is not None:
                entry.chaos_sum += chaos
                entry.chaos_count += 1
            if self._last_flush_t is None or t < self._last_flush_t:
                self._last_flush_t = t  # first record, or the frame clock restarted
            elif t - self._last_flush_t >= self.flush_every_s:
                self._flush_locked()
                self._last_flush_t = t

    def flush(self) -> None:
        """Write all pending deltas in one transaction."""
        with self._lock:
            self._check_open()
            self._flush_locked()

    # -- reading --------------------------------------------------------------

    def totals(self, days: int | None = None) -> dict[str, FishTotals]:
        """Aggregates per fish over the last ``days`` UTC days (``None``: all time)."""
        today = utc_day(self._clock())
        since = "" if days is None else self._since(days)
        with self._lock:
            self._check_open()
            self._flush_locked()
            rows = self._conn.execute(_TOTALS, (today, since)).fetchall()
        return {row[0]: _totals(row) for row in rows}

    def leaderboard(self, days: int = 7) -> list[dict[str, Any]]:
        """``[{rank, fish_id, name, chaos_score, bits}]`` over the last ``days`` UTC days.

        Ranked by mean chaos score (fish without chaos samples last), then by
        bits, then by fish id, so the order is total and stable.
        """
        totals = self.totals(days)
        ranked = sorted(
            totals.values(),
            key=lambda f: (
                f.chaos_score is None,
                -(f.chaos_score or 0.0),
                -f.bits_total,
                f.fish_id,
            ),
        )
        return [
            {
                "rank": rank,
                "fish_id": f.fish_id,
                "name": f.name,
                "chaos_score": round(f.chaos_score, 4) if f.chaos_score is not None else None,
                "bits": round(f.bits_total, 3),
            }
            for rank, f in enumerate(ranked, start=1)
        ]

    # -- internals --------------------------------------------------------------

    def _since(self, days: int) -> str:
        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise ValueError("days must be a positive integer")
        now = dt.datetime.fromtimestamp(self._clock(), dt.UTC).date()
        return (now - dt.timedelta(days=days - 1)).isoformat()

    def _check_open(self) -> None:
        if self._closed:
            raise FishStatsError("FishStats is closed")

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        rows = [
            (fish_id, day, p.name, p.bits, p.seen_s, p.distance_px, p.chaos_sum, p.chaos_count)
            for (fish_id, day), p in self._pending.items()
        ]
        self._conn.execute("BEGIN")
        try:
            self._conn.executemany(_UPSERT, rows)
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        self._pending.clear()

    def _migrate(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise FishStatsError(
                f"{self.path}: schema version {version} is newer than supported ({SCHEMA_VERSION})"
            )
        for target in range(version + 1, SCHEMA_VERSION + 1):
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _MIGRATIONS[target - 1]:
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {target:d}")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")


def _totals(row: tuple[Any, ...]) -> FishTotals:
    fish_id, name, _day, bits, seen_s, distance_px, chaos_sum, chaos_count, today = row
    return FishTotals(
        fish_id=str(fish_id),
        name=str(name),
        bits_total=float(bits),
        bits_today=float(today),
        seen_s=float(seen_s),
        distance_px=float(distance_px),
        chaos_score=float(chaos_sum) / int(chaos_count) if chaos_count else None,
    )
