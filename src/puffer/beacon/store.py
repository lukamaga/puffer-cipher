"""SQLite persistence for the beacon chain and its one secret: the pending reveal.

Two tables, one invariant
-------------------------
``pulses`` holds every published pulse (primary key ``pulse_index``, value
the pulse JSON). ``pending`` holds at most one row: the ``local_random`` that
the latest pulse precommitted to, together with the summary recorded when it
was drawn. The store guarantees, in a single SQLite transaction per write:

* every appended pulse fully verifies (signature, output, summary hash) and
  links to the current latest pulse under the chain's genesis key - the
  table can never hold an invalid chain;
* a pending reveal is only ever stored next to the pulse that commits to it
  (``pending.index == latest.index + 1`` and
  ``BLAKE2b-512(pending.local_random) == latest.precommitment``), and a
  pending row is deleted as soon as a pulse reveals it.

Appending a pulse and storing the next pending reveal happen in the *same*
transaction (``append(pulse, pending=...)``). That is what makes the beacon
crash-consistent: there is no instant at which a published precommitment
lacks the value needed to honour it (see :mod:`puffer.beacon.service`).

The database is server-private
------------------------------
Until it is revealed, the pending ``local_random`` is a secret: whoever reads
it knows the next pulse's ``local_random`` one period early. The database
file is therefore created with mode ``0600``, and an existing file (which
another component may have created first: DESIGN.md 9.1 keeps one
``puffer.db``) loses any group and other permissions on open; SQLite gives its
journal the same permissions. Never serve, back up to shared storage or copy
this file to a public mirror. Public data is exposed only through :meth:`latest`,
:meth:`get` and :meth:`range`. :class:`PendingReveal` keeps the secret out of
its ``repr`` so it cannot leak into logs.

Durability matters as much as atomicity: ``PRAGMA synchronous = FULL``
makes a commit durable before ``append`` returns, so a pulse is never handed
to listeners and then lost (together with its pending reveal) by a power cut.

Thread safety: one connection opened with ``check_same_thread=False`` and
serialized by a re-entrant lock, so the beacon thread and API handlers can
share one store. Writes use ``BEGIN IMMEDIATE``, which also serializes
writers in other processes; a concurrent writer that loses the race fails
the link check instead of forking the chain.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Final, Self

from puffer.beacon.pulse import (
    LOCAL_RANDOM_BYTES,
    MAX_JSON_INT,
    BeaconFormatError,
    ErrorCode,
    Pulse,
    check_canonical_summary,
    precommit,
    verify_pulse,
)

MEMORY: Final = ":memory:"
"""Path value for a throw-away in-memory store (tests, demos)."""

SCHEMA_VERSION: Final = 1
"""Current schema version, stored in ``PRAGMA user_version``."""

# _MIGRATIONS[v - 1] upgrades a database from schema version v - 1 to v.
# Append new migrations; never edit a released one.
_MIGRATIONS: Final[tuple[tuple[str, ...], ...]] = (
    (
        """
        CREATE TABLE pulses (
            pulse_index INTEGER PRIMARY KEY CHECK (pulse_index >= 0),
            pulse_json  TEXT    NOT NULL
        )
        """,
        """
        CREATE TABLE pending (
            slot         INTEGER PRIMARY KEY CHECK (slot = 0),
            pulse_index  INTEGER NOT NULL CHECK (pulse_index >= 1),
            local_random BLOB    NOT NULL CHECK (length(local_random) = 64),
            summary      TEXT    NOT NULL
        )
        """,
    ),
)
assert len(_MIGRATIONS) == SCHEMA_VERSION


class BeaconStoreError(Exception):
    """Base class for store failures."""


class SchemaVersionError(BeaconStoreError):
    """The database was written by an incompatible (newer) schema."""


class ChainLinkError(BeaconStoreError):
    """A pulse was refused because it does not verify and extend the stored chain."""

    def __init__(self, index: int, errors: tuple[ErrorCode, ...]) -> None:
        self.index = index
        self.errors = errors
        super().__init__(f"pulse {index} rejected: {', '.join(errors)}")


class PendingMismatchError(BeaconStoreError):
    """A pending reveal does not match the precommitment it is stored against."""


class CorruptStoreError(BeaconStoreError):
    """A stored row is unreadable or inconsistent with its key."""


@dataclass(frozen=True, slots=True)
class PendingReveal:
    """The precommitted next ``local_random`` and the summary drawn with it.

    SECRET until pulse :attr:`index` reveals it; excluded from ``repr``.
    """

    index: int  # the pulse that will reveal this value
    local_random: bytes = field(repr=False)
    summary: str  # canonical summary JSON describing the entropy of ``local_random``

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 1:
            raise ValueError("pending reveal index must be an int >= 1")
        if type(self.local_random) is not bytes or len(self.local_random) != LOCAL_RANDOM_BYTES:
            raise ValueError(f"pending local_random must be {LOCAL_RANDOM_BYTES} bytes")
        # Checked now, not at reveal time: a non-canonical summary would make
        # the precommitted value impossible to publish and stall the chain.
        check_canonical_summary(self.summary)


def check_pending_matches(pulse: Pulse, pending: PendingReveal) -> None:
    """Require ``pending`` to be exactly the value that ``pulse`` precommits to.

    Raises:
        PendingMismatchError: wrong index or wrong preimage.
    """
    if pending.index != pulse.index + 1:
        raise PendingMismatchError(
            f"pending reveal is for pulse {pending.index}, expected {pulse.index + 1}"
        )
    if precommit(pending.local_random) != pulse.precommitment:
        raise PendingMismatchError(
            f"pending reveal does not match the precommitment of pulse {pulse.index}"
        )


class BeaconStore:
    """Durable, verified pulse chain plus the server-private pending reveal.

    ``path`` is a database file or :data:`MEMORY`. Use as a context manager
    or call :meth:`close`.

    Raises:
        SchemaVersionError: the file uses an unknown schema version.
    """

    def __init__(self, path: str | os.PathLike[str] = MEMORY) -> None:
        location = os.fspath(path)
        if location != MEMORY:
            _make_private(location)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(location, isolation_level=None, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA synchronous = FULL")
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the connection. Committed data is already durable."""
        with self._lock:
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
        """Schema version of the open database (``PRAGMA user_version``)."""
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # -- public chain -------------------------------------------------------

    def append(self, pulse: Pulse, *, pending: PendingReveal | None = None) -> None:
        """Atomically append ``pulse`` and replace the pending reveal.

        ``pulse`` must fully verify and extend the current latest pulse under
        the genesis key (or be a valid genesis pulse when the store is empty).
        The pending reveal that ``pulse`` just revealed is deleted; ``pending``,
        when given, must be the value that ``pulse`` precommits to and is
        stored in the same transaction. Without ``pending`` the store acts as
        a passive mirror of someone else's chain.

        Raises:
            ChainLinkError: the pulse does not verify or does not extend the chain.
            PendingMismatchError: ``pending`` does not match ``pulse``.
        """
        if pending is not None:
            check_pending_matches(pulse, pending)
        with self._transaction() as conn:
            latest = self._latest(conn)
            if latest is None:
                errors = verify_pulse(pulse).errors
                if pulse.index != 0:
                    errors = (*errors, ErrorCode.BAD_INDEX)
            else:
                errors = verify_pulse(pulse, public_key=latest.public_key, prev=latest).errors
            if errors:
                raise ChainLinkError(pulse.index, errors)
            conn.execute(
                "INSERT INTO pulses (pulse_index, pulse_json) VALUES (?, ?)",
                (pulse.index, _encode_pulse(pulse)),
            )
            conn.execute("DELETE FROM pending")
            if pending is not None:
                _insert_pending(conn, pending)

    def latest(self) -> Pulse | None:
        """The pulse with the highest index, or ``None`` for an empty chain."""
        with self._lock:
            return self._latest(self._conn)

    def get(self, index: int) -> Pulse | None:
        """The pulse with ``index``, or ``None`` if it does not exist."""
        if not 0 <= index <= MAX_JSON_INT:  # no pulse can have it (and SQLite would overflow)
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT pulse_index, pulse_json FROM pulses WHERE pulse_index = ?", (index,)
            ).fetchone()
        return None if row is None else _decode_pulse(row)

    def range(self, start: int = 0, limit: int = 50) -> list[Pulse]:
        """Up to ``limit`` pulses with ``index >= start``, ascending.

        The store imposes no upper bound on ``limit``; network-facing callers
        should cap it.

        Raises:
            ValueError: ``start`` or ``limit`` is negative.
        """
        if start < 0 or limit < 0:
            raise ValueError("start and limit must be non-negative")
        if start > MAX_JSON_INT:  # beyond every possible index (and SQLite's INTEGER)
            return []
        limit = min(limit, MAX_JSON_INT)
        with self._lock:
            rows = self._conn.execute(
                "SELECT pulse_index, pulse_json FROM pulses WHERE pulse_index >= ? "
                "ORDER BY pulse_index LIMIT ?",
                (start, limit),
            ).fetchall()
        return [_decode_pulse(row) for row in rows]

    def count(self) -> int:
        """Number of stored pulses."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM pulses").fetchone()[0])

    # -- server-private pending reveal --------------------------------------

    def set_pending(self, pending: PendingReveal) -> None:
        """Store ``pending`` against the latest pulse, replacing any previous one.

        Normal operation stores the pending reveal through :meth:`append`;
        this is for tooling such as restoring a private backup. It only accepts
        the value the latest pulse actually committed to, so it cannot be used
        to rewrite what the next pulse will reveal.

        Raises:
            PendingMismatchError: the store is empty or ``pending`` does not match.
        """
        with self._transaction() as conn:
            latest = self._latest(conn)
            if latest is None:
                raise PendingMismatchError("no pulse has precommitted to a pending reveal yet")
            check_pending_matches(latest, pending)
            conn.execute("DELETE FROM pending")
            _insert_pending(conn, pending)

    def get_pending(self) -> PendingReveal | None:
        """The secret pending reveal, or ``None`` if there is none."""
        with self._lock:
            row = self._conn.execute(
                "SELECT pulse_index, local_random, summary FROM pending WHERE slot = 0"
            ).fetchone()
        if row is None:
            return None
        try:
            return PendingReveal(index=row[0], local_random=bytes(row[1]), summary=row[2])
        except ValueError as exc:
            raise CorruptStoreError(f"unreadable pending reveal: {exc}") from exc

    # -- internals ----------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT``, rolled back on any exception.

        Explicit statements (with ``isolation_level=None``) instead of the
        sqlite3 module's implicit transactions, whose rules differ between
        Python versions; ``IMMEDIATE`` takes the write lock up front so the
        read of the latest pulse and the insert see the same state.
        """
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def _migrate(self) -> None:
        with self._transaction() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database schema version {version} is newer than supported {SCHEMA_VERSION}"
                )
            for target in range(version + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[target - 1]:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {int(target)}")

    @staticmethod
    def _latest(conn: sqlite3.Connection) -> Pulse | None:
        row = conn.execute(
            "SELECT pulse_index, pulse_json FROM pulses ORDER BY pulse_index DESC LIMIT 1"
        ).fetchone()
        return None if row is None else _decode_pulse(row)


def _make_private(location: str) -> None:
    """Create ``location`` with mode ``0600``, or strip group/other access from an existing file.

    Works on the open descriptor, so there is no window between checking and
    changing the mode. Fails (``PermissionError``) rather than continue when
    the file belongs to someone else and cannot be made private.
    """
    fd = os.open(location, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        mode = stat.S_IMODE(os.fstat(fd).st_mode)
        if mode & 0o077:
            os.fchmod(fd, mode & 0o700)
    finally:
        os.close(fd)


def _encode_pulse(pulse: Pulse) -> str:
    return json.dumps(pulse.to_json(), separators=(",", ":"))


def _decode_pulse(row: tuple[int, str]) -> Pulse:
    key, text = row
    try:
        pulse = Pulse.from_json(json.loads(text))
    except (json.JSONDecodeError, BeaconFormatError) as exc:
        raise CorruptStoreError(f"unreadable pulse row {key}: {exc}") from exc
    if pulse.index != key:
        raise CorruptStoreError(f"pulse row {key} holds pulse {pulse.index}")
    return pulse


def _insert_pending(conn: sqlite3.Connection, pending: PendingReveal) -> None:
    conn.execute(
        "INSERT INTO pending (slot, pulse_index, local_random, summary) VALUES (0, ?, ?, ?)",
        (pending.index, pending.local_random, pending.summary),
    )
