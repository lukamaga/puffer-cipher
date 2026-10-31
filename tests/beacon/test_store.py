"""SQLite store: persistence, verified appends, the pending reveal, schema, threads."""

from __future__ import annotations

import dataclasses
import sqlite3
import sys
import threading
from contextlib import closing
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from puffer.beacon.pulse import ErrorCode, Pulse, canonical_summary, create_pulse
from puffer.beacon.store import (
    SCHEMA_VERSION,
    BeaconStore,
    ChainLinkError,
    CorruptStoreError,
    PendingMismatchError,
    PendingReveal,
    SchemaVersionError,
)
from puffer.beacon.vectors import (
    FOREIGN_SEED,
    build_chain,
    vector_local_random,
    vector_summary,
)


@pytest.fixture
def chain(signing_key: SigningKey) -> list[Pulse]:
    return build_chain(signing_key, 6)


def _pending_after(pulse: Pulse) -> PendingReveal:
    """The reveal that the vector pulse ``pulse`` precommits to."""
    nxt = pulse.index + 1
    return PendingReveal(nxt, vector_local_random(nxt), canonical_summary(vector_summary(nxt)))


def _fill(store: BeaconStore, chain: list[Pulse]) -> None:
    for pulse in chain:
        store.append(pulse, pending=_pending_after(pulse))


def _raw_sql(path: Path, *statements: str) -> None:
    """Edit the database behind the store's back, as an attacker or a bug would."""
    with closing(sqlite3.connect(path)) as conn, conn:
        for statement in statements:
            conn.execute(statement)


def test_empty_store(memory_store: BeaconStore) -> None:
    assert memory_store.latest() is None
    assert memory_store.count() == 0
    assert memory_store.get(0) is None
    assert memory_store.range() == []
    assert memory_store.get_pending() is None
    assert memory_store.schema_version == SCHEMA_VERSION


def test_queries(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    _fill(memory_store, chain)
    assert memory_store.count() == 6
    assert memory_store.latest() == chain[-1]
    assert memory_store.get(3) == chain[3]
    assert memory_store.get(99) is None
    assert memory_store.range(2, 3) == chain[2:5]
    assert memory_store.range(4, 50) == chain[4:]
    assert memory_store.range(0, 0) == []
    with pytest.raises(ValueError, match="non-negative"):
        memory_store.range(-1, 5)


def test_lookups_beyond_the_index_range_find_nothing(
    memory_store: BeaconStore, chain: list[Pulse]
) -> None:
    """Indices come from URLs; one that cannot exist is "not found", not a crash."""
    _fill(memory_store, chain)
    assert memory_store.get(2**53) is None
    assert memory_store.get(2**64) is None
    assert memory_store.range(2**64) == []
    assert memory_store.range(0, 2**64) == chain


def test_persistence_across_reopen(tmp_path: Path, chain: list[Pulse]) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        _fill(store, chain[:4])
    with BeaconStore(path) as store:
        assert store.count() == 4
        assert store.range(0, 10) == chain[:4]
        assert store.get_pending() == _pending_after(chain[3])
        _fill(store, chain[4:])
    with BeaconStore(str(path)) as store:
        assert store.latest() == chain[-1]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_new_database_file_is_private(tmp_path: Path, chain: list[Pulse]) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        _fill(store, chain[:1])
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_existing_database_file_is_made_private(tmp_path: Path, chain: list[Pulse]) -> None:
    """The pending reveal is secret, whoever created the file (DESIGN 9.1 shares ``puffer.db``)."""
    path = tmp_path / "puffer.db"
    with closing(sqlite3.connect(path)) as conn:  # e.g. another component opened it first
        conn.execute("CREATE TABLE fish_daily (fish_id TEXT)")
    path.chmod(0o644)
    with BeaconStore(path) as store:
        _fill(store, chain[:1])
        assert store.get_pending() is not None
    assert path.stat().st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------
# append verifies before inserting
# --------------------------------------------------------------------------


def _rejected(store: BeaconStore, pulse: Pulse) -> tuple[ErrorCode, ...]:
    before = (store.count(), store.get_pending())
    with pytest.raises(ChainLinkError) as info:
        store.append(pulse)
    assert (store.count(), store.get_pending()) == before, "a rejected append changed state"
    return info.value.errors


def test_empty_store_accepts_only_genesis(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    assert _rejected(memory_store, chain[1]) == (ErrorCode.BAD_INDEX,)
    memory_store.append(chain[0])
    assert memory_store.count() == 1


def test_append_rejects_broken_links(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    _fill(memory_store, chain[:3])
    assert ErrorCode.BAD_INDEX in _rejected(memory_store, chain[4])  # gap
    assert ErrorCode.BAD_INDEX in _rejected(memory_store, chain[2])  # duplicate
    forged = dataclasses.replace(chain[3], prev_output=bytes(64))
    assert ErrorCode.BROKEN_LINK in _rejected(memory_store, forged)


def test_append_rejects_invalid_signature(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    _fill(memory_store, chain[:2])
    forged = dataclasses.replace(chain[2], signature=bytes(64))
    assert ErrorCode.BAD_SIGNATURE in _rejected(memory_store, forged)


def test_append_pins_the_genesis_key(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    _fill(memory_store, chain[:2])
    foreign = build_chain(SigningKey(FOREIGN_SEED), 3)[2]
    errors = _rejected(memory_store, foreign)
    assert errors[0] == ErrorCode.WRONG_PUBLIC_KEY


def test_chain_link_error_message(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    with pytest.raises(ChainLinkError, match=r"pulse 1 rejected: bad_index") as info:
        memory_store.append(chain[1])
    assert info.value.index == 1


# --------------------------------------------------------------------------
# pending reveal
# --------------------------------------------------------------------------


def test_append_rejects_mismatched_pending_atomically(
    memory_store: BeaconStore, chain: list[Pulse]
) -> None:
    _fill(memory_store, chain[:2])
    wrong_value = PendingReveal(3, vector_local_random(99), '{"x":1}')
    wrong_index = dataclasses.replace(_pending_after(chain[2]), index=4)
    for pending in (wrong_value, wrong_index):
        with pytest.raises(PendingMismatchError):
            memory_store.append(chain[2], pending=pending)
        assert memory_store.count() == 2
        assert memory_store.get_pending() == _pending_after(chain[1])


def test_revealed_pending_is_deleted(memory_store: BeaconStore, chain: list[Pulse]) -> None:
    _fill(memory_store, chain[:2])
    memory_store.append(chain[2])  # mirror-style append: reveals pending #2, stores none
    assert memory_store.get_pending() is None


def test_set_pending_only_accepts_the_committed_value(tmp_path: Path, chain: list[Pulse]) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        with pytest.raises(PendingMismatchError, match="no pulse"):
            store.set_pending(_pending_after(chain[0]))
        store.append(chain[0])
        with pytest.raises(PendingMismatchError):
            store.set_pending(_pending_after(chain[1]))
        store.set_pending(_pending_after(chain[0]))
        store.set_pending(_pending_after(chain[0]))  # idempotent replace
    with BeaconStore(path) as store:
        assert store.get_pending() == _pending_after(chain[0])


def test_pending_reveal_hides_its_secret() -> None:
    pending = PendingReveal(1, bytes([0xAB]) * 64, '{"a":1}')
    assert repr(pending) == "PendingReveal(index=1, summary='{\"a\":1}')"


@pytest.mark.parametrize(
    ("index", "local_random", "summary", "match"),
    [
        (0, bytes(64), "{}", "index"),
        (1, bytes(63), "{}", "64 bytes"),
        (1, bytes(64), '{"b":1,"a":2}', "canonical"),
        (1, bytes(64), '{"a":0.5}', "floats"),
    ],
)
def test_pending_reveal_validation(
    index: int, local_random: bytes, summary: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PendingReveal(index, local_random, summary)


# --------------------------------------------------------------------------
# schema and corruption
# --------------------------------------------------------------------------


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "beacon.db"
    BeaconStore(path).close()
    _raw_sql(path, "PRAGMA user_version = 99")
    with pytest.raises(SchemaVersionError, match="99"):
        BeaconStore(path)


def test_corrupt_rows_are_reported(tmp_path: Path, chain: list[Pulse]) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        _fill(store, chain[:2])
    _raw_sql(
        path,
        "UPDATE pulses SET pulse_json = '{}' WHERE pulse_index = 1",
        "UPDATE pending SET summary = 'nope'",
    )
    with BeaconStore(path) as store:
        with pytest.raises(CorruptStoreError, match="row 1"):
            store.latest()
        with pytest.raises(CorruptStoreError, match="pending"):
            store.get_pending()


def test_misfiled_row_is_reported(tmp_path: Path, chain: list[Pulse]) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        _fill(store, chain[:1])
    _raw_sql(path, "UPDATE pulses SET pulse_index = 5")
    with BeaconStore(path) as store, pytest.raises(CorruptStoreError, match="holds pulse 0"):
        store.get(5)


# --------------------------------------------------------------------------
# threads
# --------------------------------------------------------------------------


def test_concurrent_readers_and_writer(signing_key: SigningKey) -> None:
    chain = build_chain(signing_key, 40)
    errors: list[BaseException] = []
    done = threading.Event()

    with BeaconStore() as store:

        def reader() -> None:
            try:
                while not done.is_set():
                    latest = store.latest()
                    if latest is not None:
                        assert store.get(latest.index) == latest
                        assert len(store.range(0, 100)) >= latest.index + 1
            except BaseException as exc:  # surfaced in the main thread
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        try:
            _fill(store, chain)
        finally:
            done.set()
            for thread in threads:
                thread.join(timeout=10)
        assert not errors
        assert store.count() == 40


def test_racing_writers_cannot_fork_the_chain(signing_key: SigningKey) -> None:
    genesis, honest = build_chain(signing_key, 2)
    # A second, equally valid continuation of genesis: only one may win.
    rival = create_pulse(
        signing_key,
        index=1,
        timestamp_ms=honest.timestamp_ms + 1,
        period_ms=honest.period_ms,
        local_random=honest.local_random,
        next_local_random=vector_local_random(77),
        prev_output=genesis.output,
        summary={"rival": True},
    )
    with BeaconStore() as store:
        store.append(genesis)
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def write(pulse: Pulse) -> None:
            barrier.wait()
            try:
                store.append(pulse)
                outcomes.append("ok")
            except ChainLinkError:
                outcomes.append("rejected")

        threads = [threading.Thread(target=write, args=(p,)) for p in (honest, rival)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert sorted(outcomes) == ["ok", "rejected"]
        assert store.count() == 2
