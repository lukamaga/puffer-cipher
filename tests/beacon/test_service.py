"""Beacon scheduler: chain production, restarts, crash consistency, the run loop."""

from __future__ import annotations

import itertools
import logging
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nacl.signing import SigningKey

import puffer.beacon.service as service_module
import puffer.beacon.store as store_module
from puffer.beacon.chain import verify_chain
from puffer.beacon.pulse import Pulse, canonical_summary, precommit
from puffer.beacon.service import (
    BeaconEntropyError,
    BeaconService,
    BeaconStateError,
    BeaconTimingError,
    wall_clock_ms,
)
from puffer.beacon.store import BeaconStore
from puffer.beacon.vectors import FOREIGN_SEED, VECTOR_T0_MS

from .fakes import FakeClock, FakeEntropy

PERIOD = 60_000


def _service(
    store: BeaconStore,
    key: SigningKey,
    entropy: Callable[[int, str], tuple[bytes, Any]],
    clock: FakeClock,
    period_ms: int = PERIOD,
) -> BeaconService:
    return BeaconService(store, key, entropy, period_ms=period_ms, clock_ms=clock)


def _assert_honest(store: BeaconStore, public_key: bytes) -> list[Pulse]:
    """The whole stored chain verifies and the latest precommitment can be honoured."""
    pulses = store.range(0, store.count())
    report = verify_chain(pulses, public_key)
    assert report.ok, report.errors
    assert [p.index for p in pulses] == list(range(len(pulses)))
    if pulses:
        pending = store.get_pending()
        assert pending is not None
        assert pending.index == pulses[-1].index + 1
        assert precommit(pending.local_random) == pulses[-1].precommitment
    return pulses


# --------------------------------------------------------------------------
# Producing a chain
# --------------------------------------------------------------------------


def test_hundred_pulses_form_a_valid_chain(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    service = _service(memory_store, signing_key, entropy, clock)
    for _ in range(100):
        service.tick()
        clock.advance(PERIOD)
    pulses = _assert_honest(memory_store, service.public_key)
    assert len(pulses) == 100
    assert [p.timestamp_ms for p in pulses] == [VECTOR_T0_MS + i * PERIOD for i in range(100)]
    assert all(p.period_ms == PERIOD for p in pulses)
    # Pulse i reveals draw i, with the summary recorded when that value was drawn.
    for pulse, (value, summary) in zip(pulses, entropy.draws, strict=False):
        assert pulse.local_random == value
        assert pulse.summary == canonical_summary(summary)
    assert len(entropy.draws) == 101  # genesis draws two values


def test_genesis_reveals_one_value_and_keeps_the_next_pending(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    genesis = _service(memory_store, signing_key, entropy, clock).tick()
    assert genesis.index == 0
    assert genesis.prev_output == bytes(64)
    pending = memory_store.get_pending()
    assert pending is not None
    assert pending.index == 1
    assert pending.local_random == entropy.draws[1][0]
    assert pending.summary == canonical_summary(entropy.draws[1][1])


def test_tick_timestamp_is_the_slot_start(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    service = _service(memory_store, signing_key, entropy, clock)
    assert service.tick(VECTOR_T0_MS + 12_345).timestamp_ms == VECTOR_T0_MS


def test_one_pulse_per_slot(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    service = _service(memory_store, signing_key, entropy, clock)
    service.tick()
    draws = len(entropy.draws)
    with pytest.raises(BeaconTimingError):
        service.tick(clock.now_ms + PERIOD - 1)
    with pytest.raises(BeaconTimingError):
        service.tick(clock.now_ms - 10 * PERIOD)  # wall clock jumped backwards
    assert len(entropy.draws) == draws  # refused before drawing anything
    assert service.tick(clock.now_ms + PERIOD).index == 1


def test_downtime_skips_slots_but_keeps_indices_consecutive(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    service = _service(memory_store, signing_key, entropy, clock)
    service.tick()
    later = service.tick(clock.now_ms + 7 * PERIOD + 5)
    assert later.index == 1
    assert later.timestamp_ms == VECTOR_T0_MS + 7 * PERIOD
    _assert_honest(memory_store, service.public_key)


def test_next_tick_ms_is_aligned(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    service = _service(memory_store, signing_key, entropy, clock)
    assert service.next_tick_ms(VECTOR_T0_MS) == VECTOR_T0_MS
    assert service.next_tick_ms(VECTOR_T0_MS + 1) == VECTOR_T0_MS + PERIOD
    service.tick(VECTOR_T0_MS + 1)
    assert service.next_tick_ms(VECTOR_T0_MS + 2) == VECTOR_T0_MS + PERIOD
    assert service.next_tick_ms(VECTOR_T0_MS - 5 * PERIOD) == VECTOR_T0_MS + PERIOD
    assert service.next_tick_ms(VECTOR_T0_MS + 9 * PERIOD + 1) == VECTOR_T0_MS + 10 * PERIOD
    clock.now_ms = VECTOR_T0_MS + 3 * PERIOD - 1
    assert service.next_tick_ms() == VECTOR_T0_MS + 3 * PERIOD


def test_service_properties_and_wall_clock(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy
) -> None:
    service = BeaconService(memory_store, signing_key, entropy, period_ms=5_000)
    assert service.public_key == bytes(signing_key.verify_key)
    assert service.period_ms == 5_000
    assert service.store is memory_store
    before = time.time_ns() // 1_000_000
    assert before <= wall_clock_ms() <= time.time_ns() // 1_000_000


@pytest.mark.parametrize("period_ms", [0, -1, 2**32, 1.5])
def test_period_validation(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, period_ms: Any
) -> None:
    with pytest.raises(ValueError, match="period_ms"):
        BeaconService(memory_store, signing_key, entropy, period_ms=period_ms)


# --------------------------------------------------------------------------
# Restarts and refusing to publish
# --------------------------------------------------------------------------


def test_pending_survives_restart_and_the_chain_continues(
    tmp_path: Path, signing_key: SigningKey, rng: np.random.Generator, clock: FakeClock
) -> None:
    path = tmp_path / "beacon.db"
    first = FakeEntropy(rng)
    with BeaconStore(path) as store:
        service = _service(store, signing_key, first, clock)
        for _ in range(3):
            service.tick()
            clock.advance(PERIOD)
    second = FakeEntropy(np.random.default_rng(7))
    with BeaconStore(path) as store:
        service = _service(store, signing_key, second, clock)
        resumed = service.tick()
        assert resumed.index == 3
        assert resumed.local_random == first.draws[3][0]  # drawn before the restart
        assert resumed.summary == canonical_summary(first.draws[3][1])
        clock.advance(PERIOD)
        service.tick()
        assert len(_assert_honest(store, service.public_key)) == 5


def test_missing_pending_refuses_to_publish(
    tmp_path: Path, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        _service(store, signing_key, entropy, clock).tick()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DELETE FROM pending")
    with BeaconStore(path) as store:
        service = _service(store, signing_key, entropy, clock)
        with pytest.raises(BeaconStateError, match="cannot honour"):
            service.tick(clock.now_ms + PERIOD)
        assert store.count() == 1


def test_swapped_pending_refuses_to_publish(
    tmp_path: Path, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        _service(store, signing_key, entropy, clock).tick()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE pending SET local_random = ?", (bytes(64),))
    with BeaconStore(path) as store:
        service = _service(store, signing_key, entropy, clock)
        with pytest.raises(BeaconStateError, match="inconsistent"):
            service.tick(clock.now_ms + PERIOD)


def test_store_of_another_key_is_refused(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    _service(memory_store, signing_key, entropy, clock).tick()
    with pytest.raises(BeaconStateError, match="different beacon key"):
        _service(memory_store, SigningKey(FOREIGN_SEED), entropy, clock)


@pytest.mark.parametrize(
    "bad_output",
    [
        (b"short", {}),
        (bytearray(64), {}),
        (bytes(64), {"bits": 1.5}),
        (bytes(64), {"bits": None}),
    ],
)
def test_unusable_entropy_is_rejected_without_writing(
    memory_store: BeaconStore,
    signing_key: SigningKey,
    clock: FakeClock,
    bad_output: tuple[Any, dict[str, Any]],
) -> None:
    service = _service(memory_store, signing_key, lambda n, purpose: bad_output, clock)
    with pytest.raises(BeaconEntropyError):
        service.tick()
    assert memory_store.count() == 0


# --------------------------------------------------------------------------
# Crash injection
# --------------------------------------------------------------------------


class Crash(BaseException):
    """Simulated process death: not an Exception, so no handler swallows it."""


class Faults:
    """Arms one crash at a named point of the tick."""

    POINTS = ("draw", "sign", "commit", "after_commit")

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.armed: str | None = None
        real_create = service_module.create_pulse
        real_insert = store_module._insert_pending

        def create_pulse(*args: Any, **kwargs: Any) -> Pulse:
            self._maybe_crash("sign")
            return real_create(*args, **kwargs)

        def insert_pending(conn: sqlite3.Connection, pending: Any) -> None:
            # The pulse row is already INSERTed inside the open transaction.
            self._maybe_crash("commit")
            real_insert(conn, pending)

        monkeypatch.setattr(service_module, "create_pulse", create_pulse)
        monkeypatch.setattr(store_module, "_insert_pending", insert_pending)

    def entropy(self, inner: FakeEntropy) -> Callable[[int, str], tuple[bytes, Any]]:
        def draw(n: int, purpose: str) -> tuple[bytes, Any]:
            self._maybe_crash("draw")
            return inner(n, purpose)

        return draw

    def listener(self, pulse: Pulse) -> None:
        self._maybe_crash("after_commit")

    def _maybe_crash(self, point: str) -> None:
        if self.armed == point:
            self.armed = None
            raise Crash(point)


def test_crash_at_any_point_never_yields_an_invalid_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signing_key: SigningKey,
    rng: np.random.Generator,
    clock: FakeClock,
) -> None:
    path = tmp_path / "beacon.db"
    faults = Faults(monkeypatch)
    inner = FakeEntropy(rng)
    published: list[Pulse] = []
    crashes = dict.fromkeys(Faults.POINTS, 0)

    def boot() -> tuple[BeaconStore, BeaconService]:
        store = BeaconStore(path)
        service = _service(store, signing_key, faults.entropy(inner), clock)
        service.on_pulse(faults.listener)  # runs first: its crash loses the push below
        service.on_pulse(published.append)
        return store, service

    store, service = boot()
    for _ in range(120):
        if rng.random() < 0.4:
            faults.armed = Faults.POINTS[int(rng.integers(len(Faults.POINTS)))]
        point = faults.armed
        try:
            service.tick()
        except Crash:
            assert point is not None
            crashes[point] += 1
            store.close()  # the process died: drop every in-memory object
            store, service = boot()
            _assert_honest(store, service.public_key)
            latest = store.latest()
            slot_taken = latest is not None and latest.timestamp_ms >= clock.now_ms
            if slot_taken or rng.random() < 0.5:  # else retry within the same slot
                clock.advance(PERIOD)
        else:
            clock.advance(PERIOD)
    pulses = _assert_honest(store, service.public_key)
    store.close()

    assert all(count > 0 for count in crashes.values()), crashes
    # Everything announced is durable, and nothing durable was announced twice.
    stored = {p.index: p for p in pulses}
    assert all(stored[p.index] == p for p in published)
    assert len({p.index for p in published}) == len(published)
    # A crash after commit only loses the push, never the pulse.
    assert len(pulses) == len(published) + crashes["after_commit"]


def test_hard_kill_inside_the_commit_is_rolled_back(
    tmp_path: Path, signing_key: SigningKey, entropy: FakeEntropy, clock: FakeClock
) -> None:
    """A real process exit (no Python cleanup, hot journal left behind)."""
    path = tmp_path / "beacon.db"
    with BeaconStore(path) as store:
        service = _service(store, signing_key, entropy, clock)
        for _ in range(3):
            service.tick()
            clock.advance(PERIOD)
        before = store.latest()
        pending_before = store.get_pending()
    script = textwrap.dedent(
        f"""
        import os, sys
        import puffer.beacon.store as store_module
        from puffer.beacon.service import BeaconService
        from puffer.beacon.store import BeaconStore
        from puffer.beacon.vectors import vector_signing_key

        def die(conn, pending):
            conn.execute("SELECT 1")  # the pulse row is written but uncommitted
            os._exit(17)

        store_module._insert_pending = die
        service = BeaconService(
            BeaconStore({str(path)!r}),
            vector_signing_key(),
            lambda n, purpose: (os.urandom(n), {{"child": 1}}),
            clock_ms=lambda: {clock.now_ms},
        )
        service.tick()
        sys.exit(0)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=False)
    assert result.returncode == 17, result.stderr.decode()

    with BeaconStore(path) as store:
        assert store.latest() == before
        assert store.get_pending() == pending_before
        service = _service(store, signing_key, entropy, clock)
        service.tick()
        assert len(_assert_honest(store, service.public_key)) == 4


# --------------------------------------------------------------------------
# Listeners and the run loop
# --------------------------------------------------------------------------


def test_listeners_and_unsubscribe(
    memory_store: BeaconStore,
    signing_key: SigningKey,
    entropy: FakeEntropy,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _service(memory_store, signing_key, entropy, clock)
    seen: list[int] = []

    def broken(pulse: Pulse) -> None:
        raise RuntimeError("websocket hub down")

    service.on_pulse(broken)
    unsubscribe = service.on_pulse(lambda pulse: seen.append(pulse.index))
    with caplog.at_level(logging.ERROR, logger="puffer.beacon.service"):
        assert service.tick().index == 0
    assert "listener" in caplog.text
    clock.advance(PERIOD)
    unsubscribe()
    unsubscribe()  # idempotent
    service.tick()
    assert seen == [0]
    assert memory_store.count() == 2


def test_run_ticks_on_boundaries_and_survives_entropy_outages(
    memory_store: BeaconStore,
    signing_key: SigningKey,
    entropy: FakeEntropy,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock(now_ms=VECTOR_T0_MS + 1, step_ms=1)  # time passes on every read
    service = _service(memory_store, signing_key, entropy, clock, period_ms=4)
    stop = threading.Event()
    pulses: list[Pulse] = []

    def collect(pulse: Pulse) -> None:
        pulses.append(pulse)
        if len(pulses) == 3:
            entropy.fail_next = 1  # the next slot's draw fails once
        if len(pulses) == 10:
            stop.set()

    service.on_pulse(collect)
    with caplog.at_level(logging.ERROR, logger="puffer.beacon.service"):
        service.run(stop)
    assert "entropy unavailable" in caplog.text
    assert len(pulses) == 10
    assert all(p.timestamp_ms % 4 == 0 for p in pulses)
    assert all(b.timestamp_ms > a.timestamp_ms for a, b in itertools.pairwise(pulses))
    _assert_honest(memory_store, service.public_key)


def test_run_fails_closed_on_inconsistent_state(
    tmp_path: Path, signing_key: SigningKey, entropy: FakeEntropy
) -> None:
    path = tmp_path / "beacon.db"
    clock = FakeClock(step_ms=1)
    with BeaconStore(path) as store:
        _service(store, signing_key, entropy, clock, period_ms=4).tick()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DELETE FROM pending")
    with BeaconStore(path) as store:
        service = _service(store, signing_key, entropy, clock, period_ms=4)
        with pytest.raises(BeaconStateError):
            service.run(threading.Event())


def test_run_returns_when_stopped(
    memory_store: BeaconStore, signing_key: SigningKey, entropy: FakeEntropy
) -> None:
    service = _service(memory_store, signing_key, entropy, FakeClock(now_ms=VECTOR_T0_MS + 1))
    stop = threading.Event()
    worker = threading.Thread(target=service.run, args=(stop,))
    worker.start()
    stop.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert memory_store.count() == 0
