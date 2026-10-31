"""The beacon scheduler: one signed pulse per period, crash-consistently.

Each tick reveals the ``local_random`` that the previous pulse precommitted
to, draws the *next* ``local_random`` from the entropy source (PufferRNG in
production), precommits to it, signs, and stores the pulse together with the
new pending reveal. The genesis tick draws two values: ``local_random_0``,
revealed immediately, and ``local_random_1``, kept pending.

Crash consistency
-----------------
The danger is publishing a pulse whose precommitment can never be honoured:
if pulse *n* were stored but the value it commits to were lost, pulse *n+1*
could only reveal a value that fails verification, and the chain would be
broken forever. The tick is therefore built so that every crash point leaves
a state the next tick can continue from honestly:

1. *Before or during the entropy draw, and while building and signing the
   pulse:* nothing has been written. The freshly drawn value never left the
   process and its hash was never published, so it is simply discarded; the
   restarted service reveals the same stored pending value and draws a new
   next value.
2. *During the commit:* :meth:`BeaconStore.append` writes the pulse and the
   new pending reveal in one SQLite transaction, so either both are durable
   or neither is (back to case 1).
3. *After the commit, before listeners run:* the pulse is durable and served
   by the REST API; only the WebSocket push is missed. The restarted service
   continues from the stored pending reveal.

Listeners are notified only after the commit returns, and the store commits
with ``synchronous = FULL``, so nothing is ever pushed to the public that a
power cut could take back. Every tick re-checks the invariant "the latest
pulse's precommitment is matched by the stored pending reveal" and refuses
to publish (:class:`BeaconStateError`) rather than paper over a broken
state, e.g. a pulses table whose pending reveal was lost or swapped.
Recovering from that is an explicit operator decision: start a new chain
(new database), leaving the old chain visibly ended at an unrevealed
precommitment.

**Never restore a database backup.** A consistent snapshot taken after pulse
*k* satisfies that invariant, so a service restarted on it would sign a
second, different pulse *k+1* - a fork of the public chain that every
verifier accepts. Nothing local can tell it from ordinary downtime; after
losing a database, start a new chain.

Timing
------
Pulse timestamps name period slots: ``timestamp_ms`` is the start of the
period containing the tick time (``now - now % period_ms``), and at most one
pulse is issued per slot. After downtime the next pulse takes the current
slot; indices stay consecutive while timestamps skip the missed slots.
:meth:`BeaconService.run` sleeps until the next slot boundary.

One producer per store: :class:`BeaconService` serializes its own ticks, and
the store's transactional link check makes a second producer on the same
database fail instead of forking the chain.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, TypeAlias

from nacl.signing import SigningKey

from puffer.beacon.pulse import (
    GENESIS_PREV_OUTPUT,
    LOCAL_RANDOM_BYTES,
    MAX_PERIOD_MS,
    Pulse,
    canonical_summary,
    create_pulse,
)
from puffer.beacon.store import (
    BeaconStore,
    PendingMismatchError,
    PendingReveal,
    check_pending_matches,
)

logger = logging.getLogger(__name__)

EntropySource: TypeAlias = Callable[[int, str], tuple[bytes, Mapping[str, Any]]]
"""``entropy(n, purpose) -> (n random bytes, beacon summary dict for them)``."""

PulseListener: TypeAlias = Callable[[Pulse], None]

ENTROPY_PURPOSE: Final = "beacon"
"""Purpose string passed to the entropy source (and thus to PufferRNG receipts)."""

_MAX_SLEEP_MS: Final = 1_000  # re-read the clock at least this often (wall-clock jumps)


class BeaconStateError(RuntimeError):
    """The stored chain cannot be continued honestly; the service refuses to publish."""


class BeaconTimingError(RuntimeError):
    """A pulse was requested for a period slot that already has one (or an earlier one)."""


class BeaconEntropyError(RuntimeError):
    """The entropy source failed or returned unusable data; nothing was written."""


def wall_clock_ms() -> int:
    """Unix epoch time in milliseconds (the beacon's notion of time)."""
    return time.time_ns() // 1_000_000


class BeaconService:
    """Produces the pulse chain stored in ``store``, signed with ``signing_key``.

    ``entropy`` must return fresh, secret, full-entropy bytes (PufferRNG) plus
    the summary describing them; beacon values are public, so the beacon draws
    its own randomness like any other consumer and nothing it publishes may
    ever be used as a key. ``clock_ms`` is injectable for tests.

    Raises:
        ValueError: ``period_ms`` is out of range.
        BeaconStateError: ``store`` holds a chain signed by a different key.
    """

    def __init__(
        self,
        store: BeaconStore,
        signing_key: SigningKey,
        entropy: EntropySource,
        *,
        period_ms: int = 60_000,
        clock_ms: Callable[[], int] = wall_clock_ms,
    ) -> None:
        if type(period_ms) is not int or not 1 <= period_ms <= MAX_PERIOD_MS:
            raise ValueError(f"period_ms must be an int in [1, {MAX_PERIOD_MS}]")
        self._store = store
        self._signing_key = signing_key
        self._public_key = bytes(signing_key.verify_key)
        self._entropy = entropy
        self._period_ms = period_ms
        self._clock_ms = clock_ms
        self._tick_lock = threading.Lock()
        self._listeners: list[PulseListener] = []
        self._listeners_lock = threading.Lock()
        latest = store.latest()
        if latest is not None and latest.public_key != self._public_key:
            raise BeaconStateError("the stored chain was signed by a different beacon key")

    @property
    def public_key(self) -> bytes:
        """The beacon's Ed25519 public key (32 bytes), published at ``/api/beacon/public-key``."""
        return self._public_key

    @property
    def period_ms(self) -> int:
        """Pulse period in milliseconds."""
        return self._period_ms

    @property
    def store(self) -> BeaconStore:
        """The store this service appends to (read it for the public chain)."""
        return self._store

    # -- listeners ----------------------------------------------------------

    def on_pulse(self, listener: PulseListener) -> Callable[[], None]:
        """Call ``listener(pulse)`` after each committed pulse; returns an unsubscribe function.

        Listeners run on the ticking thread (the :meth:`run` thread), after the
        pulse is durable. An asyncio consumer such as the WebSocket hub must
        hand the pulse over with ``loop.call_soon_threadsafe``. A listener that
        raises is logged and does not affect the beacon or other listeners.
        """
        with self._listeners_lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._listeners_lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    # -- scheduling ---------------------------------------------------------

    def slot_start(self, t_ms: int) -> int:
        """Start of the period slot containing ``t_ms``."""
        return t_ms - t_ms % self._period_ms

    def next_tick_ms(self, now_ms: int | None = None) -> int:
        """Earliest time at which :meth:`tick` will issue the next pulse.

        The first period boundary that is not before ``now_ms`` and lies in a
        slot after the latest pulse's slot.
        """
        now = self._clock_ms() if now_ms is None else now_ms
        due = -(-now // self._period_ms) * self._period_ms  # ceil to a boundary
        latest = self._store.latest()
        if latest is not None:
            due = max(due, self.slot_start(latest.timestamp_ms) + self._period_ms)
        return due

    def tick(self, now_ms: int | None = None) -> Pulse:
        """Issue, store and announce the pulse for the slot containing ``now_ms``.

        Raises:
            BeaconTimingError: the slot is not after the latest pulse's timestamp.
            BeaconStateError: the stored pending reveal is missing or wrong.
            BeaconEntropyError: the entropy source failed (safe to retry).
        """
        with self._tick_lock:
            now = self._clock_ms() if now_ms is None else now_ms
            timestamp_ms = self.slot_start(now)
            latest = self._store.latest()
            if latest is None:
                index, prev_output = 0, GENESIS_PREV_OUTPUT
                local_random, summary = self._draw()
            else:
                if timestamp_ms <= latest.timestamp_ms:
                    raise BeaconTimingError(
                        f"slot {timestamp_ms} is not after the latest pulse ({latest.timestamp_ms})"
                    )
                reveal = self._pending_for(latest)
                index, prev_output = latest.index + 1, latest.output
                local_random, summary = reveal.local_random, reveal.summary
            next_random, next_summary = self._draw()
            pulse = create_pulse(
                self._signing_key,
                index=index,
                timestamp_ms=timestamp_ms,
                period_ms=self._period_ms,
                local_random=local_random,
                next_local_random=next_random,
                prev_output=prev_output,
                summary=summary,
            )
            self._store.append(pulse, pending=PendingReveal(index + 1, next_random, next_summary))
        logger.info("beacon pulse %d issued for slot %d", pulse.index, pulse.timestamp_ms)
        self._notify(pulse)
        return pulse

    def run(self, stop_event: threading.Event) -> None:
        """Tick at every period boundary until ``stop_event`` is set.

        An entropy failure skips the current slot (it is logged and retried
        at the next boundary, which is safe because nothing was written).
        Any other error - an inconsistent store above all - propagates and
        stops the loop: failing closed is the only honest option for a beacon.
        """
        retry_at: int | None = None
        while not stop_event.is_set():
            now = self._clock_ms()
            due = self.next_tick_ms(now)
            if retry_at is not None:
                due = max(due, retry_at)
            if now < due:
                stop_event.wait(min(due - now, _MAX_SLEEP_MS) / 1000)
                continue
            try:
                self.tick(now)
            except BeaconEntropyError:
                logger.exception("beacon entropy unavailable; skipping slot %d", due)
                retry_at = self.slot_start(now) + self._period_ms
            else:
                retry_at = None

    # -- internals ----------------------------------------------------------

    def _pending_for(self, latest: Pulse) -> PendingReveal:
        pending = self._store.get_pending()
        if pending is None:
            raise BeaconStateError(
                f"pulse {latest.index} precommitted to a value that is not in the store; "
                "refusing to publish a pulse that cannot honour it"
            )
        try:
            check_pending_matches(latest, pending)
        except PendingMismatchError as exc:
            raise BeaconStateError(f"stored pending reveal is inconsistent: {exc}") from exc
        return pending

    def _draw(self) -> tuple[bytes, str]:
        """Draw a fresh ``local_random`` and its canonical summary, or raise BeaconEntropyError."""
        try:
            value, summary = self._entropy(LOCAL_RANDOM_BYTES, ENTROPY_PURPOSE)
            if type(value) is not bytes or len(value) != LOCAL_RANDOM_BYTES:
                raise TypeError(f"entropy source must return {LOCAL_RANDOM_BYTES} bytes")
            return value, canonical_summary(summary)
        except Exception as exc:
            raise BeaconEntropyError(f"entropy draw failed: {exc}") from exc

    def _notify(self, pulse: Pulse) -> None:
        with self._listeners_lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(pulse)
            except Exception:
                logger.exception("beacon pulse listener %r failed", listener)
