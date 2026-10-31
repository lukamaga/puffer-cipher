"""Entropy pool: per-channel BLAKE2b accumulators plus a credit ledger.

Design (DESIGN.md section 4.4)
------------------------------
* **One accumulator per channel.** Each camera channel hashes its ``absorb``
  bytes into its own running BLAKE2b-512 state
  (``person = b"puffer-pool-" + name``, truncated to BLAKE2b's 16-byte
  limit). Keeping channels separate means a channel that turns out to be
  attacker-controlled cannot cancel or overwrite another channel's
  contribution, and receipts can attribute credit honestly. Every absorb is
  framed with its u64 length so the absorbed sequence is encoded
  injectively.
* **Credit is a ledger, not a property of the hash.** ``absorb`` adds the
  caller-computed credit (from the channel's *samples*, see
  :mod:`puffer.entropy.channels`) and saturates per channel at
  ``capacity_bits``. A 512-bit digest can never carry more than 512 bits,
  and the RNG compresses everything to a 256-bit seed, so credit is a record
  of conservatively estimated *input* entropy, not a claim about output
  strength (output strength is guaranteed by OS randomness, see
  :mod:`puffer.entropy.rng`).
* **Draws ratchet.** :meth:`EntropyPool.draw` returns the digests and then
  re-seeds each accumulator with ``BLAKE2b-512(prev_digest,
  person=b"puffer-ratchet")``. Entropy that was absorbed but not yet
  credited carries over into future draws, while the new state is a one-way
  function of the old one: compromising the pool later does not reveal
  material handed out earlier (backtracking resistance).
* **Thread-safe.** The engine absorbs from its capture thread while API
  handlers draw. One lock guards all state, and a condition variable on the
  same lock wakes :meth:`EntropyPool.wait_for_credit` callers on every
  change.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from puffer.types import CAMERA_CHANNELS, ChannelOutput

__all__ = ["DIGEST_BYTES", "EntropyPool", "PoolDraw", "pool_person"]

DIGEST_BYTES = 64
_POOL_PREFIX = b"puffer-pool-"
_RATCHET_PERSON = b"puffer-ratchet"
_SNAPSHOT_PERSON = b"puffer-snapshot"
_PERSON_MAX = 16  # BLAKE2b personalization limit


def pool_person(channel: str) -> bytes:
    """Personalization of a channel accumulator: ``b"puffer-pool-" + name``, max 16 bytes."""
    return (_POOL_PREFIX + channel.encode("utf-8"))[:_PERSON_MAX]


def _new_accumulator(channel: str) -> hashlib.blake2b:
    return hashlib.blake2b(digest_size=DIGEST_BYTES, person=pool_person(channel))


@dataclass(frozen=True, slots=True)
class PoolDraw:
    """Everything one draw consumed from the pool.

    ``material`` is the concatenation of the channel digests in sorted
    channel order (``channels``); it is secret-ish (never published) and is
    only ever used as data inside the OS-keyed seed derivation. ``credits``
    and ``fish_credits`` are the credited bits consumed by this draw;
    ``frames`` is the ``(first, last)`` frame index absorbed since the
    previous draw, or ``None`` if no frame was absorbed.
    """

    draw_index: int
    material: bytes
    channels: tuple[str, ...]
    credits: dict[str, float]
    fish_credits: dict[str, float]
    frames: tuple[int, int] | None

    @property
    def camera_bits(self) -> float:
        """Credited bits from the camera channels (sensor + fish)."""
        return sum(self.credits.get(name, 0.0) for name in CAMERA_CHANNELS)


class EntropyPool:
    """Per-channel BLAKE2b-512 accumulators with saturating credit counters."""

    def __init__(self, capacity_bits: float = 4096.0) -> None:
        if not (math.isfinite(capacity_bits) and capacity_bits > 0):
            raise ValueError("capacity_bits must be a positive number")
        self.capacity_bits = float(capacity_bits)
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._acc: dict[str, hashlib.blake2b] = {}
        self._credit: dict[str, float] = {}
        self._fish: dict[str, float] = {}
        self._frames: tuple[int, int] | None = None
        self._draw_index = 0

    @property
    def draw_index(self) -> int:
        """Index the next draw will carry (number of draws so far)."""
        with self._lock:
            return self._draw_index

    def absorb(self, output: ChannelOutput, credit_bits: float) -> None:
        """Hash ``output.absorb`` into its channel and add ``credit_bits`` of credit.

        Credit beyond ``capacity_bits`` is dropped (the bytes are still
        absorbed). The credit actually added is split between fish in
        proportion to how many of the output's samples each fish owns;
        samples with an unknown owner leave their share unattributed.
        """
        if not (math.isfinite(credit_bits) and credit_bits >= 0):
            raise ValueError(f"credit_bits must be finite and non-negative, got {credit_bits!r}")
        owners = output.sample_owner
        n_samples = int(output.samples.size)
        if owners is not None and len(owners) != n_samples:
            raise ValueError("sample_owner must have one entry per sample")
        name = output.channel
        with self._changed:
            acc = self._acc.get(name)
            if acc is None:
                acc = self._acc[name] = _new_accumulator(name)
            acc.update(len(output.absorb).to_bytes(8, "big"))
            acc.update(output.absorb)
            before = self._credit.get(name, 0.0)
            after = min(self.capacity_bits, before + credit_bits)
            self._credit[name] = after
            added = after - before
            if owners is not None and added > 0 and n_samples:
                for fish_id, count in Counter(o for o in owners if o is not None).items():
                    self._fish[fish_id] = self._fish.get(fish_id, 0.0) + added * count / n_samples
            index = output.frame_index
            first, last = self._frames if self._frames is not None else (index, index)
            self._frames = (min(first, index), max(last, index))
            self._changed.notify_all()

    def credits(self) -> dict[str, float]:
        """Credited bits per channel since the last draw."""
        with self._lock:
            return dict(self._credit)

    def fish_credits(self) -> dict[str, float]:
        """Credited bits per fish id since the last draw."""
        with self._lock:
            return dict(self._fish)

    def camera_credit(self) -> float:
        """Credited bits from the camera channels since the last draw."""
        with self._lock:
            return self._sum_credit(CAMERA_CHANNELS)

    def draw(self) -> PoolDraw:
        """Consume all credit, return the channel digests and ratchet every accumulator."""
        with self._changed:
            channels = tuple(sorted(self._acc))
            digests = [self._acc[name].digest() for name in channels]
            for name, digest in zip(channels, digests, strict=True):
                seed = hashlib.blake2b(
                    digest, digest_size=DIGEST_BYTES, person=_RATCHET_PERSON
                ).digest()
                acc = _new_accumulator(name)
                acc.update(seed)
                self._acc[name] = acc
            result = PoolDraw(
                draw_index=self._draw_index,
                material=b"".join(digests),
                channels=channels,
                credits=dict(self._credit),
                fish_credits=dict(self._fish),
                frames=self._frames,
            )
            self._credit = dict.fromkeys(self._credit, 0.0)
            self._fish = {}
            self._frames = None
            self._draw_index += 1
            self._changed.notify_all()
            return result

    def snapshot_digest(self) -> bytes:
        """A 64-byte digest of the current pool state; consumes no credit.

        Domain-separated (``person=b"puffer-snapshot"``) and one-way, so a
        published function of it (the dashboard's display bits) reveals
        neither the accumulators nor future draws' material.
        """
        with self._lock:
            h = hashlib.blake2b(digest_size=DIGEST_BYTES, person=_SNAPSHOT_PERSON)
            h.update(self._draw_index.to_bytes(8, "big"))
            for name in sorted(self._acc):
                encoded = name.encode("utf-8")
                h.update(len(encoded).to_bytes(4, "big") + encoded)
                h.update(self._acc[name].digest())
            return h.digest()

    def wait_for_credit(
        self,
        bits: float,
        timeout: float | None,
        channels: Iterable[str] = CAMERA_CHANNELS,
    ) -> bool:
        """Block until ``channels`` hold at least ``bits`` of credit; False on timeout."""
        names = tuple(channels)
        with self._changed:
            return self._changed.wait_for(lambda: self._sum_credit(names) >= bits, timeout)

    def _sum_credit(self, names: Iterable[str]) -> float:
        return sum(self._credit.get(name, 0.0) for name in names)
