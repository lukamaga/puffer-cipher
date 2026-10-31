"""PufferRNG: fish-seeded random bytes that are never weaker than the OS.

Construction (DESIGN.md section 4.4)
------------------------------------
For every request::

    draw  = pool.draw()                       # channel digests, credit consumed
    K     = os.urandom(32)                    # fresh for every request
    seed  = BLAKE2b-256(key=K, person=b"puffer-seed-v1",
                        data = u32(len(purpose)) || purpose || u64(counter) || draw.material)
    out   = randombytes_deterministic(n, seed)    # libsodium ChaCha20

Why the output can never be weaker than ``os.urandom``
    Keyed BLAKE2b is a PRF. With a key ``K`` the adversary does not know,
    ``seed`` is indistinguishable from uniform *for every possible data
    input* - including pool material the adversary fully controls (a
    replayed video, a camera swapped for an attacker's feed, dead fish).
    ``out`` is then ChaCha20 keystream under a uniform key. So the fish can
    only add entropy; they can never subtract it.

Why the fish still help if the OS RNG were weak
    If ``K`` were predictable (early-boot starvation, a VM snapshot that
    replays its RNG state, a backdoored generator), ``seed`` is still
    BLAKE2b of the pool material, which an attacker who cannot see the tank
    cannot predict. BLAKE2b is then acting as an entropy extractor: the
    seed keeps up to 256 bits of the camera's unpredictability. The ``u64``
    counter and the pool ratchet additionally guarantee that two requests
    never derive the same seed even if ``K`` repeats.

The purpose string is length-prefixed so different purposes can never
produce colliding inputs, and binds each output to its use ("key",
"nonce", "salt", "beacon") as domain separation.

Requests larger than :data:`CHUNK_BYTES` are split into chunks with
consecutive counter values (one seed per chunk): libsodium aborts the
whole process (``sodium_misuse``) on oversized deterministic requests, and
small chunks keep every call far inside its limits on every platform.

Display bits
    :meth:`PufferRNG.display_bits` drives the dashboard's scrolling bits. It
    is *public*: derived from :meth:`EntropyPool.snapshot_digest
    <puffer.entropy.pool.EntropyPool.snapshot_digest>` with no OS key,
    consumes no credit, and must never be used for anything secret.
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
import struct
import threading
import time
from collections.abc import Callable

import nacl.utils

from puffer.entropy.pool import EntropyPool
from puffer.entropy.receipt import EntropyReceipt, ReceiptContext

__all__ = ["CHUNK_BYTES", "SYSTEM_KEY_BYTES", "PufferRNG", "derive_seed"]

SYSTEM_KEY_BYTES = 32
SEED_BYTES = 32
CHUNK_BYTES = 1 << 20
_SEED_PERSON = b"puffer-seed-v1"
_DISPLAY_PERSON = b"puffer-display"
_DISPLAY_BLOCK_BITS = 512
_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")


def derive_seed(system_key: bytes, purpose: bytes, counter: int, material: bytes) -> bytes:
    """The 32-byte ChaCha20 seed for one output chunk (DESIGN.md section 4.4)."""
    if len(system_key) != SYSTEM_KEY_BYTES:
        raise ValueError(f"system key must be {SYSTEM_KEY_BYTES} bytes")
    h = hashlib.blake2b(digest_size=SEED_BYTES, key=system_key, person=_SEED_PERSON)
    h.update(_U32.pack(len(purpose)))
    h.update(purpose)
    h.update(_U64.pack(counter))
    h.update(material)
    return h.digest()


class PufferRNG:
    """Random bytes + receipts from an :class:`EntropyPool` keyed with OS randomness.

    ``system_random`` and ``clock`` are injectable for deterministic tests
    (never replace ``system_random`` in production). ``context`` supplies
    the engine's mode/liveness/health/fish names for receipts. ``pump``, if
    given, is called while :meth:`wait_for_credit` waits, so a
    single-threaded caller (the CLI with a file source) can keep feeding
    frames into the pool while it waits. Thread-safe.
    """

    def __init__(
        self,
        pool: EntropyPool,
        *,
        system_random: Callable[[int], bytes] = os.urandom,
        clock: Callable[[], float] = time.time,
        context: Callable[[], ReceiptContext] | None = None,
        pump: Callable[[], None] | None = None,
    ) -> None:
        self.pool = pool
        self._system_random = system_random
        self._clock = clock
        self._context = context
        self._pump = pump
        self._lock = threading.Lock()
        self._counter = 0
        self._display_counter = 0

    def generate(self, n: int, purpose: str = "bytes") -> tuple[bytes, EntropyReceipt]:
        """``n`` secret random bytes for ``purpose``, and the receipt describing them."""
        n = operator.index(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        purpose_bytes = purpose.encode("utf-8")
        if len(purpose_bytes) > 0xFFFFFFFF:
            raise ValueError("purpose too long")
        n_chunks = max(1, math.ceil(n / CHUNK_BYTES))
        with self._lock:
            first = self._counter
            self._counter += n_chunks
        draw = self.pool.draw()
        key = self._system_random(SYSTEM_KEY_BYTES)
        if len(key) != SYSTEM_KEY_BYTES:
            raise RuntimeError("system_random returned the wrong number of bytes")
        out = b"".join(
            nacl.utils.randombytes_deterministic(
                min(CHUNK_BYTES, n - i * CHUNK_BYTES),
                derive_seed(key, purpose_bytes, first + i, draw.material),
            )
            for i in range(n_chunks)
        )
        receipt = EntropyReceipt.from_draw(
            draw,
            purpose=purpose,
            n_bytes=n,
            timestamp_ms=int(self._clock() * 1000),
            context=self._context() if self._context is not None else ReceiptContext(),
        )
        return out, receipt

    def random_bytes(self, n: int) -> bytes:
        """``n`` secret random bytes (receipt discarded)."""
        return self.generate(n, "bytes")[0]

    def generate_key(self) -> tuple[bytes, EntropyReceipt]:
        """A 32-byte secret key and its receipt."""
        return self.generate(32, "key")

    def nonce(self) -> bytes:
        """A 24-byte XChaCha20 nonce (random nonces are safe at 192 bits)."""
        return self.generate(24, "nonce")[0]

    def display_bits(self, n_bits: int) -> str:
        """``n_bits`` *public* bits as a ``'0'``/``'1'`` string, for display only.

        ``BLAKE2b-512(snapshot_digest || u64(counter), person=b"puffer-display")``
        per 512-bit block. No credit is consumed and no OS key is involved,
        so the stream is reproducible from the pool state: it is fine to
        show, and never fine to use as a secret.
        """
        n_bits = operator.index(n_bits)
        if n_bits < 0:
            raise ValueError("n_bits must be non-negative")
        n_blocks = math.ceil(n_bits / _DISPLAY_BLOCK_BITS)
        snapshot = self.pool.snapshot_digest()
        with self._lock:
            first = self._display_counter
            self._display_counter += n_blocks
        data = b"".join(
            hashlib.blake2b(
                snapshot + _U64.pack(first + i), digest_size=64, person=_DISPLAY_PERSON
            ).digest()
            for i in range(n_blocks)
        )
        return "".join(f"{byte:08b}" for byte in data)[:n_bits]

    def wait_for_credit(self, bits: float, timeout: float | None, poll_s: float = 0.05) -> bool:
        """Wait until the camera channels hold ``bits`` of credit; False on timeout.

        Wakes on every pool change (condition variable) and at least every
        ``poll_s`` seconds to run the ``pump`` hook. ``timeout=None`` waits
        forever.
        """
        if poll_s <= 0:
            raise ValueError("poll_s must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._pump is not None:
                self._pump()
            remaining = math.inf if deadline is None else deadline - time.monotonic()
            if self.pool.wait_for_credit(bits, timeout=max(0.0, min(poll_s, remaining))):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
