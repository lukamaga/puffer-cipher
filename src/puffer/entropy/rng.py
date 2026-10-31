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

Public values (``consume=False``)
    Nonces and salts are published next to the ciphertext: they must be
    unique and unpredictable, never secret. Spending camera credit on them
    would be pure loss - a draw resets *all* credit, so a key generated
    right after a nonce would get almost no fish entropy. Public values
    therefore use::

        seed = BLAKE2b-256(key=K, person=b"puffer-public-v1",
                           data = u32(len(purpose)) || purpose || u64(counter)
                                  || pool.snapshot_digest())

    ``K`` is still a fresh ``os.urandom(32)``, so public output keeps the
    "never weaker than the OS" guarantee; the snapshot still mixes in
    the pool (if the OS RNG were stuck, values keep changing with the
    tank); the counter is shared with the consuming path, and the separate
    personalisation means a public seed can never equal a secret one. The
    snapshot is one-way and already the input of the public display stream,
    so nothing that later keys depend on is revealed, and nothing is
    consumed: the receipt reports zero camera credit.

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

from puffer.entropy.pool import EntropyPool, PoolDraw
from puffer.entropy.receipt import EntropyReceipt, ReceiptContext

__all__ = ["CHUNK_BYTES", "SYSTEM_KEY_BYTES", "PufferRNG", "derive_seed"]

SYSTEM_KEY_BYTES = 32
SEED_BYTES = 32
CHUNK_BYTES = 1 << 20
_SEED_PERSON = b"puffer-seed-v1"
_PUBLIC_PERSON = b"puffer-public-v1"
_NONCE_BYTES = 24  # crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
_SALT_BYTES = 16  # crypto_pwhash_SALTBYTES
_DISPLAY_PERSON = b"puffer-display"
_DISPLAY_BLOCK_BITS = 512
_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")


def derive_seed(
    system_key: bytes,
    purpose: bytes,
    counter: int,
    material: bytes,
    *,
    person: bytes = _SEED_PERSON,
) -> bytes:
    """The 32-byte ChaCha20 seed for one output chunk (DESIGN.md section 4.4).

    ``material`` is the pool draw's material for secret output; public
    output passes the pool snapshot and ``person=b"puffer-public-v1"``.
    """
    if len(system_key) != SYSTEM_KEY_BYTES:
        raise ValueError(f"system key must be {SYSTEM_KEY_BYTES} bytes")
    h = hashlib.blake2b(digest_size=SEED_BYTES, key=system_key, person=person)
    h.update(_U32.pack(len(purpose)))
    h.update(purpose)
    h.update(_U64.pack(counter))
    h.update(material)
    return h.digest()


def _nothing_drawn(draw_index: int) -> PoolDraw:
    """The (empty) pool draw behind a public output: no credit, no frames."""
    return PoolDraw(
        draw_index=draw_index,
        material=b"",
        channels=(),
        credits={},
        fish_credits={},
        frames=None,
    )


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

    def generate(
        self, n: int, purpose: str = "bytes", *, consume: bool = True
    ) -> tuple[bytes, EntropyReceipt]:
        """``n`` random bytes for ``purpose``, and the receipt describing them.

        ``consume=True`` (the default, for keys and anything secret) draws
        the pool: its material goes into the seed and all camera credit is
        consumed and reported. ``consume=False`` is for *public* values
        (nonces, salts): the seed mixes the pool snapshot instead, no credit
        is consumed and the receipt reports zero camera credit (see the
        module docstring). Both are keyed with fresh OS randomness.
        """
        n = operator.index(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        purpose_bytes = purpose.encode("utf-8")
        if len(purpose_bytes) > 0xFFFFFFFF:
            raise ValueError("purpose too long")
        n_chunks = max(1, math.ceil(n / CHUNK_BYTES))
        # The OS key comes first: if it fails, no camera credit is thrown away.
        key = self._system_random(SYSTEM_KEY_BYTES)
        if len(key) != SYSTEM_KEY_BYTES:
            raise RuntimeError("system_random returned the wrong number of bytes")
        with self._lock:
            first = self._counter
            self._counter += n_chunks
        if consume:
            draw = self.pool.draw()
            material, person = draw.material, _SEED_PERSON
        else:
            draw = _nothing_drawn(self.pool.draw_index)
            material, person = self.pool.snapshot_digest(), _PUBLIC_PERSON
        out = b"".join(
            nacl.utils.randombytes_deterministic(
                min(CHUNK_BYTES, n - i * CHUNK_BYTES),
                derive_seed(key, purpose_bytes, first + i, material, person=person),
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

    def nonce(self, n: int = _NONCE_BYTES) -> bytes:
        """A 24-byte XChaCha20 nonce (random nonces are safe at 192 bits).

        Public: OS-keyed, consumes no camera credit (``consume=False``).
        ``n`` lets it serve as the ``random_bytes`` hook of
        :func:`puffer.crypto.encrypt_message`.
        """
        return self.generate(n, "nonce", consume=False)[0]

    def salt(self, n: int = _SALT_BYTES) -> bytes:
        """``n`` bytes of public salt (default: an Argon2id salt).

        OS-keyed, consumes no camera credit (``consume=False``). Matches the
        ``RandomBytes`` signature, so ``random_bytes=rng.salt`` can be passed
        to :func:`puffer.crypto.encrypt_file`.
        """
        return self.generate(n, "salt", consume=False)[0]

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
