"""Conditioned test streams - ANALYSIS ONLY, NEVER A SOURCE OF SECRETS.

!!! Do not use this module to generate keys, nonces or anything secret. !!!

:class:`ConditionedStream` compresses a channel's raw samples into output
*without* any OS randomness, so that reports can compare "fish only",
"sensor only" and "combined" streams with statistical test suites
(DESIGN.md section 4.6). Production output always goes through
:class:`~puffer.entropy.rng.PufferRNG`, which keys everything with
``os.urandom``; that is exactly what makes production output useless for
judging the camera, and why this separate, unkeyed path exists.

Even here, hashed output passing SP 800-22 proves little about entropy:
BLAKE2b output of a counter passes too. The interesting comparisons are
between conditioning *rates* - a block is compressed only once its
*credited* min-entropy (samples x ``h_per_sample``) reaches
``block_entropy_bits`` - and the raw-sample estimators in
:mod:`puffer.entropy.estimators`.

Block accounting: a block of ``ceil(block_entropy_bits / h_per_sample)``
samples becomes ``BLAKE2b(block, digest_size=out_bytes,
person=b"puffer-test-cond")``. Output never claims more entropy than went
in (``8 * out_bytes <= block_entropy_bits``). Samples that do not complete
a block are kept for the next :meth:`ConditionedStream.feed`; they are never
padded and never emitted early, so every output byte is backed by a full
block.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import numpy.typing as npt

__all__ = ["ConditionedStream"]

_PERSON = b"puffer-test-cond"
_MAX_BITS_PER_SYMBOL = 8.0  # samples are uint8


class ConditionedStream:
    """Unkeyed BLAKE2b compression of raw samples for analysis reports (not secrets)."""

    def __init__(
        self, h_per_sample: float, block_entropy_bits: int = 512, out_bytes: int = 32
    ) -> None:
        if not (math.isfinite(h_per_sample) and 0 < h_per_sample <= _MAX_BITS_PER_SYMBOL):
            raise ValueError("h_per_sample must be in (0, 8] bits per uint8 sample")
        if not 1 <= out_bytes <= 64:
            raise ValueError("out_bytes must be in 1..64 (BLAKE2b digest size)")
        if 8 * out_bytes > block_entropy_bits:
            raise ValueError("a block must carry at least as much entropy as its output")
        self.h_per_sample = h_per_sample
        self.block_entropy_bits = block_entropy_bits
        self.out_bytes = out_bytes
        self.block_samples = math.ceil(block_entropy_bits / h_per_sample)
        self.samples_in = 0
        self.blocks_out = 0
        self._pending = np.empty(0, dtype=np.uint8)

    @property
    def pending_samples(self) -> int:
        """Samples waiting for their block to fill up."""
        return int(self._pending.size)

    def feed(self, samples: npt.ArrayLike) -> bytes:
        """Add samples; return the output of every block they completed (maybe ``b""``)."""
        s = np.asarray(samples).ravel()
        if s.size and (s.dtype.kind not in "iu" or int(s.min()) < 0 or int(s.max()) > 255):
            raise ValueError("samples must be integer symbols in 0..255")
        self.samples_in += int(s.size)
        buf = np.concatenate((self._pending, s.astype(np.uint8)))
        n_blocks = buf.size // self.block_samples
        out = b"".join(
            hashlib.blake2b(
                buf[i * self.block_samples : (i + 1) * self.block_samples].tobytes(),
                digest_size=self.out_bytes,
                person=_PERSON,
            ).digest()
            for i in range(n_blocks)
        )
        self._pending = buf[n_blocks * self.block_samples :].copy()
        self.blocks_out += n_blocks
        return out
