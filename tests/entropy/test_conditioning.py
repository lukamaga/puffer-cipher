"""Conditioned analysis streams: block accounting, exact output, batching."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from puffer.entropy.conditioning import ConditionedStream


def reference(block: np.ndarray, out_bytes: int = 32) -> bytes:
    return hashlib.blake2b(
        block.astype(np.uint8).tobytes(), digest_size=out_bytes, person=b"puffer-test-cond"
    ).digest()


def test_block_accounting_keeps_incomplete_blocks(rng):
    stream = ConditionedStream(h_per_sample=2.0)
    assert stream.block_samples == 256  # 512 bits / 2 bits per sample
    data = rng.integers(0, 256, size=256 * 3 + 10, dtype=np.uint8)
    assert stream.feed(data[:255]) == b""
    assert stream.pending_samples == 255
    first = stream.feed(data[255:256])
    assert first == reference(data[:256])
    assert stream.pending_samples == 0
    rest = stream.feed(data[256:])
    assert rest == reference(data[256:512]) + reference(data[512:768])
    assert (stream.pending_samples, stream.blocks_out, stream.samples_in) == (10, 3, data.size)


def test_fractional_entropy_rounds_block_size_up():
    assert ConditionedStream(h_per_sample=0.3).block_samples == 1707  # ceil(512 / 0.3)
    assert ConditionedStream(h_per_sample=3.0, block_entropy_bits=1024).block_samples == 342


def test_batching_does_not_change_output(rng):
    data = rng.integers(0, 256, size=5000, dtype=np.uint8)
    one = ConditionedStream(1.5, out_bytes=16).feed(data)
    stream = ConditionedStream(1.5, out_bytes=16)
    pieces = np.split(data, np.sort(rng.integers(0, data.size, size=25)))
    assert b"".join(stream.feed(p) for p in pieces) == one
    assert len(one) == 16 * (5000 // stream.block_samples)


def test_validation():
    with pytest.raises(ValueError, match="h_per_sample"):
        ConditionedStream(0.0)
    with pytest.raises(ValueError, match="h_per_sample"):
        ConditionedStream(9.0)
    with pytest.raises(ValueError, match="at least as much entropy"):
        ConditionedStream(2.0, block_entropy_bits=128, out_bytes=32)
    with pytest.raises(ValueError, match="out_bytes"):
        ConditionedStream(2.0, block_entropy_bits=1024, out_bytes=65)
    stream = ConditionedStream(2.0)
    with pytest.raises(ValueError, match=r"0\.\.255"):
        stream.feed(np.array([1, 256]))
    with pytest.raises(ValueError, match=r"0\.\.255"):
        stream.feed(np.array([0.5]))
    assert stream.feed([]) == b""
