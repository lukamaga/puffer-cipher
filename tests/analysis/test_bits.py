"""Bit conventions: MSB-first expansion and the NIST STS file formats."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from puffer.analysis.bits import (
    as_bits,
    ascii_to_bits,
    bits_to_ascii,
    bits_to_bytes,
    bytes_to_bits,
    load_bits,
    save_bits,
    split_sequences,
)


def test_bytes_expand_most_significant_bit_first():
    assert bytes_to_bits(b"\x80\x01\xa5").tolist() == [
        *[1, 0, 0, 0, 0, 0, 0, 0],
        *[0, 0, 0, 0, 0, 0, 0, 1],
        *[1, 0, 1, 0, 0, 1, 0, 1],
    ]
    assert bytes_to_bits(b"").size == 0
    assert bytes_to_bits(b"\x00").dtype == np.uint8


@given(st.binary(max_size=256))
def test_bytes_bits_round_trip(data: bytes):
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_bits_to_bytes_requires_whole_bytes_unless_padding():
    with pytest.raises(ValueError, match="whole number of bytes"):
        bits_to_bytes([1, 0, 1])
    assert bits_to_bytes([1, 0, 1], pad=True) == b"\xa0"


def test_as_bits_accepts_bools_and_rejects_non_bits():
    assert as_bits(np.array([True, False])).tolist() == [1, 0]
    assert as_bits([0, 1, 1]).dtype == np.uint8
    with pytest.raises(ValueError, match="only contain 0 and 1"):
        as_bits(np.frombuffer(b"\x02", dtype=np.uint8))
    with pytest.raises(ValueError, match="only contain 0 and 1"):
        as_bits([-1, 0])
    with pytest.raises(ValueError, match="1-D"):
        as_bits([[0, 1]])
    with pytest.raises(TypeError, match="integer or boolean"):
        as_bits([0.0, 1.0])


def test_as_bits_does_not_copy_canonical_input():
    bits = np.array([0, 1, 1], dtype=np.uint8)
    assert as_bits(bits) is bits


def test_ascii_format_ignores_whitespace_and_rejects_other_characters():
    assert ascii_to_bits("10 1\n1\t0\r\n").tolist() == [1, 0, 1, 1, 0]
    assert ascii_to_bits(b"01").tolist() == [0, 1]
    with pytest.raises(ValueError, match=r"invalid character '2' at offset 3"):
        ascii_to_bits("1012")


def test_bits_to_ascii_wraps_lines():
    assert bits_to_ascii([1, 0, 1]) == "101"
    assert bits_to_ascii([1, 0, 1, 1, 0], line_length=2) == "10\n11\n0"
    with pytest.raises(ValueError, match="positive"):
        bits_to_ascii([1], line_length=0)


@pytest.mark.parametrize("fmt", ["binary", "ascii"])
def test_save_and_load_round_trip(tmp_path, rng, fmt):
    bits = rng.integers(0, 2, size=1000, dtype=np.uint8)
    path = tmp_path / f"bits.{fmt}"
    save_bits(path, bits, fmt=fmt)
    np.testing.assert_array_equal(load_bits(path, fmt=fmt), bits)
    np.testing.assert_array_equal(load_bits(path, fmt=fmt, max_bits=77), bits[:77])


def test_binary_file_is_raw_msb_first_bytes(tmp_path):
    path = tmp_path / "x.bin"
    save_bits(path, [1, 1, 1, 1, 0, 0, 0, 0])
    assert path.read_bytes() == b"\xf0"


def test_ascii_file_wraps_at_64_characters(tmp_path):
    path = tmp_path / "x.txt"
    save_bits(path, np.ones(130, dtype=np.uint8), fmt="ascii")
    assert [len(line) for line in path.read_text().splitlines()] == [64, 64, 2]


def test_load_and_save_reject_unknown_formats(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unknown bit format"):
        load_bits(path, fmt="hex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown bit format"):
        save_bits(path, [0] * 8, fmt="hex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        load_bits(path, max_bits=-1)


def test_split_sequences_is_a_read_only_view_dropping_the_tail():
    bits = np.arange(10) % 2
    seqs = split_sequences(bits, 3)
    assert seqs.shape == (3, 3)
    assert seqs.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert not seqs.flags.writeable
    with pytest.raises(ValueError, match="positive"):
        split_sequences(bits, 0)
