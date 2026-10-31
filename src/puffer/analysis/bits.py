"""Bit-sequence conventions shared by the statistical test suites.

NIST SP 800-22 is defined over a sequence of bits ``ε_1 … ε_n``, while every
generator in this project (fish, sensor noise, PufferRNG) produces *bytes*.
This module pins down the one conversion everything else relies on: a byte
expands **most significant bit first**, the order NIST STS uses for its
``binary`` input format and the order ``dieharder``/``ent -b`` read bits in.
Getting this wrong silently would not break any test - it would just test a
different (bit-permuted) sequence than the one a reviewer reproduces with the
reference tools.

Bit arrays are 1-D ``numpy.uint8`` arrays holding only 0 and 1. That costs 8x
the memory of packed bytes, but a million-bit sequence is still only 1 MB and
the unpacked form lets every SP 800-22 test be a handful of vectorised numpy
operations instead of bit twiddling.

Two file formats are supported, matching NIST STS: ``"binary"`` (raw bytes,
MSB first) and ``"ascii"`` (the characters ``0``/``1``, whitespace ignored -
the format of STS's ``data/data.e`` reference files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

BitArray = npt.NDArray[np.uint8]
"""1-D array of 0/1 values, one bit per element."""

BitFormat = Literal["binary", "ascii"]
"""On-disk representation of a bit sequence (see module docstring)."""

_ASCII_ZERO, _ASCII_ONE = ord("0"), ord("1")
_ASCII_WHITESPACE = np.frombuffer(b" \t\n\r\v\f", dtype=np.uint8)


def as_bits(bits: npt.ArrayLike) -> BitArray:
    """Validate ``bits`` and return it as a contiguous 1-D ``uint8`` 0/1 array.

    Accepts any array-like of integers or booleans. No copy is made when the
    input already has the canonical form. Values other than 0 and 1 raise
    ``ValueError`` instead of being coerced: a byte array passed where a bit
    array was expected is a bug that would otherwise produce plausible but
    meaningless p-values.
    """
    arr = np.asarray(bits)
    if arr.ndim != 1:
        raise ValueError(f"bit sequence must be 1-D, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"bit sequence must be integer or boolean, got dtype {arr.dtype}")
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 1):
        raise ValueError("bit sequence may only contain 0 and 1")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def bytes_to_bits(data: bytes | bytearray | memoryview) -> BitArray:
    """Expand bytes into bits, most significant bit of each byte first."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: npt.ArrayLike, *, pad: bool = False) -> bytes:
    """Pack bits (MSB first) into bytes; the inverse of :func:`bytes_to_bits`.

    A length that is not a multiple of 8 raises ``ValueError`` unless ``pad``
    is true, in which case the last byte is zero-padded on the right. Padding
    is opt-in because it changes the data: re-testing the packed bytes would
    test extra zero bits.
    """
    arr = as_bits(bits)
    if arr.size % 8 and not pad:
        raise ValueError(f"{arr.size} bits is not a whole number of bytes (pass pad=True)")
    return np.packbits(arr).tobytes()


def ascii_to_bits(text: str | bytes) -> BitArray:
    """Parse NIST STS ASCII format: ``0``/``1`` characters, whitespace ignored."""
    raw = text.encode("ascii") if isinstance(text, str) else text
    arr = np.frombuffer(raw, dtype=np.uint8)
    is_bit = (arr == _ASCII_ZERO) | (arr == _ASCII_ONE)
    bad = ~is_bit & ~np.isin(arr, _ASCII_WHITESPACE)
    if bad.any():
        pos = int(np.argmax(bad))
        raise ValueError(f"invalid character {chr(int(arr[pos]))!r} at offset {pos} in ASCII bits")
    return (arr[is_bit] - _ASCII_ZERO).astype(np.uint8)


def bits_to_ascii(bits: npt.ArrayLike, *, line_length: int | None = None) -> str:
    """Render bits as ``0``/``1`` characters, optionally wrapped every ``line_length``."""
    arr = as_bits(bits)
    text = (arr + _ASCII_ZERO).tobytes().decode("ascii")
    if line_length is None:
        return text
    if line_length <= 0:
        raise ValueError("line_length must be positive")
    return "\n".join(text[i : i + line_length] for i in range(0, len(text), line_length))


def load_bits(
    path: str | Path, *, fmt: BitFormat = "binary", max_bits: int | None = None
) -> BitArray:
    """Read a bit sequence from ``path`` in the given format.

    ``max_bits`` truncates the result (NIST STS's "sequence length"); for the
    binary format only the bytes actually needed are read.
    """
    if max_bits is not None and max_bits < 0:
        raise ValueError("max_bits must be non-negative")
    p = Path(path)
    if fmt == "binary":
        if max_bits is None:
            return bytes_to_bits(p.read_bytes())
        with p.open("rb") as fh:
            data = fh.read((max_bits + 7) // 8)
        return bytes_to_bits(data)[:max_bits]
    if fmt == "ascii":
        bits = ascii_to_bits(p.read_bytes())
        return bits if max_bits is None else bits[:max_bits]
    raise ValueError(f"unknown bit format {fmt!r}")


def save_bits(path: str | Path, bits: npt.ArrayLike, *, fmt: BitFormat = "binary") -> None:
    """Write ``bits`` to ``path``.

    The binary format requires a whole number of bytes (see
    :func:`bits_to_bytes`); the ASCII format wraps lines at 64 characters
    like the NIST reference data files and ends with a newline.
    """
    p = Path(path)
    if fmt == "binary":
        p.write_bytes(bits_to_bytes(bits))
    elif fmt == "ascii":
        p.write_text(bits_to_ascii(bits, line_length=64) + "\n", encoding="ascii")
    else:
        raise ValueError(f"unknown bit format {fmt!r}")


def split_sequences(bits: npt.ArrayLike, length: int) -> npt.NDArray[np.uint8]:
    """Cut a long bit stream into ``m = len(bits) // length`` sequences.

    Returns a read-only ``(m, length)`` view (no copy); trailing bits that do
    not fill a whole sequence are dropped. This mirrors how NIST STS takes
    "m bitstreams of length n" from one input file for the multi-sequence
    analysis of SP 800-22 section 4.2.
    """
    if length <= 0:
        raise ValueError("sequence length must be positive")
    arr = as_bits(bits)
    m = arr.size // length
    view = arr[: m * length].reshape(m, length)
    view.flags.writeable = False
    return view
