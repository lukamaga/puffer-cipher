"""In-process equivalent of John Walker's ``ent`` pseudorandom-sequence test.

``ent`` (fourmilab.ch) is the de-facto first look at a random byte stream and
the numbers people expect next to a LavaRand-style demo: entropy per byte,
optimum compression, chi-square, arithmetic mean, Monte Carlo pi and serial
correlation. Re-implementing it on numpy means the dashboard and reports do
not depend on the binary being installed; :mod:`puffer.analysis.external`
runs the real tool and the tests cross-check the two to ent's printed
precision.

Faithfulness to ``ent`` matters more than "improving" it, so the quirks are
kept on purpose:

* serial correlation is *circular* (the last sample is paired with the
  first) and undefined when all samples are equal;
* Monte Carlo pi uses consecutive 6-byte groups as two 24-bit coordinates
  and ignores a trailing partial group - also in bit mode;
* bit mode (``ent -b``) treats the input as a stream of bits, most
  significant bit of each byte first.

Two deliberate differences: sums are exact integers instead of doubles
(identical for any realistic size, exact for huge ones), and the chi-square
probability uses scipy's regularised gamma function instead of ent's
Hill & Pike series - agreement is far below ent's 0.01 % display resolution.

As everywhere in ``puffer.analysis`` (DESIGN.md rule 4): these statistics
describe a byte stream; on hashed output they say nothing about entropy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import special

_MONTE_GROUP = 6  # bytes per Monte Carlo point: two 24-bit coordinates
_MONTE_RADIUS_SQ = (256**3 - 1) ** 2  # ent's "incirc"
_CHUNK = _MONTE_GROUP * (1 << 18)  # 1.5 MiB, a multiple of the group size
_POPCOUNT = np.array([bin(v).count("1") for v in range(256)], dtype=np.int64)
_ADJACENT_ONES = np.array([bin(v & (v >> 1)).count("1") for v in range(256)], dtype=np.int64)
_BYTE_VALUES = np.arange(256, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class EntStats:
    """The figures ``ent`` reports, in ent's units.

    ``size`` counts samples: bytes, or bits in ``bit_mode``. Entropy is per
    sample (bits per byte, or bits per bit). ``chi_square_percent`` is ent's
    "would exceed this value N percent of the times": the probability, in
    percent, that a truly random sequence gives a larger chi-square (ent
    flags < 1 % and > 99 % as suspicious). ``monte_carlo_pi`` is ``None``
    with fewer than 6 bytes and ``serial_correlation`` is ``None`` when all
    samples are equal (ent prints "undefined").
    """

    size: int
    bit_mode: bool
    entropy: float
    compression_percent: float
    chi_square: float
    chi_square_percent: float
    mean: float
    monte_carlo_pi: float | None
    monte_carlo_error_percent: float | None
    serial_correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def report(self) -> str:
        """Render the statistics exactly as ``ent`` prints them (without ``-t``).

        Byte-identical to ent's output except where ent prints garbage: a
        Monte Carlo value for fewer than 6 bytes ("-nan") and its "more than
        than 99.99" typo.
        """
        unit = "bit" if self.bit_mode else "byte"
        ideal = "0.5" if self.bit_mode else "127.5"
        if self.chi_square_percent < 0.01:
            chi_p = "less than 0.01"
        elif self.chi_square_percent > 99.99:
            chi_p = "more than 99.99"
        else:
            chi_p = f"{self.chi_square_percent:1.2f}"
        if self.monte_carlo_pi is None or self.monte_carlo_error_percent is None:
            monte = "undefined (fewer than 6 bytes)."
        else:
            monte = (
                f"{self.monte_carlo_pi:1.9f} (error {self.monte_carlo_error_percent:1.2f} percent)."
            )
        if self.serial_correlation is None:
            scc = "undefined (all values equal!)."
        else:
            scc = f"{self.serial_correlation:f} (totally uncorrelated = 0.0)."
        return (
            f"Entropy = {self.entropy:f} bits per {unit}.\n\n"
            "Optimum compression would reduce the size\n"
            f"of this {self.size} {unit} file by {int(self.compression_percent)} percent.\n\n"
            f"Chi square distribution for {self.size} samples is {self.chi_square:1.2f}, "
            "and randomly\n"
            f"would exceed this value {chi_p} percent of the times.\n\n"
            f"Arithmetic mean value of data {unit}s is {self.mean:1.4f} ({ideal} = random).\n"
            f"Monte Carlo value for Pi is {monte}\n"
            f"Serial correlation coefficient is {scc}\n"
        )


def derive_ent_stats(
    *,
    size: int,
    bit_mode: bool,
    entropy: float,
    chi_square: float,
    mean: float,
    monte_carlo_pi: float | None,
    serial_correlation: float | None,
) -> EntStats:
    """Build :class:`EntStats` from ent's primary figures.

    ``ent -t`` prints only the primary figures; compression, the chi-square
    probability and the Monte Carlo error are derived exactly as ent derives
    them for its human-readable output. Shared by :func:`ent_stats` and the
    ``ent -t`` parser so both sides of a cross-check use identical formulas.
    """
    bits_per_sample = 1.0 if bit_mode else 8.0
    freedom = 1 if bit_mode else 255
    error = None
    if monte_carlo_pi is not None:
        error = 100.0 * abs(math.pi - monte_carlo_pi) / math.pi
    return EntStats(
        size=size,
        bit_mode=bit_mode,
        entropy=entropy,
        compression_percent=100.0 * (bits_per_sample - entropy) / bits_per_sample,
        chi_square=chi_square,
        chi_square_percent=100.0 * float(special.chdtrc(freedom, chi_square)),
        mean=mean,
        monte_carlo_pi=monte_carlo_pi,
        monte_carlo_error_percent=error,
        serial_correlation=serial_correlation,
    )


class _EntAccumulator:
    """Streaming sums behind ent's statistics; feed chunks with :meth:`update`.

    Only a byte histogram and a handful of exact integer sums are kept, so a
    file of any size is processed in constant memory. Bit-mode figures are
    derived from the same byte stream: per-byte popcounts give the bit
    histogram and ``popcount(v & v >> 1)`` the adjacent-ones products inside
    a byte; only byte boundaries need the neighbouring byte.
    """

    def __init__(self) -> None:
        self.histogram = np.zeros(256, dtype=np.int64)
        self.byte_products = 0  # sum of x_i * x_(i+1) over consecutive bytes
        self.bit_boundary_products = 0  # sum of lsb(x_i) * msb(x_(i+1))
        self.first: int | None = None
        self.last: int | None = None
        self.monte_points = 0
        self.monte_inside = 0
        self._pending = np.zeros(0, dtype=np.uint8)  # partial Monte Carlo group

    def update(self, data: bytes | bytearray | memoryview) -> None:
        chunk = np.frombuffer(data, dtype=np.uint8)
        if chunk.size == 0:
            return
        self.histogram += np.bincount(chunk, minlength=256)
        wide = chunk.astype(np.int64)
        if self.last is None:
            self.first = int(wide[0])
        else:
            self.byte_products += self.last * int(wide[0])
            self.bit_boundary_products += (self.last & 1) * (int(wide[0]) >> 7)
        self.byte_products += int(np.dot(wide[:-1], wide[1:]))
        self.bit_boundary_products += int(np.dot(wide[:-1] & 1, wide[1:] >> 7))
        self.last = int(wide[-1])
        self._monte_carlo(chunk)

    def _monte_carlo(self, chunk: npt.NDArray[np.uint8]) -> None:
        data = np.concatenate([self._pending, chunk]) if self._pending.size else chunk
        usable = data.size - data.size % _MONTE_GROUP
        groups = data[:usable].reshape(-1, _MONTE_GROUP).astype(np.int64)
        x = (groups[:, 0] << 16) | (groups[:, 1] << 8) | groups[:, 2]
        y = (groups[:, 3] << 16) | (groups[:, 4] << 8) | groups[:, 5]
        self.monte_points += groups.shape[0]
        self.monte_inside += int(np.count_nonzero(x * x + y * y <= _MONTE_RADIUS_SQ))
        self._pending = data[usable:].copy()

    def result(self, *, bit_mode: bool) -> EntStats:
        n_bytes = int(self.histogram.sum())
        if n_bytes == 0 or self.first is None or self.last is None:
            raise ValueError("ent statistics need at least one byte")
        if bit_mode:
            ones = int(np.dot(self.histogram, _POPCOUNT))
            counts = np.array([8 * n_bytes - ones, ones], dtype=np.int64)
            total, value_sum, square_sum = 8 * n_bytes, ones, ones
            products = (
                int(np.dot(self.histogram, _ADJACENT_ONES))
                + self.bit_boundary_products
                + (self.last & 1) * (self.first >> 7)
            )
        else:
            counts = self.histogram
            total = n_bytes
            value_sum = int(np.dot(self.histogram, _BYTE_VALUES))
            square_sum = int(np.dot(self.histogram, _BYTE_VALUES * _BYTE_VALUES))
            products = self.byte_products + self.last * self.first
        probabilities = counts[counts > 0] / total
        entropy = float(np.sum(probabilities * np.log2(1.0 / probabilities)))  # as ent: no -0.0
        expected = total / counts.size
        chi_square = float(np.sum((counts - expected) ** 2) / expected)
        denominator = total * square_sum - value_sum * value_sum
        scc = None if denominator == 0 else (total * products - value_sum**2) / denominator
        monte_pi = None
        if self.monte_points:
            monte_pi = 4.0 * self.monte_inside / self.monte_points
        return derive_ent_stats(
            size=total,
            bit_mode=bit_mode,
            entropy=entropy,
            chi_square=chi_square,
            mean=value_sum / total,
            monte_carlo_pi=monte_pi,
            serial_correlation=scc,
        )


def _accumulate(chunks: Iterable[bytes | memoryview], *, bit_mode: bool) -> EntStats:
    acc = _EntAccumulator()
    for chunk in chunks:
        acc.update(chunk)
    return acc.result(bit_mode=bit_mode)


def ent_stats(data: bytes | bytearray | memoryview, *, bit_mode: bool = False) -> EntStats:
    """ent's statistics for ``data`` (``ent -b`` semantics when ``bit_mode``).

    Raises ``ValueError`` for empty input, where ent prints NaNs.
    """
    view = memoryview(data).cast("B")
    return _accumulate(
        (view[i : i + _CHUNK] for i in range(0, len(view), _CHUNK)), bit_mode=bit_mode
    )


def ent_stats_file(path: str | Path, *, bit_mode: bool = False) -> EntStats:
    """Like :func:`ent_stats` but streams the file in 1.5 MiB chunks."""
    with Path(path).open("rb") as fh:
        return _accumulate(iter(lambda: fh.read(_CHUNK), b""), bit_mode=bit_mode)
