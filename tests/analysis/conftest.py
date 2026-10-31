"""Reference sequences for the analysis tests, computed exactly.

SP 800-22 validates its tests on two famous constants: the first 100 bits of
pi (most worked examples) and the first 1,000,000 bits of e (Appendix B and
the larger examples). Instead of shipping NIST's ``data/data.e`` (and trusting
a transcription) both are computed here with exact integer arithmetic.

NIST's ``data.e`` starts with the integer part of e: ``10.1011011111...``
gives the bits ``101011011111...``. The frequency test settles it: with the
leading ``10`` the first 10^6 bits have S_n = 58 and P = 0.953749 (the
published value); the fractional bits alone give S_n = 60, P = 0.952156.
``test_sp800_22_reference.py`` checks this explicitly.
"""

from __future__ import annotations

import math

import pytest

from puffer.analysis.bits import BitArray, ascii_to_bits

GUARD_BITS = 64


def _strip_guard(value: int, max_error: int) -> int:
    """``floor(x * 2^bits)`` from ``value`` = x * 2^(bits + GUARD) +- ``max_error``.

    Dropping the guard bits is exact unless they lie within the error bound
    of a carry boundary, which would need ~50 consecutive equal bits of x.
    """
    low = value & ((1 << GUARD_BITS) - 1)
    assert max_error <= low < (1 << GUARD_BITS) - max_error, "too close to a carry boundary"
    return value >> GUARD_BITS


def e_binary_digits(fraction_bits: int) -> str:
    """Binary expansion of e: ``"10"`` followed by ``fraction_bits`` fractional bits.

    e = sum_k 1/k! is summed by binary splitting into one exact fraction P/Q
    with enough terms K that the tail (< 2/(K+1)!) is below 2^-(bits + guard).
    """
    scale = fraction_bits + GUARD_BITS
    terms = 1
    while math.lgamma(terms + 2) / math.log(2) < scale + 2:
        terms += 1024

    def split(a: int, b: int) -> tuple[int, int]:
        # sum_{k=a+1}^{b} a!/k! == p / q with q = (a+1)...(b)
        if b - a == 1:
            return 1, b
        mid = (a + b) // 2
        p1, q1 = split(a, mid)
        p2, q2 = split(mid, b)
        return p1 * q2 + p2, q1 * q2

    p, q = split(0, terms)  # p/q = sum_{k=1}^{terms} 1/k!
    # Truncating the series and the division each lose less than one unit.
    return format(_strip_guard(((p + q) << scale) // q, max_error=2), "b")


def pi_binary_digits(fraction_bits: int) -> str:
    """Binary expansion of pi (``"11"`` + fractional bits) via Machin's formula."""
    one = 1 << (fraction_bits + GUARD_BITS)

    def arctan_inv(x: int) -> int:
        # arctan(1/x) * one; every truncated term errs by less than one unit.
        total = term = one // x
        k, x2 = 1, x * x
        while term:
            term //= x2
            total += (term if k % 2 == 0 else -term) // (2 * k + 1)
            k += 1
        return total

    pi = 4 * (4 * arctan_inv(5) - arctan_inv(239))
    return format(_strip_guard(pi, max_error=1 << 16), "b")


@pytest.fixture(scope="session")
def e_digits() -> str:
    """``"10"`` + 1,000,000 fractional bits of e (a little more than needed)."""
    return e_binary_digits(1_000_000)


@pytest.fixture(scope="session")
def e_bits(e_digits: str) -> BitArray:
    """The first 1,000,000 bits of e as NIST's ``data.e`` has them."""
    bits = ascii_to_bits(e_digits[:1_000_000])
    bits.flags.writeable = False
    return bits


@pytest.fixture(scope="session")
def pi_100() -> BitArray:
    """The first 100 bits of pi (``11.00100100...``), the sequence of most worked examples."""
    return ascii_to_bits(pi_binary_digits(128)[:100])
