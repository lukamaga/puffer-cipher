"""Reference results for the first 1,000,000 bits of e (SP 800-22 Appendix B).

The bits are computed exactly in ``conftest.py``. Appendix B and the worked
examples on e were produced with NIST STS, and two of its tests deviate from
the rev1a text:

* overlapping template matching: STS 2.1.2 recomputes the class
  probabilities with the pre-correction approximation
  ``P(U = u) = e^-eta 2^-u sum_l C(u-1, l-1) eta^l / l!`` (overwriting the
  corrected table in its own source). Appendix B (0.110434) and example
  2.8.8 come from those; with the rev1a probabilities that we use by
  default the same counts give 0.159032.
* linear complexity: STS 2.1.2 has pi_0 = 0.01047 (typo for 0.010417);
  Appendix B (0.826335) and example 2.10.8 inherit it. The rev1a value gives
  0.826194.

Both are reproduced below by passing STS's probabilities explicitly, which
also proves the statistics (counts) are identical. The cumulative sums
values agree to 7e-7 rather than 5e-7: z = 956 / 898 are integers and the
series was evaluated three independent ways (scipy ndtr, libm erf, a port of
STS's cephes erf), all giving 0.6698865 / 0.7242653.
"""

from __future__ import annotations

import math

import pytest

from puffer.analysis import sp800_22 as sp
from puffer.analysis.bits import ascii_to_bits

STS_LINEAR_COMPLEXITY_PI = (0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833)


def sts_overlapping_probabilities(m: int = 9, block_size: int = 1032, k: int = 5) -> list[float]:
    """NIST STS 2.1.2 ``Pr(u, eta)`` - the approximation rev1a replaced."""
    eta = (block_size - m + 1) / 2**m / 2

    def pr(u: int) -> float:
        if u == 0:
            return math.exp(-eta)
        return sum(
            math.exp(-eta - u * math.log(2) + j * math.log(eta) - math.lgamma(j + 1))
            * math.comb(u - 1, j - 1)
            for j in range(1, u + 1)
        )

    probabilities = [pr(u) for u in range(k)]
    return [*probabilities, 1.0 - sum(probabilities)]


def approx(value: float) -> object:
    return pytest.approx(value, abs=1e-6)


def test_data_e_includes_the_integer_part(e_digits):
    """NIST's data.e = "10" + fraction bits: only that reproduces 0.953749 *and*
    the block frequency value (dropping just the leading 1 also has S_n = 58)."""
    with_integer = ascii_to_bits(e_digits[:1_000_000])
    fraction_only = ascii_to_bits(e_digits[2:1_000_002])
    shifted = ascii_to_bits(e_digits[1:1_000_001])
    assert sp.frequency(with_integer).p_values == [approx(0.953749)]
    assert sp.frequency(fraction_only).statistics["sum"] == 60
    assert sp.frequency(fraction_only).p_values != [approx(0.953749)]
    assert sp.frequency(shifted).p_values == [approx(0.953749)]
    assert sp.block_frequency(shifted).p_values != [approx(0.211072)]
    assert sp.block_frequency(with_integer).p_values == [approx(0.211072)]


def test_appendix_b_e_column(e_bits):
    suite = sp.run_suite(e_bits)
    assert [r.name for r in suite.results] == list(sp.SUITE_TESTS)
    assert not suite.skipped
    expected = {
        "frequency": [0.953749],
        "block_frequency": [0.211072],
        "runs": [0.561917],
        "longest_run_of_ones": [0.718945],
        "binary_matrix_rank": [0.306156],
        "discrete_fourier_transform": [0.847187],
        "universal": [0.282568],
        "approximate_entropy": [0.700073],
        "serial": [0.766182, 0.462921],
    }
    for name, values in expected.items():
        assert suite[name].p_values == [approx(v) for v in values], name
    cusum = suite["cumulative_sums"].p_values
    assert cusum == [pytest.approx(0.669887, abs=1e-6), pytest.approx(0.724266, abs=1e-6)]
    template = suite["non_overlapping_template_matching"]
    assert len(template.p_values) == 148
    assert template.p_values[template.labels.index("000000001")] == approx(0.078790)
    excursions = suite["random_excursions"]
    assert excursions.p_values[excursions.labels.index("x=+1")] == approx(0.786868)
    variant = suite["random_excursions_variant"]
    assert variant.p_values[variant.labels.index("x=-1")] == approx(0.826009)
    # The two tests where STS deviates from rev1a (see module docstring):
    assert suite["overlapping_template_matching"].p_values == [approx(0.159032)]
    assert suite["linear_complexity"].p_values == [approx(0.826194)]


def test_overlapping_template_e_counts_and_sts_probabilities(e_bits):
    """Example 2.8.8 / Appendix B: identical counts, STS's legacy probabilities."""
    r = sp.overlapping_template_matching(e_bits)
    assert r.statistics["blocks"] == 968
    assert r.statistics["counts"] == [329, 164, 150, 111, 78, 136]
    legacy = sp.overlapping_template_matching(e_bits, probabilities=sts_overlapping_probabilities())
    assert legacy.statistics["chi_square"] == approx(8.965859)
    assert legacy.p_values == [approx(0.110434)]


def test_linear_complexity_e_with_sts_probabilities(e_bits):
    """Appendix B (M = 500) and example 2.10.8 (M = 1000) use STS's pi_0 = 0.01047."""
    r500 = sp.linear_complexity(e_bits, probabilities=STS_LINEAR_COMPLEXITY_PI)
    assert r500.statistics["blocks"] == 2000
    assert r500.p_values == [approx(0.826335)]
    r1000 = sp.linear_complexity(e_bits, block_size=1000, probabilities=STS_LINEAR_COMPLEXITY_PI)
    assert r1000.statistics["counts"] == [11, 31, 116, 501, 258, 57, 26]
    assert r1000.statistics["chi_square"] == approx(2.700348)
    assert r1000.p_values == [approx(0.845406)]


def test_rank_example_on_100000_bits_of_e(e_bits):
    r = sp.binary_matrix_rank(e_bits[:100_000])
    assert r.statistics["matrices"] == 97
    assert r.statistics["counts"] == [23, 60, 14]
    assert r.statistics["chi_square"] == pytest.approx(1.2619656, abs=1e-7)
    assert r.p_values == [approx(0.532069)]


def test_serial_example_m2_on_e(e_bits):
    r = sp.serial(e_bits, block_length=2)
    assert r.statistics["delta1"] == approx(0.339764)
    assert r.statistics["delta2"] == approx(0.336400)
    assert r.p_values == [approx(0.843764), approx(0.561915)]


def test_random_excursions_example_on_e(e_bits):
    """Example 2.14.8. The printed values for the negative states match to the
    last digit; x = +1 is checked against Appendix B in test_appendix_b_e_column."""
    r = sp.random_excursions(e_bits)
    assert r.statistics["cycles"] == 1490
    printed = {
        "x=-4": (3.835698, 0.573306),
        "x=-3": (7.318707, 0.197996),
        "x=-2": (7.861927, 0.164011),
        "x=-1": (15.692617, 0.007779),
    }
    for label, (chi2, p) in printed.items():
        i = r.labels.index(label)
        assert r.statistics["chi_square"][i] == approx(chi2), label
        assert r.p_values[i] == approx(p), label
    assert not r.passed  # x = -1 is below 0.01, as printed in the example


def test_random_excursions_variant_example_on_e(e_bits):
    r = sp.random_excursions_variant(e_bits)
    assert r.statistics["cycles"] == 1490
    assert r.statistics["visits"] == [
        *[1450, 1435, 1380, 1366, 1412, 1475, 1480, 1468, 1502],
        *[1409, 1369, 1396, 1479, 1599, 1628, 1619, 1620, 1610],
    ]
    printed = [
        *[0.858946, 0.794755, 0.576249, 0.493417, 0.633873, 0.917283, 0.934708, 0.816012],
        *[0.826009, 0.137861, 0.200642, 0.441254, 0.939291, 0.505683, 0.445935, 0.512207],
        *[0.538635, 0.593930],
    ]
    assert r.p_values == [approx(p) for p in printed]
