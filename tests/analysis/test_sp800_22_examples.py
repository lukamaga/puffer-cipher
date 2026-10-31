"""The worked examples of NIST SP 800-22 Rev. 1a, section by section.

Values are the ones printed in the specification (6 decimals, compared with
an absolute tolerance of 1e-6). Where the printed example is internally
inconsistent, the test pins what can be verified by hand and says why.
The examples computed on the first 10^6 bits of e live in
``test_sp800_22_reference.py``.
"""

from __future__ import annotations

import math

import pytest
from scipy import special

from puffer.analysis import sp800_22 as sp
from puffer.analysis.bits import ascii_to_bits

SPEC_PI_100 = (
    "11001001000011111101101010100010001000010110100011"
    "00001000110100110001001100011001100010100010111000"
)


def bits(text: str):
    return ascii_to_bits(text)


def approx(value: float) -> object:
    return pytest.approx(value, abs=1e-6)


def test_pi_fixture_is_the_spec_example_sequence(pi_100):
    assert pi_100.tolist() == bits(SPEC_PI_100).tolist()


# 2.1 Frequency (monobit) ----------------------------------------------------


def test_frequency_10_bits():
    r = sp.frequency(bits("1011010101"))
    assert r.statistics["sum"] == 2
    assert r.statistics["s_obs"] == approx(0.632456)
    assert r.p_values == [approx(0.527089)]
    assert r.passed


def test_frequency_100_bits(pi_100):
    r = sp.frequency(pi_100)
    assert r.statistics["sum"] == -16
    assert r.statistics["s_obs"] == approx(1.6)
    assert r.p_values == [approx(0.109599)]


# 2.2 Block frequency --------------------------------------------------------


def test_block_frequency_10_bits():
    r = sp.block_frequency(bits("0110011010"), block_size=3)
    assert r.statistics["blocks"] == 3
    assert r.statistics["chi_square"] == approx(1.0)
    assert r.p_values == [approx(0.801252)]


def test_block_frequency_100_bits(pi_100):
    r = sp.block_frequency(pi_100, block_size=10)
    assert r.statistics["chi_square"] == approx(7.2)
    assert r.p_values == [approx(0.706438)]


# 2.3 Runs -------------------------------------------------------------------


def test_runs_10_bits():
    r = sp.runs(bits("1001101011"))
    assert r.statistics["pi"] == approx(0.6)
    assert r.statistics["runs"] == 7
    assert r.p_values == [approx(0.147232)]


def test_runs_100_bits(pi_100):
    r = sp.runs(pi_100)
    assert r.statistics["pi"] == approx(0.42)
    assert r.statistics["runs"] == 52
    assert r.p_values == [approx(0.500798)]


# 2.4 Longest run of ones ----------------------------------------------------


def test_longest_run_128_bits():
    r = sp.longest_run_of_ones(
        bits(
            "11001100000101010110110001001100111000000000001001001101010100010001"
            "001111010110100000001101011111001100111001101101100010110010"
        )
    )
    assert r.statistics["block_size"] == 8
    assert r.statistics["counts"] == [4, 9, 3, 0]
    assert r.statistics["chi_square"] == approx(4.882457)
    assert r.p_values == [approx(0.180609)]


# 2.5 Binary matrix rank -----------------------------------------------------


def test_rank_3x3_example_counts_and_the_spec_s_probabilities():
    """Section 2.5.4: two 3x3 matrices with ranks 2 and 3.

    The printed chi2 = 0.596953 (P = 0.741948) plugs the *32x32* class
    probabilities (0.2888, 0.5776, 0.1336) into a 3x3 example. We use the
    exact 3x3 probabilities (21/64, 147/256, 25/256), so our P differs; the
    test reproduces the printed numbers from our counts to show that the
    difference is only that substitution.
    """
    r = sp.binary_matrix_rank(bits("01011001001010101101"), rows=3, cols=3)
    assert r.statistics["matrices"] == 2
    assert r.statistics["counts"] == [1, 1, 0]
    assert r.statistics["probabilities"] == pytest.approx([21 / 64, 147 / 256, 25 / 256])
    expected = [2 * p for p in (0.2888, 0.5776, 0.1336)]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(r.statistics["counts"], expected, strict=True))
    assert chi2 == approx(0.596953)
    assert math.exp(-chi2 / 2) == approx(0.741948)
    assert r.p_values == [approx(math.exp(-r.statistics["chi_square"] / 2))]


# 2.6 Discrete Fourier transform ---------------------------------------------


def test_dft_10_bits():
    """X = (1,-1,-1,1,-1,1,-1,-1,1,1): |S_k| for k = 0..4 is (0, 2, 2 sqrt 5, 2, 2 sqrt 5),
    all below T = sqrt(10 ln 20) = 5.4733, so N1 = 5 and d = 0.25 / sqrt(0.11875)."""
    r = sp.discrete_fourier_transform(bits("1001010011"))
    assert r.statistics["threshold"] == approx(math.sqrt(10 * math.log(20)))
    assert r.statistics["expected"] == approx(4.75)
    assert r.statistics["observed"] == 5
    assert r.statistics["d"] == approx(0.725476)
    assert r.p_values == [approx(0.468160)]


# 2.7 Non-overlapping template matching ---------------------------------------


def test_non_overlapping_template_20_bits():
    r = sp.non_overlapping_template_matching(
        bits("10100100101110010110"), template_length=3, n_blocks=2, templates=["001"]
    )
    assert r.labels == ["001"]
    assert r.statistics["matches"] == [[2, 1]]
    assert r.statistics["mean"] == approx(1.0)
    assert r.statistics["variance"] == approx(0.46875)
    assert r.statistics["chi_square"] == [approx(2.133333)]
    assert r.p_values == [approx(0.344154)]


def test_aperiodic_template_counts_match_nist_sts_template_files():
    # STS ships templates/template2 .. template21; the counts per length are
    # the numbers of unbordered binary words (OEIS A003000).
    assert [len(sp.aperiodic_templates(m)) for m in range(2, 11)] == [
        2,
        4,
        6,
        12,
        20,
        40,
        74,
        148,
        284,
    ]
    t9 = sp.aperiodic_templates(9)
    assert t9[:3] == ["000000001", "000000011", "000000101"]
    assert t9[-1] == "111111110"
    assert t9 == sorted(t9)


# 2.8 Overlapping template matching ------------------------------------------


def test_overlapping_probabilities_reproduce_the_rev1a_table():
    """The corrected pi_i of section 2.8.4 (Hamano & Kaneko) are exact class
    probabilities; an independent dynamic program reproduces every printed digit."""
    exact = sp.overlapping_template_probabilities(9, 1032, 5)
    for printed, value in zip(sp.OVERLAPPING_PI_REV1A, exact, strict=True):
        digits = 7 if printed < 0.1 else 6  # the table prints 0.0704323
        assert round(value, digits) == printed
    assert sum(exact) == pytest.approx(1.0, abs=1e-12)


# 2.9 Maurer's universal test --------------------------------------------------


def test_universal_20_bits():
    """Section 2.9.4 with L = 2, Q = 4: K = 6 and f_n = 1.1949875.

    The printed P = 0.767189 uses sigma = sqrt(variance(L)), without the
    c(L, K) / sqrt(K) factor of the test's actual formula (which STS uses and
    so do we). Both are checked from our statistics.
    """
    r = sp.universal(bits("01011010011101010111"), block_length=2, init_blocks=4)
    s = r.statistics
    assert s["test_blocks"] == 6
    assert s["fn"] == pytest.approx(1.1949875, abs=1e-7)
    assert special.erfc(abs(s["fn"] - 1.5374383) / (math.sqrt(2) * math.sqrt(1.338))) == approx(
        0.767189
    )
    sigma = s["c"] * math.sqrt(1.338 / 6)
    assert r.p_values == [approx(special.erfc(abs(s["fn"] - 1.5374383) / (math.sqrt(2) * sigma)))]


# 2.10 Linear complexity ------------------------------------------------------


def test_berlekamp_massey_13_bits():
    assert sp.berlekamp_massey(int(c) for c in "1101011110001") == 4


# 2.11 Serial ---------------------------------------------------------------


def test_serial_10_bits():
    r = sp.serial(bits("0011011101"), block_length=3)
    assert r.statistics["psi_squared"] == [approx(2.8), approx(1.2), approx(0.4)]
    assert r.statistics["delta1"] == approx(1.6)
    assert r.statistics["delta2"] == approx(0.8)
    assert r.labels == ["delta1", "delta2"]
    assert r.p_values == [approx(0.808792), approx(0.670320)]


# 2.12 Approximate entropy ---------------------------------------------------


def test_approximate_entropy_10_bits():
    r = sp.approximate_entropy(bits("0100110101"), block_length=3)
    assert r.statistics["phi_m"] == approx(-1.643418)
    assert r.statistics["phi_m_plus_1"] == approx(-1.834372)
    assert r.statistics["apen"] == approx(0.190954)
    assert r.statistics["chi_square"] == approx(10.043859)
    assert r.p_values == [approx(0.261961)]


def test_approximate_entropy_100_bits(pi_100):
    r = sp.approximate_entropy(pi_100, block_length=2)
    assert r.statistics["apen"] == approx(0.665393)
    assert r.statistics["chi_square"] == approx(5.550792)
    assert r.p_values == [approx(0.235301)]


# 2.13 Cumulative sums -------------------------------------------------------


def test_cumulative_sums_10_bits():
    r = sp.cumulative_sums(bits("1011010111"))
    assert r.statistics["z_forward"] == 4
    assert r.p_values[0] == pytest.approx(0.4116588, abs=1e-6)


def test_cumulative_sums_100_bits(pi_100):
    r = sp.cumulative_sums(pi_100)
    assert (r.statistics["z_forward"], r.statistics["z_backward"]) == (16, 19)
    assert r.labels == ["forward", "backward"]
    assert r.p_values == [approx(0.219194), approx(0.114866)]


# 2.14 / 2.15 Random excursions ------------------------------------------------


def test_random_excursions_10_bits():
    """Section 2.14.4: J = 3 cycles; state +1 is visited 0, 1 and 3 times.

    By hand chi2 = 1/6 + 1/12 + 3/8 + 169/48 + 3/32 + 3/32 = 13/3; the spec
    prints 4.333033 (a transposed digit) and the P = 0.502529 computed from
    it. We pin the exact value.
    """
    r = sp.random_excursions(bits("0110110101"), min_cycles=1)
    assert r.statistics["cycles"] == 3
    assert r.statistics["counts"]["x=+1"] == [1, 1, 0, 1, 0, 0]
    i = r.labels.index("x=+1")
    assert r.statistics["chi_square"][i] == pytest.approx(13 / 3, rel=1e-12)
    assert r.p_values[i] == pytest.approx(float(special.gammaincc(2.5, 13 / 6)), rel=1e-12)
    assert r.p_values[i] == pytest.approx(0.502529, abs=1e-4)


def test_random_excursions_variant_10_bits():
    r = sp.random_excursions_variant(bits("0110110101"), min_cycles=1)
    i = r.labels.index("x=+1")
    assert r.statistics["cycles"] == 3
    assert r.statistics["visits"][i] == 4
    assert r.p_values[i] == approx(0.683091)
