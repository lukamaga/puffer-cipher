"""SP 800-22 section 4.2: proportion of passing sequences and p-value uniformity."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import special

from puffer.analysis import sp800_22 as sp
from puffer.analysis.bits import split_sequences

FAST_TESTS = ["frequency", "block_frequency", "cumulative_sums", "runs"]


def test_random_sequences_pass_proportion_and_uniformity(rng):
    m = 100
    stream = rng.integers(0, 2, size=m * 20_000, dtype=np.uint8)
    result = sp.run_multi(split_sequences(stream, 20_000), tests=FAST_TESTS)
    assert result.n_sequences == m
    assert result.n_bits == 20_000
    assert [s.name for s in result.subtests] == [
        "frequency",
        "block_frequency",
        "cumulative_sums[forward]",
        "cumulative_sums[backward]",
        "runs",
    ]
    half = 3 * math.sqrt(0.99 * 0.01 / m)
    for sub in result.subtests:
        assert sub.sequences == m
        assert sub.proportion_interval == pytest.approx((0.99 - half, 0.99 + half))
        assert sum(sub.histogram) == m
        assert sub.uniform is True
        assert sub.passed
    assert result.passed
    assert result.flagged == []


def test_summary_statistics_follow_section_4_2(rng):
    stream = rng.integers(0, 2, size=60 * 5_000, dtype=np.uint8)
    result = sp.run_multi(split_sequences(stream, 5_000), tests=["frequency"])
    (sub,) = result.subtests
    p_values = [s["frequency"].p_values[0] for s in result.suites]
    assert sub.passed_sequences == sum(p >= 0.01 for p in p_values)
    assert sub.proportion == sub.passed_sequences / 60
    histogram = [
        sum(int(10 * p) == i or (i == 9 and p == 1.0) for p in p_values) for i in range(10)
    ]
    assert list(sub.histogram) == histogram
    chi2 = sum((f - 6.0) ** 2 / 6.0 for f in histogram)
    assert sub.uniformity_p == pytest.approx(float(special.gammaincc(4.5, chi2 / 2)))


def test_biased_sequences_are_flagged_by_proportion(rng):
    stream = (rng.random(60 * 20_000) < 0.52).astype(np.uint8)
    result = sp.run_multi(split_sequences(stream, 20_000), tests=["frequency", "runs"])
    flagged = {s.name for s in result.flagged}
    assert "frequency" in flagged
    frequency = next(s for s in result.subtests if s.name == "frequency")
    assert not frequency.proportion_ok
    assert frequency.proportion < frequency.proportion_interval[0]
    assert not result.passed


def test_skewed_p_values_are_flagged_by_uniformity():
    """Non-uniform but mostly passing p-values: only the uniformity check catches it."""
    sub = sp.SubtestSummary(
        test="frequency",
        label="",
        sequences=100,
        passed_sequences=100,
        proportion_interval=(0.960150, 1.019850),
        histogram=(0, 0, 0, 0, 0, 0, 0, 0, 50, 50),
        uniformity_p=float(special.gammaincc(4.5, 410.0 / 2)),
    )
    assert sub.proportion_ok
    assert sub.uniform is False
    assert not sub.passed


def test_proportion_above_the_interval_is_suspicious_too():
    # 1000/1000 passing at alpha = 0.01 happens with probability 0.99^1000 = 4e-5.
    half = 3 * math.sqrt(0.99 * 0.01 / 1000)
    sub = sp.SubtestSummary("runs", "", 1000, 1000, (0.99 - half, 0.99 + half), (100,) * 10, 1.0)
    assert not sub.proportion_ok


def test_uniformity_is_not_assessed_below_55_sequences(rng):
    stream = rng.integers(0, 2, size=20 * 1_000, dtype=np.uint8)
    result = sp.run_multi(split_sequences(stream, 1_000), tests=["frequency"])
    (sub,) = result.subtests
    assert sub.sequences == 20
    assert sub.uniform is None
    assert sub.passed == sub.proportion_ok


def test_sub_tests_count_only_sequences_where_the_test_applied(rng):
    sequences = [rng.integers(0, 2, size=2_000, dtype=np.uint8) for _ in range(5)]
    result = sp.run_multi(sequences, tests=["frequency", "universal"])
    assert [s.name for s in result.subtests] == ["frequency"]  # universal skipped everywhere
    assert all(not s["universal"].applicable for s in result.suites)


def test_run_multi_validates_input(rng):
    with pytest.raises(ValueError, match="at least one sequence"):
        sp.run_multi([], tests=["frequency"])
    with pytest.raises(ValueError, match="equal length"):
        sp.run_multi([np.zeros(100, np.uint8), np.zeros(200, np.uint8)], tests=["frequency"])


def test_multi_result_is_json_serialisable(rng):
    stream = rng.integers(0, 2, size=10 * 1_000, dtype=np.uint8)
    result = sp.run_multi(split_sequences(stream, 1_000), tests=FAST_TESTS)
    decoded = json.loads(json.dumps(result.to_dict(include_suites=True)))
    assert decoded["n_sequences"] == 10
    assert len(decoded["suites"]) == 10
    assert decoded["subtests"][2]["name"] == "cumulative_sums[forward]"


@pytest.mark.slow
def test_full_suite_over_many_random_sequences(rng):
    """All 188 sub-tests over 100 random 10^6-bit sequences (NIST's usual setup)."""
    stream = rng.integers(0, 2, size=100 * 1_000_000, dtype=np.uint8)
    result = sp.run_multi(split_sequences(stream, 1_000_000))
    names = {s.test for s in result.subtests}
    assert names == set(sp.SUITE_TESTS)
    assert len(result.subtests) == 188
    excursion = [s for s in result.subtests if s.test == "random_excursions"]
    assert all(s.sequences < 100 for s in excursion)  # J >= 500 holds for ~60 % of sequences
    # Each sub-test's 3-sigma interval is exceeded with probability ~0.2 %, so
    # among 188 a stray flag or two is expected; many would mean a bug.
    assert len(result.flagged) <= 3
