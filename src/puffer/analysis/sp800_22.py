"""NIST SP 800-22 Rev. 1a statistical test suite for random bit sequences.

Why an in-process implementation
--------------------------------
The report tooling compares raw channel bits, conditioned test streams and
PufferRNG output. Shelling out to NIST STS for that means scraping text
files; implementing the 15 tests on numpy arrays gives structured results and
runs the full suite on a 1,000,000-bit sequence in about a second, so the
dashboard and CI can afford it.

What a pass means (DESIGN.md rule 4)
------------------------------------
Passing SP 800-22 is a *necessary* property of a good generator and never
evidence of entropy: a counter hashed with BLAKE2b passes every test here.
Results are only meaningful on data whose provenance the caller reports.
These are statistics, not cryptography; nothing in this module touches keys.

Parameterisation
----------------
Defaults are the reference parameters of NIST STS 2.1.2 (the tool that
produced the published reference results). Where SP 800-22 rev1a and the STS
code disagree the choice is deliberate and documented on the test:

* overlapping template matching uses the corrected rev1a probabilities
  (Hamano & Kaneko 2007). STS 2.1.2 still computes the older compound-Poisson
  approximation, and the reference p-values printed in SP 800-22 for this test
  were produced with it; the validation tests show both.
* linear complexity uses the rev1a class probabilities; STS 2.1.2 carries a
  typo (pi_0 = 0.01047 instead of 0.010417) that the worked example inherits.
* the longest-run test uses every complete block, ``N = n // M``, as STS does
  (the spec's N = 16/49/75 are the block counts at the minimum lengths).

Conventions
-----------
Every test takes a 1-D 0/1 array (see :mod:`puffer.analysis.bits`), its
parameters as keyword-only arguments with reference defaults, and ``alpha``.
A test returns ``skipped`` only when it is undefined for the input (e.g. too
few cycles for the random excursion tests). The *recommended* minimum input
sizes of SP 800-22 sections 2.x.7 are applied one level up, in
:func:`run_suite`, so that the worked examples of the specification (often
10-bit sequences) can still be reproduced test by test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
from scipy import special

from puffer.analysis.bits import BitArray, as_bits

DEFAULT_ALPHA = 0.01
"""Significance level recommended by SP 800-22 section 4.2."""

UNIFORMITY_THRESHOLD = 0.0001
"""P-values of sub-tests are considered uniform if P_T >= this (section 4.2.2)."""

MIN_SEQUENCES_FOR_UNIFORMITY = 55
"""Fewest sequences for which the 10-bin uniformity check is assessed (section 4.2.2)."""

OVERLAPPING_PI_REV1A: tuple[float, ...] = (
    0.364091,
    0.185659,
    0.139381,
    0.100571,
    0.0704323,
    0.139865,
)
"""Rev1a class probabilities for m = 9, M = 1032, K = 5 (section 2.8.4)."""

LINEAR_COMPLEXITY_PI: tuple[float, ...] = (
    0.010417,
    0.03125,
    0.125,
    0.5,
    0.25,
    0.0625,
    0.020833,
)
"""Class probabilities of the linear complexity statistic T (section 2.10.4)."""

_SQRT2 = math.sqrt(2.0)
_MAX_MATRIX_COLS = 64  # rows are packed into one uint64 for GF(2) elimination
_MAX_PATTERN_BITS = 24  # 2**m counters per pattern table

# Maurer's universal test, section 2.9.4: expected value and variance of f_n
# for L = 1..16 (index 0 unused).
_UNIVERSAL_EXPECTED = (
    0.0,
    0.7326495,
    1.5374383,
    2.4016068,
    3.3112247,
    4.2534266,
    5.2177052,
    6.1962507,
    7.1836656,
    8.1764248,
    9.1723243,
    10.170032,
    11.168765,
    12.168070,
    13.167693,
    14.167488,
    15.167379,
)
_UNIVERSAL_VARIANCE = (
    0.0,
    0.690,
    1.338,
    1.901,
    2.358,
    2.705,
    2.954,
    3.125,
    3.238,
    3.311,
    3.356,
    3.384,
    3.401,
    3.410,
    3.416,
    3.419,
    3.421,
)


class _LongestRunParams(NamedTuple):
    min_n: int
    block_size: int
    v_min: int  # runs <= v_min fall into class 0
    v_max: int  # runs >= v_max fall into class K
    probabilities: tuple[float, ...]


# Section 2.4.4 / STS 2.1.2, largest applicable entry first.
_LONGEST_RUN_TABLE = (
    _LongestRunParams(
        750_000, 10_000, 10, 16, (0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727)
    ),
    _LongestRunParams(
        6_272,
        128,
        4,
        9,
        (0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847),
    ),
    _LongestRunParams(128, 8, 1, 4, (0.21484375, 0.3671875, 0.23046875, 0.1875)),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of one SP 800-22 test on one sequence.

    ``p_values`` has one entry per sub-test: one for most tests, two for the
    cumulative sums (forward, backward) and serial tests, one per template for
    non-overlapping template matching, one per state for the random excursion
    tests. ``labels`` names those sub-tests in the same order and is empty for
    single-valued tests.

    ``passed`` applies the decision rule of section 4.2 to *every* p-value:
    it is true iff the test ran and all p-values are >= alpha. With 188
    p-values per sequence some sub-test of a perfect generator falls below
    0.01 most of the time, which is why :class:`SuiteResult` reports the
    expected number of chance failures and why the multi-sequence analysis of
    :func:`run_multi` is the real verdict. A skipped test has no p-values and
    ``passed`` is false: "not applicable" is never counted as a pass.
    """

    __test__ = False  # imported by test modules; keep pytest from collecting it

    name: str
    p_values: list[float]
    passed: bool
    statistics: dict[str, Any]
    skipped: str | None = None
    labels: list[str] = field(default_factory=list)

    @property
    def applicable(self) -> bool:
        """Whether the test ran (``skipped`` is ``None``)."""
        return self.skipped is None

    def subtests(self) -> list[tuple[str, float]]:
        """``(qualified name, p-value)`` pairs, e.g. ``("serial[delta2]", 0.46)``."""
        if not self.labels:
            return [(self.name, p) for p in self.p_values]
        return [
            (f"{self.name}[{label}]", p)
            for label, p in zip(self.labels, self.p_values, strict=True)
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation (statistics are plain Python types)."""
        return {
            "name": self.name,
            "p_values": list(self.p_values),
            "labels": list(self.labels),
            "passed": self.passed,
            "skipped": self.skipped,
            "statistics": self.statistics,
        }


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """All SP 800-22 tests run on one sequence, in NIST STS order."""

    n_bits: int
    alpha: float
    results: list[TestResult]

    def __getitem__(self, name: str) -> TestResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)

    @property
    def applicable(self) -> list[TestResult]:
        return [r for r in self.results if r.applicable]

    @property
    def skipped(self) -> list[TestResult]:
        return [r for r in self.results if not r.applicable]

    @property
    def passed(self) -> bool:
        """Every applicable test passed (strict: every p-value >= alpha)."""
        return all(r.passed for r in self.applicable)

    @property
    def p_value_count(self) -> int:
        return sum(len(r.p_values) for r in self.applicable)

    @property
    def failed_count(self) -> int:
        """Number of individual p-values below alpha."""
        return sum(p < self.alpha for r in self.applicable for p in r.p_values)

    @property
    def expected_false_failures(self) -> float:
        """Failures expected from a perfect generator: ``alpha * p_value_count``."""
        return self.alpha * self.p_value_count

    def failures_tail_probability(self) -> float:
        """``P[Binomial(p_value_count, alpha) >= failed_count]``.

        How surprising the observed number of failing p-values would be for
        an ideal generator, treating the p-values as independent (they are
        not quite: e.g. templates sharing bits are weakly correlated). A
        small value means the failures are not just the expected 1 in 100.
        """
        if self.failed_count == 0:
            return 1.0
        # bdtrc(k, n, p) = P[X > k]
        return float(special.bdtrc(self.failed_count - 1, self.p_value_count, self.alpha))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bits": self.n_bits,
            "alpha": self.alpha,
            "passed": self.passed,
            "p_value_count": self.p_value_count,
            "failed_count": self.failed_count,
            "expected_false_failures": self.expected_false_failures,
            "failures_tail_probability": self.failures_tail_probability(),
            "results": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True, slots=True)
class SubtestSummary:
    """Section 4.2 analysis of one sub-test (one p-value index) over many sequences."""

    test: str
    label: str
    sequences: int  # sequences for which the sub-test was applicable
    passed_sequences: int  # of those, sequences with p >= alpha
    proportion_interval: tuple[float, float]
    histogram: tuple[int, ...]  # p-values per decile [0, 0.1), ..., [0.9, 1.0]
    uniformity_p: float  # P_T of the 10-bin chi-square

    @property
    def name(self) -> str:
        return f"{self.test}[{self.label}]" if self.label else self.test

    @property
    def proportion(self) -> float:
        return self.passed_sequences / self.sequences

    @property
    def proportion_ok(self) -> bool:
        """Proportion inside ``(1-a) +- 3 sqrt(a(1-a)/m)`` (both sides, section 4.2.1)."""
        lo, hi = self.proportion_interval
        return lo <= self.proportion <= hi

    @property
    def uniform(self) -> bool | None:
        """``P_T >= 0.0001``; ``None`` when fewer than 55 sequences make it meaningless."""
        if self.sequences < MIN_SEQUENCES_FOR_UNIFORMITY:
            return None
        return self.uniformity_p >= UNIFORMITY_THRESHOLD

    @property
    def passed(self) -> bool:
        return self.proportion_ok and self.uniform is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "test": self.test,
            "label": self.label,
            "sequences": self.sequences,
            "passed_sequences": self.passed_sequences,
            "proportion": self.proportion,
            "proportion_interval": list(self.proportion_interval),
            "proportion_ok": self.proportion_ok,
            "histogram": list(self.histogram),
            "uniformity_p": self.uniformity_p,
            "uniform": self.uniform,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class MultiSuiteResult:
    """SP 800-22 section 4.2 analysis of ``m`` sequences of equal length."""

    n_bits: int
    alpha: float
    suites: list[SuiteResult]
    subtests: list[SubtestSummary]

    @property
    def n_sequences(self) -> int:
        return len(self.suites)

    @property
    def flagged(self) -> list[SubtestSummary]:
        """Sub-tests whose proportion or uniformity indicates non-randomness."""
        return [s for s in self.subtests if not s.passed]

    @property
    def passed(self) -> bool:
        return not self.flagged

    def to_dict(self, *, include_suites: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_bits": self.n_bits,
            "alpha": self.alpha,
            "n_sequences": self.n_sequences,
            "passed": self.passed,
            "flagged": [s.name for s in self.flagged],
            "subtests": [s.to_dict() for s in self.subtests],
        }
        if include_suites:
            out["suites"] = [s.to_dict() for s in self.suites]
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finish(
    name: str,
    p_values: Iterable[float],
    statistics: dict[str, Any],
    alpha: float,
    labels: Sequence[str] = (),
) -> TestResult:
    _require(0.0 < alpha < 1.0, "alpha must be in (0, 1)")
    # Series such as the cumulative-sums formula can land a rounding error
    # outside [0, 1]; clip so consumers can rely on the range.
    ps = [min(1.0, max(0.0, float(p))) for p in p_values]
    return TestResult(name, ps, all(p >= alpha for p in ps), statistics, None, list(labels))


def _skip(name: str, reason: str, statistics: dict[str, Any] | None = None) -> TestResult:
    return TestResult(name, [], False, statistics or {}, reason)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _igamc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (STS ``cephes_igamc``)."""
    return float(special.gammaincc(a, x))


def _erfc(x: float) -> float:
    return float(special.erfc(x))


def _chi_square(observed: npt.ArrayLike, expected: npt.ArrayLike) -> float:
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    return float(np.sum((obs - exp) ** 2 / exp))


def _check_probabilities(probabilities: Sequence[float], classes: int, what: str) -> None:
    _require(len(probabilities) == classes, f"{what}: expected {classes} class probabilities")
    _require(all(p > 0 for p in probabilities), f"{what}: probabilities must be positive")
    _require(abs(sum(probabilities) - 1.0) < 1e-4, f"{what}: probabilities must sum to 1")


def _window_values(blocks: npt.NDArray[np.uint8], m: int) -> npt.NDArray[np.int64]:
    """Integer value of every m-bit window that lies inside a row of ``blocks``.

    ``blocks`` is ``(N, M)``; the result is ``(N, M - m + 1)`` with the first
    bit as the most significant, matching how the spec writes templates.
    """
    width = blocks.shape[1] - m + 1
    values = np.zeros((blocks.shape[0], width), dtype=np.int64)
    for j in range(m):
        values <<= 1
        values |= blocks[:, j : j + width]
    return values


def _cyclic_pattern_counts(eps: BitArray, m: int) -> npt.NDArray[np.int64]:
    """Counts of all 2**m overlapping m-bit patterns, wrapping around the end.

    Wrapping (appending the first m-1 bits, sections 2.11 and 2.12) makes
    every position start a window, so counts of shorter patterns are exact
    marginals of these: ``counts.reshape(-1, 2).sum(axis=1)`` gives m - 1.
    """
    ext = np.concatenate([eps, np.resize(eps, m - 1)])
    values = _window_values(ext[np.newaxis, :], m)[0]
    return np.bincount(values, minlength=1 << m)


def _random_walk(eps: BitArray) -> npt.NDArray[np.int64]:
    """Partial sums S_k of X_i = 2*eps_i - 1."""
    return np.cumsum(2 * eps.astype(np.int64) - 1)


# ---------------------------------------------------------------------------
# 2.1 Frequency (monobit)
# ---------------------------------------------------------------------------


def frequency(bits: npt.ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Section 2.1: is the proportion of ones close to 1/2?

    ``S_n = sum(2 eps - 1)``, ``s_obs = |S_n| / sqrt(n)``,
    ``P = erfc(s_obs / sqrt 2)``. Every other test presumes this one passes.
    """
    name = "frequency"
    eps = as_bits(bits)
    n = int(eps.size)
    if n == 0:
        return _skip(name, "empty sequence")
    s_n = 2 * int(np.count_nonzero(eps)) - n
    s_obs = abs(s_n) / math.sqrt(n)
    return _finish(name, [_erfc(s_obs / _SQRT2)], {"n": n, "sum": s_n, "s_obs": s_obs}, alpha)


# ---------------------------------------------------------------------------
# 2.2 Frequency within a block
# ---------------------------------------------------------------------------


def block_frequency(
    bits: npt.ArrayLike, *, block_size: int = 128, alpha: float = DEFAULT_ALPHA
) -> TestResult:
    """Section 2.2: proportion of ones in each of ``N = n // M`` blocks.

    ``chi2 = 4 M sum (pi_i - 1/2)^2``, ``P = igamc(N/2, chi2/2)``. Catches
    local bias that cancels out over the whole sequence. M = 128 is the STS
    default.
    """
    name = "block_frequency"
    _require(block_size > 0, "block_size must be positive")
    eps = as_bits(bits)
    n = int(eps.size)
    n_blocks = n // block_size
    if n_blocks == 0:
        return _skip(name, f"needs at least one {block_size}-bit block, got n={n}")
    ones = eps[: n_blocks * block_size].reshape(n_blocks, block_size).sum(axis=1, dtype=np.int64)
    chi2 = float(4.0 * block_size * np.sum((ones / block_size - 0.5) ** 2))
    statistics = {"n": n, "block_size": block_size, "blocks": n_blocks, "chi_square": chi2}
    return _finish(name, [_igamc(n_blocks / 2.0, chi2 / 2.0)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.3 Runs
# ---------------------------------------------------------------------------


def runs(bits: npt.ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Section 2.3: is the number of runs (maximal blocks of equal bits) as expected?

    ``V = 1 + #{k : eps_k != eps_k+1}``,
    ``P = erfc(|V - 2 n pi (1-pi)| / (2 sqrt(2n) pi (1-pi)))``.

    The statistic is only defined for nearly balanced sequences: if the
    frequency prerequisite ``|pi - 1/2| <= 2/sqrt(n)`` fails, the spec sets
    ``P = 0`` (a failure, not a skip - the sequence is demonstrably biased).
    """
    name = "runs"
    eps = as_bits(bits)
    n = int(eps.size)
    if n < 2:
        return _skip(name, f"needs at least 2 bits, got n={n}")
    pi = int(np.count_nonzero(eps)) / n
    tau = 2.0 / math.sqrt(n)
    statistics: dict[str, Any] = {"n": n, "pi": pi, "tau": tau}
    spread = pi * (1.0 - pi)
    if abs(pi - 0.5) > tau or spread == 0.0:
        statistics.update(prerequisite_passed=False, runs=None)
        return _finish(name, [0.0], statistics, alpha)
    v_obs = 1 + int(np.count_nonzero(eps[1:] != eps[:-1]))
    p = _erfc(abs(v_obs - 2.0 * n * spread) / (2.0 * math.sqrt(2.0 * n) * spread))
    statistics.update(prerequisite_passed=True, runs=v_obs)
    return _finish(name, [p], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.4 Longest run of ones in a block
# ---------------------------------------------------------------------------


def _longest_runs(blocks: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    """Longest run of ones in each row, via run boundaries of the zero-padded rows."""
    n_rows, width = blocks.shape
    padded = np.zeros((n_rows, width + 2), dtype=np.int8)
    padded[:, 1:-1] = blocks
    edges = np.diff(padded, axis=1)
    # nonzero() walks row-major, so the k-th start and k-th end of a row pair up.
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    longest = np.zeros(n_rows, dtype=np.int64)
    np.maximum.at(longest, rows, ends - starts)
    return longest


def longest_run_of_ones(bits: npt.ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Section 2.4: distribution of the longest run of ones per block.

    The block size M (8, 128 or 10,000) and class probabilities come from the
    spec table for the sequence length, and ``N = n // M`` blocks are used.
    ``P = igamc(K/2, chi2/2)``. Needs n >= 128.
    """
    name = "longest_run_of_ones"
    eps = as_bits(bits)
    n = int(eps.size)
    params = next((t for t in _LONGEST_RUN_TABLE if n >= t.min_n), None)
    if params is None:
        return _skip(name, f"needs n >= 128 bits, got n={n}")
    m = params.block_size
    n_blocks = n // m
    longest = _longest_runs(eps[: n_blocks * m].reshape(n_blocks, m))
    k = len(params.probabilities) - 1
    classes = np.clip(longest, params.v_min, params.v_max) - params.v_min
    counts = np.bincount(classes, minlength=k + 1)
    chi2 = _chi_square(counts, n_blocks * np.asarray(params.probabilities))
    statistics = {
        "n": n,
        "block_size": m,
        "blocks": n_blocks,
        "class_bounds": [params.v_min, params.v_max],
        "counts": counts.tolist(),
        "chi_square": chi2,
    }
    return _finish(name, [_igamc(k / 2.0, chi2 / 2.0)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.5 Binary matrix rank
# ---------------------------------------------------------------------------


def _rank_probability(rank: int, rows: int, cols: int) -> float:
    """P[rank of a random rows x cols GF(2) matrix == rank] (section 3.5)."""
    product = 1.0
    for i in range(rank):
        product *= (1.0 - 2.0 ** (i - rows)) * (1.0 - 2.0 ** (i - cols)) / (1.0 - 2.0 ** (i - rank))
    return math.ldexp(product, rank * (rows + cols - rank) - rows * cols)


def gf2_rank(matrices: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Rank over GF(2) of each matrix in a ``(N, rows, cols)`` 0/1 array.

    Rows are packed into ``uint64`` words and Gauss-Jordan elimination runs
    on all N matrices at once, one column per step, so the cost is ``cols``
    vectorised passes regardless of N.
    """
    mats = np.asarray(matrices)
    _require(mats.ndim == 3, "expected an (N, rows, cols) array")
    n_mats, n_rows, n_cols = mats.shape
    _require(n_cols <= _MAX_MATRIX_COLS, f"at most {_MAX_MATRIX_COLS} columns are supported")
    weights = np.left_shift(np.uint64(1), np.arange(n_cols - 1, -1, -1, dtype=np.uint64))
    rows = (mats.astype(np.uint64) * weights).sum(axis=2, dtype=np.uint64)
    rank = np.zeros(n_mats, dtype=np.int64)
    unused = np.ones((n_mats, n_rows), dtype=bool)
    index = np.arange(n_mats)
    for col in range(n_cols):
        bit = np.uint64(1) << np.uint64(n_cols - 1 - col)
        has_bit = (rows & bit) != 0
        candidates = has_bit & unused
        found = candidates.any(axis=1)
        pivot = np.argmax(candidates, axis=1)
        pivot_rows = rows[index, pivot]
        eliminate = has_bit & found[:, np.newaxis]
        eliminate[index, pivot] = False
        rows ^= np.where(eliminate, pivot_rows[:, np.newaxis], np.uint64(0))
        unused[index[found], pivot[found]] = False
        rank += found
    return rank


def binary_matrix_rank(
    bits: npt.ArrayLike, *, rows: int = 32, cols: int = 32, alpha: float = DEFAULT_ALPHA
) -> TestResult:
    """Section 2.5: rank distribution of disjoint ``rows x cols`` sub-matrices.

    Ranks are binned as full, full - 1 and lower, compared with the exact
    probabilities for random GF(2) matrices, ``P = igamc(1, chi2/2)``. Detects
    linear dependence among fixed-length substrings (e.g. LFSR output).
    Matrices are filled row by row.
    """
    name = "binary_matrix_rank"
    # With a dimension of 1 the "rank <= full - 2" class is empty.
    _require(rows >= 2 and cols >= 2, "matrix dimensions must be at least 2")
    eps = as_bits(bits)
    n = int(eps.size)
    n_mats = n // (rows * cols)
    if n_mats == 0:
        return _skip(
            name, f"needs at least one {rows}x{cols} matrix ({rows * cols} bits), got n={n}"
        )
    full = min(rows, cols)
    ranks = gf2_rank(eps[: n_mats * rows * cols].reshape(n_mats, rows, cols))
    p_full = _rank_probability(full, rows, cols)
    p_minus = _rank_probability(full - 1, rows, cols)
    probabilities = [p_full, p_minus, 1.0 - p_full - p_minus]
    f_full = int(np.count_nonzero(ranks == full))
    f_minus = int(np.count_nonzero(ranks == full - 1))
    counts = [f_full, f_minus, n_mats - f_full - f_minus]
    chi2 = _chi_square(counts, n_mats * np.asarray(probabilities))
    statistics = {
        "n": n,
        "rows": rows,
        "cols": cols,
        "matrices": n_mats,
        "counts": counts,
        "probabilities": probabilities,
        "chi_square": chi2,
    }
    return _finish(name, [_igamc(1.0, chi2 / 2.0)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.6 Discrete Fourier transform (spectral)
# ---------------------------------------------------------------------------


def discrete_fourier_transform(bits: npt.ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Section 2.6: are there too many spectral peaks (periodic features)?

    Counts the DFT magnitudes of the first n/2 frequencies below the 95%
    threshold ``T = sqrt(ln(1/0.05) n)``, expected ``N0 = 0.95 n / 2``, and
    ``d = (N1 - N0) / sqrt(n 0.95 0.05 / 4)``, ``P = erfc(|d| / sqrt 2)``.
    These are the rev1a corrections (Kim et al. 2004) of the original
    ``sqrt(3n)`` threshold and wrong variance; STS 2.1.2 uses them too.
    """
    name = "discrete_fourier_transform"
    eps = as_bits(bits)
    n = int(eps.size)
    if n < 2:
        return _skip(name, f"needs at least 2 bits, got n={n}")
    magnitudes = np.abs(np.fft.rfft(2.0 * eps - 1.0))[: n // 2]
    threshold = math.sqrt(math.log(1.0 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = int(np.count_nonzero(magnitudes < threshold))
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    statistics = {"n": n, "threshold": threshold, "expected": n0, "observed": n1, "d": d}
    return _finish(name, [_erfc(abs(d) / _SQRT2)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.7 Non-overlapping template matching
# ---------------------------------------------------------------------------


def aperiodic_templates(length: int) -> list[str]:
    """All aperiodic templates of ``length`` bits, in ascending order.

    A template is aperiodic when no proper prefix equals the suffix of the
    same length, so two occurrences can never overlap. For m = 9 there are
    148, the set and order of NIST STS's ``templates/template9`` file.
    """
    _require(2 <= length <= _MAX_PATTERN_BITS, f"template length must be 2..{_MAX_PATTERN_BITS}")
    values = np.arange(1 << length, dtype=np.int64)
    aperiodic = np.ones(values.size, dtype=bool)
    for shift in range(1, length):
        overlap = length - shift
        aperiodic &= (values >> shift) != (values & ((1 << overlap) - 1))
    return [format(int(v), f"0{length}b") for v in values[aperiodic]]


def non_overlapping_template_matching(
    bits: npt.ArrayLike,
    *,
    template_length: int = 9,
    n_blocks: int = 8,
    templates: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> TestResult:
    """Section 2.7: occurrences of each aperiodic m-bit template in N blocks.

    For each template, ``W_j`` counts non-overlapping matches in block j
    (``M = n // N`` bits); with ``mu = (M - m + 1) / 2^m`` and
    ``sigma^2 = M (2^-m - (2m - 1) 2^-2m)``,
    ``chi2 = sum_j (W_j - mu)^2 / sigma^2``, ``P = igamc(N/2, chi2/2)``.
    One p-value per template; defaults are STS's m = 9, N = 8 with all 148
    templates.

    Because the templates are aperiodic, matches cannot overlap and the
    spec's scan-and-skip count equals the plain number of occurrences. That
    is what makes the test vectorisable: one histogram of all m-bit window
    values per block answers every template at once. Custom ``templates``
    must therefore be aperiodic too (the variance above assumes it anyway).
    """
    name = "non_overlapping_template_matching"
    m = template_length
    _require(2 <= m <= _MAX_PATTERN_BITS, f"template_length must be 2..{_MAX_PATTERN_BITS}")
    _require(n_blocks > 0, "n_blocks must be positive")
    if templates is None:
        chosen = aperiodic_templates(m)
    else:
        allowed = set(aperiodic_templates(m))
        chosen = list(templates)
        bad = [t for t in chosen if t not in allowed]
        _require(not bad, f"templates must be aperiodic {m}-bit strings, got {bad[:3]}")
        _require(bool(chosen), "templates must not be empty")
    eps = as_bits(bits)
    n = int(eps.size)
    block = n // n_blocks
    if block < m:
        return _skip(name, f"blocks of {block} bits are shorter than the {m}-bit template")
    values = _window_values(eps[: n_blocks * block].reshape(n_blocks, block), m)
    offsets = (np.arange(n_blocks, dtype=np.int64) << m)[:, np.newaxis]
    table = np.bincount((values + offsets).ravel(), minlength=n_blocks << m)
    columns = np.array([int(t, 2) for t in chosen], dtype=np.int64)
    matches = table.reshape(n_blocks, 1 << m)[:, columns]  # (N, templates)
    mu = (block - m + 1) / 2.0**m
    variance = block * (1.0 / 2.0**m - (2.0 * m - 1.0) / 2.0 ** (2 * m))
    chi2 = np.sum((matches - mu) ** 2, axis=0) / variance
    p_values = special.gammaincc(n_blocks / 2.0, chi2 / 2.0)
    statistics = {
        "n": n,
        "template_length": m,
        "blocks": n_blocks,
        "block_size": block,
        "mean": mu,
        "variance": variance,
        "matches": matches.T.tolist(),
        "chi_square": chi2.tolist(),
    }
    return _finish(name, p_values.tolist(), statistics, alpha, chosen)


# ---------------------------------------------------------------------------
# 2.8 Overlapping template matching
# ---------------------------------------------------------------------------


def overlapping_template_probabilities(
    template_length: int = 9, block_size: int = 1032, degrees_of_freedom: int = 5
) -> tuple[float, ...]:
    """Exact class probabilities for overlapping matches of ``1^m`` in M random bits.

    Dynamic programming over the M bits with state (current run of ones,
    capped at m - 1; matches so far, capped at K): an occurrence is counted
    whenever a one extends the run to m. Classes are 0, 1, ..., K-1 and >= K
    matches. For (9, 1032, 5) this reproduces the rev1a table
    :data:`OVERLAPPING_PI_REV1A`, which is how the tests validate those
    constants; for other parameters it supplies the probabilities the spec
    does not tabulate.
    """
    m, k = template_length, degrees_of_freedom
    _require(m >= 1 and block_size >= m and k >= 1, "need 1 <= m <= block_size and K >= 1")
    state = np.zeros((m, k + 1), dtype=np.float64)  # [trailing ones, matches so far]
    state[0, 0] = 1.0
    for _ in range(block_size):
        # A one after (at least) m - 1 ones completes a match. The run stays
        # at m - 1, so the next one matches again: occurrences overlap.
        hit = 0.5 * state[m - 1]
        nxt = np.zeros_like(state)
        nxt[0] = 0.5 * state.sum(axis=0)  # a zero resets the run
        nxt[1:] = 0.5 * state[:-1]  # a one extends a shorter run, no match yet
        nxt[m - 1, 1:] += hit[:-1]
        nxt[m - 1, k] += hit[k]  # match counts saturate at K
        state = nxt
    return tuple(float(p) for p in state.sum(axis=0))


def overlapping_template_matching(
    bits: npt.ArrayLike,
    *,
    template_length: int = 9,
    block_size: int = 1032,
    degrees_of_freedom: int = 5,
    probabilities: Sequence[float] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> TestResult:
    """Section 2.8: overlapping occurrences of ``1^m`` in N blocks of M bits.

    Blocks are binned by match count (0, 1, ..., K-1, >= K) and compared
    with the class probabilities, ``P = igamc(K/2, chi2/2)``.

    The default probabilities are the corrected rev1a values for the
    reference parameters m = 9, M = 1032, K = 5; other parameters get exact
    values from :func:`overlapping_template_probabilities`. ``probabilities``
    overrides both, e.g. to reproduce NIST STS 2.1.2, which still uses the
    pre-correction approximation.
    """
    name = "overlapping_template_matching"
    m, k = template_length, degrees_of_freedom
    _require(1 <= m <= _MAX_PATTERN_BITS and k >= 1, "invalid template length or K")
    if probabilities is None:
        if (m, block_size, k) == (9, 1032, 5):
            probabilities = OVERLAPPING_PI_REV1A
        else:
            probabilities = overlapping_template_probabilities(m, block_size, k)
    _check_probabilities(probabilities, k + 1, name)
    eps = as_bits(bits)
    n = int(eps.size)
    _require(block_size >= m, "block_size must be at least the template length")
    n_blocks = n // block_size
    if n_blocks == 0:
        return _skip(name, f"needs at least one {block_size}-bit block, got n={n}")
    values = _window_values(eps[: n_blocks * block_size].reshape(n_blocks, block_size), m)
    hits = np.count_nonzero(values == (1 << m) - 1, axis=1)
    counts = np.bincount(np.minimum(hits, k), minlength=k + 1)
    chi2 = _chi_square(counts, n_blocks * np.asarray(probabilities))
    statistics = {
        "n": n,
        "template_length": m,
        "block_size": block_size,
        "blocks": n_blocks,
        "counts": counts.tolist(),
        "probabilities": [float(p) for p in probabilities],
        "chi_square": chi2,
    }
    return _finish(name, [_igamc(k / 2.0, chi2 / 2.0)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.9 Maurer's universal statistical test
# ---------------------------------------------------------------------------


def _universal_block_length(n: int) -> int | None:
    """Largest L in 6..16 with n >= (Q + K) L, Q = 10 * 2^L, K = 1000 * 2^L.

    This is exactly the spec's table of minimum lengths (387,840 bits for
    L = 6, 904,960 for L = 7, ...). ``None`` below L = 6.
    """
    fitting = [length for length in range(6, 17) if n >= 1010 * length * (1 << length)]
    return fitting[-1] if fitting else None


def universal(
    bits: npt.ArrayLike,
    *,
    block_length: int | None = None,
    init_blocks: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> TestResult:
    """Section 2.9: Maurer's universal test - is the sequence compressible?

    The sequence is cut into L-bit blocks; Q initialise a table of last
    occurrences and for each of the remaining K blocks ``log2(distance to the
    previous occurrence)`` is averaged into ``f_n``.
    ``P = erfc(|f_n - E[L]| / (sqrt 2 sigma))`` with
    ``sigma = c(L, K) sqrt(Var[L] / K)`` (Coron & Naccache's correction,
    as in STS).

    By default L comes from the spec table for n and ``Q = 10 * 2^L``; the
    test is skipped below 387,840 bits, where STS refuses too. Explicit
    ``block_length``/``init_blocks`` (L in 1..16) allow the worked example.
    """
    name = "universal"
    eps = as_bits(bits)
    n = int(eps.size)
    if block_length is None:
        chosen = _universal_block_length(n)
        if chosen is None:
            return _skip(name, f"needs n >= 387,840 bits for L >= 6, got n={n}")
        block_length = chosen
    length = block_length
    _require(1 <= length <= 16, "block_length must be 1..16")
    q = 10 * (1 << length) if init_blocks is None else init_blocks
    _require(q > 0, "init_blocks must be positive")
    k = n // length - q
    if k <= 0:
        return _skip(name, f"no test blocks: n={n} gives at most {n // length} blocks of {length}")
    blocks = eps[: (q + k) * length].reshape(q + k, length)
    values = _window_values(blocks, length)[:, 0]
    # Previous occurrence of each block's value (1-based index, 0 = never):
    # a stable sort groups equal values with positions ascending.
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    previous_sorted = np.zeros(q + k, dtype=np.int64)
    same = sorted_values[1:] == sorted_values[:-1]
    previous_sorted[1:][same] = order[:-1][same] + 1
    previous = np.empty_like(previous_sorted)
    previous[order] = previous_sorted
    distance = np.arange(1, q + k + 1, dtype=np.int64) - previous
    fn = float(np.sum(np.log2(distance[q:]))) / k
    expected = _UNIVERSAL_EXPECTED[length]
    variance = _UNIVERSAL_VARIANCE[length]
    c = 0.7 - 0.8 / length + (4.0 + 32.0 / length) * k ** (-3.0 / length) / 15.0
    sigma = c * math.sqrt(variance / k)
    statistics = {
        "n": n,
        "block_length": length,
        "init_blocks": q,
        "test_blocks": k,
        "fn": fn,
        "expected": expected,
        "variance": variance,
        "c": c,
        "sigma": sigma,
    }
    p = _erfc(abs(fn - expected) / (_SQRT2 * sigma))
    return _finish(name, [p], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.10 Linear complexity
# ---------------------------------------------------------------------------


def berlekamp_massey(bits: Iterable[int]) -> int:
    """Linear complexity of a GF(2) sequence (length of the shortest LFSR).

    Polynomials are Python ints (bit i = coefficient of D^i) and the
    discrepancy is the parity of ``C & window``, where ``window`` holds the
    sequence reversed (bit i = s_{k-i}). Big-int AND/XOR/popcount on a few
    hundred bits is far cheaper than the textbook inner loop.
    """
    c = b = 1
    length = 0
    last = -1
    window = 0
    for k, bit in enumerate(bits):
        window = (window << 1) | bit
        if (c & window).bit_count() & 1:
            previous = c
            c ^= b << (k - last)
            if 2 * length <= k:
                length = k + 1 - length
                last = k
                b = previous
    return length


def linear_complexity(
    bits: npt.ArrayLike,
    *,
    block_size: int = 500,
    probabilities: Sequence[float] = LINEAR_COMPLEXITY_PI,
    alpha: float = DEFAULT_ALPHA,
) -> TestResult:
    """Section 2.10: are the LFSRs generating each M-bit block long enough?

    For each of ``N = n // M`` blocks the linear complexity L_i (Berlekamp-
    Massey) gives ``T_i = (-1)^M (L_i - mu) + 2/9`` with
    ``mu = M/2 + (9 + (-1)^(M+1))/36 - (M/3 + 2/9)/2^M``; T is binned at
    -2.5, -1.5, ..., 2.5 into K + 1 = 7 classes and ``P = igamc(K/2, chi2/2)``.
    """
    name = "linear_complexity"
    _require(block_size > 0, "block_size must be positive")
    _check_probabilities(probabilities, 7, name)
    eps = as_bits(bits)
    n = int(eps.size)
    n_blocks = n // block_size
    if n_blocks == 0:
        return _skip(name, f"needs at least one {block_size}-bit block, got n={n}")
    blocks = eps[: n_blocks * block_size].reshape(n_blocks, block_size).tolist()
    complexities = np.array([berlekamp_massey(row) for row in blocks], dtype=np.float64)
    m = block_size
    sign = -1.0 if m % 2 else 1.0  # (-1)^M
    mu = m / 2.0 + (9.0 - sign) / 36.0 - (m / 3.0 + 2.0 / 9.0) / 2.0**m
    t = sign * (complexities - mu) + 2.0 / 9.0
    classes = np.searchsorted([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], t, side="left")
    counts = np.bincount(classes, minlength=7)
    chi2 = _chi_square(counts, n_blocks * np.asarray(probabilities))
    statistics = {
        "n": n,
        "block_size": m,
        "blocks": n_blocks,
        "mean": mu,
        "counts": counts.tolist(),
        "probabilities": [float(p) for p in probabilities],
        "chi_square": chi2,
    }
    return _finish(name, [_igamc(3.0, chi2 / 2.0)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.11 Serial
# ---------------------------------------------------------------------------


def serial(
    bits: npt.ArrayLike, *, block_length: int = 16, alpha: float = DEFAULT_ALPHA
) -> TestResult:
    """Section 2.11: are all overlapping m-bit patterns equally frequent?

    ``psi2_m = 2^m / n * sum(count^2) - n`` over cyclic m-bit windows;
    ``delta1 = psi2_m - psi2_(m-1)``, ``delta2 = psi2_m - 2 psi2_(m-1) +
    psi2_(m-2)``, ``P1 = igamc(2^(m-2), delta1/2)``,
    ``P2 = igamc(2^(m-3), delta2/2)``. The numerators are summed as exact
    integers: psi2 is a small difference of two numbers of order n.
    """
    name = "serial"
    m = block_length
    _require(2 <= m <= _MAX_PATTERN_BITS, f"block_length must be 2..{_MAX_PATTERN_BITS}")
    eps = as_bits(bits)
    n = int(eps.size)
    if n == 0:
        return _skip(name, "empty sequence")
    counts = _cyclic_pattern_counts(eps, m)
    # squares[j] = sum of squared counts of (m - j)-bit patterns. Shorter
    # pattern counts are marginals of the cyclic m-bit counts. psi2 for
    # m - j = 0 is 0 by definition, which the general formula also yields.
    squares = [int(np.dot(counts, counts))]
    for _ in range(2):
        counts = counts.reshape(-1, 2).sum(axis=1)
        squares.append(int(np.dot(counts, counts)))
    weighted = [(1 << (m - j)) * sq for j, sq in enumerate(squares)]
    psi2 = [(w - n * n) / n for w in weighted]
    delta1 = (weighted[0] - weighted[1]) / n
    delta2 = (weighted[0] - 2 * weighted[1] + weighted[2]) / n
    p1 = _igamc(2.0 ** (m - 2), delta1 / 2.0)
    p2 = _igamc(2.0 ** (m - 3), delta2 / 2.0)
    statistics = {
        "n": n,
        "block_length": m,
        "psi_squared": psi2,
        "delta1": delta1,
        "delta2": delta2,
    }
    return _finish(name, [p1, p2], statistics, alpha, ("delta1", "delta2"))


# ---------------------------------------------------------------------------
# 2.12 Approximate entropy
# ---------------------------------------------------------------------------


def _phi(counts: npt.NDArray[np.int64], n: int) -> float:
    freq = counts[counts > 0] / n
    return float(np.sum(freq * np.log(freq)))


def approximate_entropy(
    bits: npt.ArrayLike, *, block_length: int = 10, alpha: float = DEFAULT_ALPHA
) -> TestResult:
    """Section 2.12: compare frequencies of overlapping m- and (m+1)-bit patterns.

    ``ApEn = phi(m) - phi(m+1)`` with ``phi(m) = sum pi_i ln pi_i`` over
    cyclic windows, ``chi2 = 2n (ln 2 - ApEn)``,
    ``P = igamc(2^(m-1), chi2/2)``.
    """
    name = "approximate_entropy"
    m = block_length
    _require(1 <= m < _MAX_PATTERN_BITS, f"block_length must be 1..{_MAX_PATTERN_BITS - 1}")
    eps = as_bits(bits)
    n = int(eps.size)
    if n == 0:
        return _skip(name, "empty sequence")
    counts_m1 = _cyclic_pattern_counts(eps, m + 1)
    phi_m1 = _phi(counts_m1, n)
    phi_m = _phi(counts_m1.reshape(-1, 2).sum(axis=1), n)
    apen = phi_m - phi_m1
    chi2 = 2.0 * n * (math.log(2.0) - apen)
    statistics = {
        "n": n,
        "block_length": m,
        "phi_m": phi_m,
        "phi_m_plus_1": phi_m1,
        "apen": apen,
        "chi_square": chi2,
    }
    return _finish(name, [_igamc(2.0 ** (m - 1), chi2 / 2.0)], statistics, alpha)


# ---------------------------------------------------------------------------
# 2.13 Cumulative sums
# ---------------------------------------------------------------------------


def _cusum_p_value(n: int, z: int) -> float:
    """P-value of section 2.13 for maximal excursion ``z`` (exact integer bounds)."""
    sqrt_n = math.sqrt(n)
    top = (n - z) // (4 * z)  # floor((n/z - 1) / 4)
    k1 = np.arange(-top, top + 1)  # ceil((-n/z + 1) / 4) == -top
    k2 = np.arange(-((n + 3 * z) // (4 * z)), top + 1)  # ceil((-n/z - 3) / 4)
    sum1 = np.sum(special.ndtr((4 * k1 + 1) * z / sqrt_n) - special.ndtr((4 * k1 - 1) * z / sqrt_n))
    sum2 = np.sum(special.ndtr((4 * k2 + 3) * z / sqrt_n) - special.ndtr((4 * k2 + 1) * z / sqrt_n))
    return float(1.0 - sum1 + sum2)


def cumulative_sums(bits: npt.ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Section 2.13: maximal excursion of the +-1 random walk, both directions.

    ``z = max_k |S_k|`` for the forward walk and for the walk over the
    reversed sequence; the p-value is the spec's alternating normal series.
    Two p-values, labelled ``forward`` and ``backward``.
    """
    name = "cumulative_sums"
    eps = as_bits(bits)
    n = int(eps.size)
    if n == 0:
        return _skip(name, "empty sequence")
    z_forward = int(np.max(np.abs(_random_walk(eps))))
    z_backward = int(np.max(np.abs(_random_walk(eps[::-1]))))
    statistics = {"n": n, "z_forward": z_forward, "z_backward": z_backward}
    p_values = [_cusum_p_value(n, z_forward), _cusum_p_value(n, z_backward)]
    return _finish(name, p_values, statistics, alpha, ("forward", "backward"))


# ---------------------------------------------------------------------------
# 2.14 / 2.15 Random excursions
# ---------------------------------------------------------------------------


def _excursion_cycles(eps: BitArray) -> tuple[npt.NDArray[np.int64], int]:
    """Random walk and its number of cycles J, counted as STS does.

    A cycle ends at each return to zero; a walk that does not end at zero
    contributes one final, incomplete cycle (the spec appends a zero).
    """
    walk = _random_walk(eps)
    cycles = int(np.count_nonzero(walk == 0)) + int(walk[-1] != 0)
    return walk, cycles


def _excursion_skip_reason(n: int, cycles: int, min_cycles: int) -> str | None:
    constraint = max(min_cycles, int(0.005 * math.sqrt(n)))
    if cycles < constraint:
        return f"only {cycles} cycles, the test needs J >= {constraint}"
    return None


def _state_label(x: int) -> str:
    return f"x={x:+d}"


def _excursion_probabilities(x: int) -> npt.NDArray[np.float64]:
    """pi_k(x), k = 0..5: probability that state x is visited k (>= 5) times per cycle."""
    a = 1.0 / (2.0 * abs(x))
    k = np.arange(1, 5)
    middle = (1.0 / (4.0 * x * x)) * (1.0 - a) ** (k - 1)
    return np.concatenate([[1.0 - a], middle, [a * (1.0 - a) ** 4]])


def random_excursions(
    bits: npt.ArrayLike, *, min_cycles: int = 500, alpha: float = DEFAULT_ALPHA
) -> TestResult:
    """Section 2.14: visits to states -4..-1, 1..4 per excursion of the random walk.

    For each state x, cycles are binned by how often they visit x
    (0..4, >= 5), ``P = igamc(5/2, chi2/2)``; eight p-values. The chi-square
    approximation needs ``J >= max(500, 0.005 sqrt(n))`` cycles, otherwise
    the test is skipped (``min_cycles`` lowers the 500 for worked examples).
    """
    name = "random_excursions"
    _require(min_cycles >= 1, "min_cycles must be positive")
    eps = as_bits(bits)
    n = int(eps.size)
    if n == 0:
        return _skip(name, "empty sequence")
    walk, cycles = _excursion_cycles(eps)
    reason = _excursion_skip_reason(n, cycles, min_cycles)
    if reason is not None:
        return _skip(name, reason, {"n": n, "cycles": cycles})
    # Cycle index of each step: the number of zeros strictly before it.
    cycle_id = np.cumsum(walk == 0) - (walk == 0)
    near = np.abs(walk) <= 4
    visits = np.bincount(cycle_id[near] * 9 + walk[near] + 4, minlength=cycles * 9).reshape(
        cycles, 9
    )
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    p_values: list[float] = []
    counts: dict[str, list[int]] = {}
    chi2s: list[float] = []
    for x in states:
        nu = np.bincount(np.minimum(visits[:, x + 4], 5), minlength=6)
        chi2 = _chi_square(nu, cycles * _excursion_probabilities(x))
        counts[_state_label(x)] = nu.tolist()
        chi2s.append(chi2)
        p_values.append(_igamc(2.5, chi2 / 2.0))
    statistics = {"n": n, "cycles": cycles, "counts": counts, "chi_square": chi2s}
    return _finish(name, p_values, statistics, alpha, [_state_label(x) for x in states])


def random_excursions_variant(
    bits: npt.ArrayLike, *, min_cycles: int = 500, alpha: float = DEFAULT_ALPHA
) -> TestResult:
    """Section 2.15: total visits to states -9..-1, 1..9 over all cycles.

    ``xi(x)`` visits are compared with the J expected,
    ``P = erfc(|xi(x) - J| / sqrt(2 J (4|x| - 2)))``; eighteen p-values,
    same cycle requirement as :func:`random_excursions`.
    """
    name = "random_excursions_variant"
    _require(min_cycles >= 1, "min_cycles must be positive")
    eps = as_bits(bits)
    n = int(eps.size)
    if n == 0:
        return _skip(name, "empty sequence")
    walk, cycles = _excursion_cycles(eps)
    reason = _excursion_skip_reason(n, cycles, min_cycles)
    if reason is not None:
        return _skip(name, reason, {"n": n, "cycles": cycles})
    near = np.abs(walk) <= 9
    totals = np.bincount(walk[near] + 9, minlength=19)
    states = [x for x in range(-9, 10) if x != 0]
    p_values = [
        _erfc(abs(int(totals[x + 9]) - cycles) / math.sqrt(2.0 * cycles * (4.0 * abs(x) - 2.0)))
        for x in states
    ]
    statistics = {"n": n, "cycles": cycles, "visits": [int(totals[x + 9]) for x in states]}
    return _finish(name, p_values, statistics, alpha, [_state_label(x) for x in states])


# ---------------------------------------------------------------------------
# Suite and multi-sequence analysis (section 4.2)
# ---------------------------------------------------------------------------


class _TestFunction(Protocol):
    def __call__(self, bits: npt.ArrayLike, *, alpha: float = ...) -> TestResult: ...


class _SuiteEntry(NamedTuple):
    test: _TestFunction
    min_bits: int  # recommended minimum input size; 0 = the test decides itself
    source: str  # where the recommendation comes from


# NIST STS order. Minimums are the "input size recommendation" of each
# section (2.x.7) for the default parameters; tests whose minimum is intrinsic
# to their parameter choice (longest run, universal) enforce it themselves.
_SUITE: dict[str, _SuiteEntry] = {
    "frequency": _SuiteEntry(frequency, 100, "SP 800-22 2.1.7"),
    "block_frequency": _SuiteEntry(block_frequency, 100, "SP 800-22 2.2.7"),
    "cumulative_sums": _SuiteEntry(cumulative_sums, 100, "SP 800-22 2.13.7"),
    "runs": _SuiteEntry(runs, 100, "SP 800-22 2.3.7"),
    "longest_run_of_ones": _SuiteEntry(longest_run_of_ones, 0, "SP 800-22 2.4.2"),
    "binary_matrix_rank": _SuiteEntry(binary_matrix_rank, 38 * 32 * 32, "SP 800-22 2.5.7"),
    "discrete_fourier_transform": _SuiteEntry(discrete_fourier_transform, 1000, "SP 800-22 2.6.7"),
    "non_overlapping_template_matching": _SuiteEntry(
        non_overlapping_template_matching, 0, "SP 800-22 2.7.7"
    ),
    "overlapping_template_matching": _SuiteEntry(
        overlapping_template_matching, 1_000_000, "SP 800-22 2.8.7"
    ),
    "universal": _SuiteEntry(universal, 0, "SP 800-22 2.9.7"),
    "approximate_entropy": _SuiteEntry(
        approximate_entropy, 1 << (10 + 6), "SP 800-22 2.12.7: m < log2(n) - 5"
    ),
    "random_excursions": _SuiteEntry(random_excursions, 1_000_000, "SP 800-22 2.14.7"),
    "random_excursions_variant": _SuiteEntry(
        random_excursions_variant, 1_000_000, "SP 800-22 2.15.7"
    ),
    "serial": _SuiteEntry(serial, 1 << (16 + 3), "SP 800-22 2.11.7: m < log2(n) - 2"),
    "linear_complexity": _SuiteEntry(linear_complexity, 1_000_000, "SP 800-22 2.10.7"),
}

SUITE_TESTS: tuple[str, ...] = tuple(_SUITE)
"""Names of the 15 tests in NIST STS order (the order of ``run_suite`` results)."""


def _selected(tests: Iterable[str] | None) -> list[str]:
    if tests is None:
        return list(SUITE_TESTS)
    wanted = set(tests)
    unknown = wanted - set(SUITE_TESTS)
    _require(not unknown, f"unknown SP 800-22 tests: {sorted(unknown)}")
    return [name for name in SUITE_TESTS if name in wanted]


def run_suite(
    bits: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    *,
    tests: Iterable[str] | None = None,
    enforce_min_length: bool = True,
) -> SuiteResult:
    """Run the SP 800-22 tests (all 15 by default) with reference parameters.

    With ``enforce_min_length`` (the default) a test whose recommended input
    size (SP 800-22 section 2.x.7) exceeds ``len(bits)`` is reported as
    skipped instead of producing a p-value the spec calls meaningless; e.g.
    overlapping templates, linear complexity and the random excursion tests
    need 10^6 bits. ``enforce_min_length=False`` runs everything that is
    computable, like NIST STS does.
    """
    _require(0.0 < alpha < 1.0, "alpha must be in (0, 1)")
    eps = as_bits(bits)
    n = int(eps.size)
    results = []
    for name in _selected(tests):
        entry = _SUITE[name]
        if enforce_min_length and n < entry.min_bits:
            reason = f"needs n >= {entry.min_bits:,} bits ({entry.source}), got n={n:,}"
            results.append(_skip(name, reason, {"n": n}))
        else:
            results.append(entry.test(eps, alpha=alpha))
    return SuiteResult(n, alpha, results)


def _summarise(test: str, label: str, p_values: Sequence[float], alpha: float) -> SubtestSummary:
    m = len(p_values)
    ps = np.asarray(p_values, dtype=np.float64)
    passed = int(np.count_nonzero(ps >= alpha))
    p_hat = 1.0 - alpha
    half = 3.0 * math.sqrt(p_hat * alpha / m)
    # Same binning as STS: floor(10 p), with p = 1 in the last bin.
    histogram = np.bincount(np.minimum((ps * 10).astype(np.int64), 9), minlength=10)
    expected = m / 10.0
    chi2 = float(np.sum((histogram - expected) ** 2) / expected)
    return SubtestSummary(
        test=test,
        label=label,
        sequences=m,
        passed_sequences=passed,
        proportion_interval=(p_hat - half, p_hat + half),
        histogram=tuple(int(h) for h in histogram),
        uniformity_p=_igamc(4.5, chi2 / 2.0),
    )


def run_multi(
    sequences: Iterable[npt.ArrayLike],
    alpha: float = DEFAULT_ALPHA,
    *,
    tests: Iterable[str] | None = None,
    enforce_min_length: bool = True,
) -> MultiSuiteResult:
    """SP 800-22 section 4.2: run the suite on m sequences and assess each sub-test.

    For every sub-test (each p-value index of each test, e.g. one template
    of the non-overlapping test) two checks are made over the sequences on
    which it was applicable:

    * **proportion** of p-values >= alpha must lie in
      ``(1 - alpha) +- 3 sqrt(alpha (1 - alpha) / m)``;
    * **uniformity** of the p-values: a chi-square over 10 equal bins,
      ``P_T = igamc(9/2, chi2/2) >= 0.0001`` (assessed from 55 sequences on).

    All sequences must have the same length (as with STS). A 2-D array is
    taken as one sequence per row; see :func:`puffer.analysis.bits.split_sequences`.
    """
    selected = _selected(tests)
    suites = [
        run_suite(seq, alpha, tests=selected, enforce_min_length=enforce_min_length)
        for seq in sequences
    ]
    _require(bool(suites), "run_multi needs at least one sequence")
    lengths = {s.n_bits for s in suites}
    _require(len(lengths) == 1, f"sequences must have equal length, got {sorted(lengths)}")
    order = {name: i for i, name in enumerate(SUITE_TESTS)}
    buckets: dict[tuple[int, int, str, str], list[float]] = {}
    for suite in suites:
        for result in suite.applicable:
            labels = result.labels or [""] * len(result.p_values)
            for position, (label, p) in enumerate(zip(labels, result.p_values, strict=True)):
                key = (order[result.name], position, result.name, label)
                buckets.setdefault(key, []).append(p)
    subtests = [
        _summarise(name, label, ps, alpha) for (_, _, name, label), ps in sorted(buckets.items())
    ]
    return MultiSuiteResult(lengths.pop(), alpha, suites, subtests)
