"""NIST SP 800-90B min-entropy estimators (non-IID track, section 6.3).

Why the non-IID track
---------------------
Fish motion and sensor noise are not independent and identically
distributed: a turning fish keeps turning, caustics drift slowly and sensor
noise has row and column structure. SP 800-90B therefore says such a source
is assessed with all ten estimators of section 6.3 and credited with the
*minimum*. Each estimator looks for a different way the data could be more
predictable than its histogram suggests:

* 6.3.1 Most Common Value (MCV): the i.i.d. baseline;
* 6.3.2 - 6.3.4 (bitstrings only): collision times, a first-order Markov
  model and a Maurer-style compression statistic;
* 6.3.5 - 6.3.6: over-represented tuples and long repeated substrings;
* 6.3.7 - 6.3.10: four online predictors (most common in window, lag,
  Markov model with counting, LZ78Y). Each one is scored on how often it
  guesses the next sample. A source whose next sample can be guessed has
  no more entropy than the guess rate allows, whatever its histogram says.

Section 3.1.3 combines them. Every applicable estimator runs on the original
symbols, and all ten run on the samples written out as bits. The assessed
entropy is ``min(H_original, bits_per_symbol * H_bitstring)``
(:func:`assess_non_iid`).

Fidelity
--------
Formulas, parameters and binary-only restrictions follow the January 2018
final text. Where the text is silent, the NIST reference implementation
(``usnistgov/SP800-90B_EntropyAssessment``, v1.1.x) decides. That covers tie
breaks (MultiMMC and LZ78Y predict the *largest* symbol among equally
frequent ones) and edge cases (collision means below 2 are clamped,
compression and collision fall back to one bit per bit when the binary search
has no solution). Every fallback errs towards *less* entropy except the one
the standard itself prescribes. Two places differ deliberately from the
reference tool, and both are documented where they occur:

* MultiMMC has no cap on dictionary entries. The 2018 text has none; the
  tool keeps a 100 000-entry cap from an earlier draft. Without the cap more
  transitions are counted, so the estimate can only be lower.
* The p_local recurrence runs exactly ten iterations, as section 6.3.7
  step 5 says. The tool iterates to convergence. For the run lengths seen in
  practice the two agree to many digits.

Performance
-----------
The standard describes every predictor as a per-sample loop with a
scoreboard. Running those loops literally in Python takes minutes on a
million samples. Here the work is reorganised so that numpy does it, and
each reorganisation is checked in the tests against a literal transcription
of the standard:

* The scoreboard rule "if score_d >= score_winner then winner = d", applied
  for d = 1..D, has a closed form. After step t the winner is the largest d
  that was correct at t and reached the new maximum score. If no such d
  exists, the winner does not change. With cumulative sums over a
  (time x subpredictor) hit matrix this becomes a few row reductions
  (:func:`_follow_the_leader`).
* Markov-model-with-counting predictions (MultiMMC, LZ78Y) only ever *add*
  counts. "The most frequent successor of this context so far" can then be
  computed for every time step at once by sorting events by context
  (:func:`_online_mode`). The LZ78Y dictionary cap becomes a threshold on
  each context's first appearance.
* t-Tuple and LRS counts come from one suffix array (prefix doubling with
  numpy sorts) and an LCP array (Kasai). One pass over the LCP intervals
  gives the counts for every tuple length at once, so even a constant
  million-sample input costs O(n log n).

Only three loops stay sequential: the Kasai LCP pass, the LCP-interval
stack and the sliding-window mode for non-binary MultiMCW. They are tight
pure-Python loops over lists. Assessing 100 000 byte samples (both tracks,
800 000 bits) takes about ten seconds on one core.

These are statistical tools only; there is no cryptography in this module.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "BINARY_ONLY",
    "BITSTRING_LIMIT",
    "ESTIMATORS",
    "Z_99",
    "Assessment",
    "EstimatorResult",
    "InsufficientDataError",
    "assess_non_iid",
    "collision",
    "compression",
    "lag",
    "lrs",
    "lz78y",
    "markov",
    "most_common_value",
    "multi_mcw",
    "multi_mmc",
    "prediction_estimate",
    "quick_min_entropy",
    "solve_p_local",
    "t_tuple",
    "to_bitstring",
]

Symbols = npt.NDArray[np.int64]
Mask = npt.NDArray[np.bool_]

Z_99 = 2.576  # z_{1-0.005}: the 99% upper confidence bound used throughout 800-90B

# Parameters fixed by SP 800-90B section 6.3 (January 2018).
MARKOV_SEQUENCE_LENGTH: Final = 128  # 6.3.3 step 3
COMPRESSION_BLOCK_BITS: Final = 6  # 6.3.4 step 1 (b)
COMPRESSION_DICTIONARY_BLOCKS: Final = 1000  # 6.3.4 step 2 (d)
COMPRESSION_SIGMA_FACTOR: Final = 0.5907  # 6.3.4 step 5 (c)
TUPLE_COUNT_THRESHOLD: Final = 35  # 6.3.5 step 1, 6.3.6 step 1
MULTI_MCW_WINDOWS: Final = (63, 255, 1023, 4095)  # 6.3.7 step 1
LAG_DEPTH: Final = 128  # 6.3.8 step 1 (D)
MULTI_MMC_DEPTH: Final = 16  # 6.3.9 step 1 (D)
LZ78Y_DEPTH: Final = 16  # 6.3.10 step 1 (B)
LZ78Y_MAX_DICTIONARY: Final = 65_536  # 6.3.10 step 1 (maxDictionarySize)
P_LOCAL_CONFIDENCE: Final = 0.99  # 6.3.7 step 5
P_LOCAL_ITERATIONS: Final = 10  # 6.3.7 step 5: x = x_10
BITSTRING_LIMIT: Final = 1_000_000  # 3.1.3: the bitstring track uses the first 10^6 bits
MAX_BITS_PER_SYMBOL: Final = 8  # same limit as the NIST tool; keeps alphabets <= 256

_PACK_LIMIT: Final = 1 << 62  # largest packed context id before re-densifying
_LAG_CHUNK_ROWS: Final = 4096  # Lag hit matrix is built 4096 x 128 at a time
_SCOREBOARD_CHUNK_ROWS: Final = 65_536


class InsufficientDataError(ValueError):
    """The input is too short (or too degenerate) for an estimator to be defined.

    :func:`assess_non_iid` records these as *skipped* rather than failing, as
    the NIST tool does: e.g. the compression estimate needs more than
    6 * 1000 bits, the t-tuple estimate needs a symbol seen 35 times.
    """


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    """One estimator's verdict.

    ``min_entropy`` is in bits per sample of the data the estimator saw (per
    bit on the bitstring track). ``p_upper`` is ``2 ** -min_entropy``: the
    upper bound on the probability of the most likely outcome (or of a
    correct prediction) that the estimate is built on. ``details`` holds the
    intermediate quantities named in the standard, as plain Python scalars
    so they serialise to JSON unchanged.
    """

    name: str
    min_entropy: float  # bits per sample
    p_upper: float  # upper bound on the probability of the most likely outcome / prediction
    details: dict[str, Any] = field(default_factory=dict)


def _result(name: str, min_entropy: float, details: dict[str, Any]) -> EstimatorResult:
    h = max(0.0, float(min_entropy))  # never report -0.0 or rounding noise below zero
    return EstimatorResult(name, h, 2.0**-h, details)


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def _as_symbols(samples: npt.ArrayLike) -> npt.NDArray[np.integer[Any]]:
    """Flatten ``samples`` to a 1-D integer array (bool is promoted to uint8)."""
    a = np.asarray(samples).ravel()
    if a.size == 0:  # an empty list arrives as float64; there is nothing to reject
        return np.zeros(0, dtype=np.int64)
    if a.dtype == np.bool_:
        return a.astype(np.uint8)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"samples must be integer symbols, got dtype {a.dtype}")
    return a


def _dense(samples: npt.ArrayLike) -> tuple[Symbols, int]:
    """Map symbols to ``0..k-1`` (order preserving) and return ``(symbols, k)``.

    The predictors and tuple counters need a dense alphabet. ``k`` is the
    number of *observed* symbols, the same alphabet size the NIST tool uses
    for the ``1/k`` floor of the prediction estimates.
    """
    a = _as_symbols(samples)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    if int(np.min(a)) >= 0 and int(np.max(a)) < 1 << 16:
        idx = a.astype(np.intp)
        present = np.bincount(idx) > 0
        lut = np.cumsum(present, dtype=np.int64) - 1
        return lut[idx], int(np.count_nonzero(present))
    values, inverse = np.unique(a, return_inverse=True)
    return inverse.astype(np.int64).ravel(), int(values.size)


def _binary(samples: npt.ArrayLike, section: str) -> Symbols:
    """Dense 0/1 view of ``samples``; the 6.3.2-6.3.4 estimators are bitstring-only."""
    s, k = _dense(samples)
    if k > 2:
        raise ValueError(
            f"SP 800-90B {section} is defined for binary data only "
            f"(got {k} distinct symbols); assess to_bitstring(samples, n) instead"
        )
    return s


def _sort_key(ids: Symbols) -> npt.NDArray[np.integer[Any]]:
    """``uint16`` copy of small non-negative ids: numpy's stable sort radix-sorts those."""
    if ids.size and int(np.max(ids)) < 1 << 16:
        return ids.astype(np.uint16)
    return ids


# --------------------------------------------------------------------------
# 6.3.1 Most Common Value
# --------------------------------------------------------------------------


def _upper_bound(p_hat: float, n: int) -> float:
    """``min(1, p + 2.576 sqrt(p (1 - p) / (n - 1)))``: the bound of 6.3.1, 6.3.5, 6.3.6."""
    return min(1.0, p_hat + Z_99 * math.sqrt(p_hat * (1.0 - p_hat) / (n - 1)))


def _most_common_value(s: Symbols) -> EstimatorResult:
    n = int(s.size)
    if n < 2:
        raise InsufficientDataError("most_common_value needs at least 2 samples")
    p_hat = int(np.max(np.bincount(s))) / n
    p_u = _upper_bound(p_hat, n)
    return _result("most_common_value", -math.log2(p_u), {"p_hat": p_hat, "n": n})


def most_common_value(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.1 Most Common Value estimate.

    p_hat = max_count / L
    p_u   = min(1, p_hat + 2.576 * sqrt(p_hat * (1 - p_hat) / (L - 1)))
    H     = -log2(p_u)

    It is the only estimator that sees nothing but the histogram. That makes
    it the cheapest one, O(L), and the right choice for the rolling runtime
    estimate (:func:`quick_min_entropy`). It cannot see dependence between
    samples; the other nine estimators exist to catch that.
    """
    return _most_common_value(_dense(samples)[0])


def quick_min_entropy(samples: npt.ArrayLike) -> float:
    """Cheap per-sample min-entropy for runtime credit: the MCV estimate.

    The rolling estimate in :mod:`puffer.entropy.channels` uses this on the
    last few thousand samples. Fewer than two samples prove nothing, so the
    result is 0.0 bits rather than an error: no evidence means no credit.
    """
    s, _ = _dense(samples)
    if s.size < 2:
        return 0.0
    return _most_common_value(s).min_entropy


# --------------------------------------------------------------------------
# 6.3.2 Collision estimate (binary only)
# --------------------------------------------------------------------------


def _collision_times(bits: Symbols) -> Symbols:
    """Steps 1-5 of 6.3.2: lengths of consecutive non-overlapping collision windows.

    For binary data a window starting at i ends after 2 samples if
    ``s_i == s_{i+1}`` and after 3 samples otherwise (three bits always hold
    a repeat). The windows form a chain i -> i + t_i. The chain is followed
    by pointer doubling, i.e. composing the jump table with itself, so the
    work is O(L log L) in numpy rather than O(L) in Python. A window that
    would run past the end of the data is incomplete and dropped, as in the
    standard.
    """
    n = int(bits.size)
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    terminal = n - 1  # positions 0..n-2 can start a window
    step = np.where(bits[:-1] == bits[1:], 2, 3).astype(np.int64)
    jump = np.append(np.minimum(np.arange(terminal, dtype=np.int64) + step, terminal), terminal)
    path = np.zeros(1, dtype=np.int64)
    while path[-1] != terminal:
        path = np.concatenate([path, jump[path]])  # first 2^(m+1) chain nodes, in order
        jump = jump[jump]  # jump 2^(m+1) windows ahead
    starts = path[path < terminal]
    times = step[starts]
    return times[starts + times <= n]


def _collision(bits: Symbols) -> EstimatorResult:
    times = _collision_times(bits)
    v = int(times.size)
    if v < 2:
        raise InsufficientDataError("collision estimate needs at least 2 collisions (>= 6 bits)")
    mean = float(times.mean())
    sigma = float(times.std(ddof=1))
    lower = mean - Z_99 * sigma / math.sqrt(v)
    # X' >= 2.5: no solution, step 8 gives 1 bit. X' < 2: clamp to p = 1 (0 bits).
    p = 0.5 if lower >= 2.5 else 0.5 + math.sqrt(1.25 - 0.5 * max(lower, 2.0))
    details = {"v": v, "mean": mean, "sigma": sigma, "mean_lower": lower, "p": p}
    return _result("collision", -math.log2(p), details)


def collision(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.2 Collision estimate (binary data only).

    t_i are the collision times (:func:`_collision_times`) and
    ``X' = mean(t) - 2.576 * std(t) / sqrt(v)``. Step 7 solves
    ``X' = p q^-2 (1 + (p^-1 - q^-1)/2) F(q) - p q^-1 (p^-1 - q^-1)/2`` for
    p in [0.5, 1]. Appendix G.1.1 reduces F to ``F(q) = q (2q^2 + 2q + 1)``.
    Substituting it, the right-hand side simplifies exactly to
    ``2 + 2 p (1 - p)``: the expected collision time of a biased coin.
    Solving the quadratic gives ``p = 1/2 + sqrt(5/4 - X'/2)``. The NIST
    reference does the same, and the closed form avoids the catastrophic
    cancellation of the original expression near p = 1. Edge cases:
    ``X' >= 2.5`` has no solution, and step 8 then gives 1 bit.
    ``X' < 2`` (more collisions than even a constant source would show) is
    clamped to p = 1, i.e. 0 bits: the conservative reading.

    Complexity O(L log L).
    """
    return _collision(_binary(samples, "6.3.2"))


# --------------------------------------------------------------------------
# 6.3.3 Markov estimate (binary only)
# --------------------------------------------------------------------------


def markov(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.3 Markov estimate (binary data only).

    P0 = C0 / L and P1 = 1 - P0 are estimated over the whole sequence.
    The transition probabilities ``P_ab = o_ab / (o_a0 + o_a1)`` come from the
    L - 1 adjacent pairs. A state that is never left has transition
    probabilities 0, as in the NIST tool. The most likely 128-bit sequence is
    one of 00..0, 0101..01, 011..1, 100..0, 1010..10 and 11..1, so its
    probability p_max is the largest of the six table entries.
    ``H = min(-log2(p_max) / 128, 1)``.

    The products are formed as log2 sums. The probability of a 128-bit
    sequence underflows easily, and a zero transition probability rules a
    sequence out. If every sequence is ruled out, the estimate is 1 bit. Only
    two-sample inputs can do that.

    There is deliberately no confidence bound here: the 2018 final text
    dropped the earlier drafts' confidence intervals for this estimator.

    Complexity O(L).
    """
    return _markov(_binary(samples, "6.3.3"))


def _markov(bits: Symbols) -> EstimatorResult:
    n = int(bits.size)
    if n < 2:
        raise InsufficientDataError("markov estimate needs at least 2 bits")
    first, second = bits[:-1], bits[1:]
    from0 = int(np.count_nonzero(first == 0))
    from1 = (n - 1) - from0
    p00 = int(np.count_nonzero((first == 0) & (second == 0))) / from0 if from0 else 0.0
    p01 = 1.0 - p00 if from0 else 0.0
    p10 = int(np.count_nonzero((first == 1) & (second == 0))) / from1 if from1 else 0.0
    p11 = 1.0 - p10 if from1 else 0.0
    p0 = int(np.count_nonzero(bits == 0)) / n
    p1 = 1.0 - p0

    def lg(p: float) -> float:
        return math.log2(p) if p > 0.0 else -math.inf

    m = MARKOV_SEQUENCE_LENGTH
    half = m // 2
    log2_sequences = (
        lg(p0) + (m - 1) * lg(p00),  # 00...0
        lg(p0) + half * lg(p01) + (half - 1) * lg(p10),  # 0101...01
        lg(p0) + lg(p01) + (m - 2) * lg(p11),  # 011...1
        lg(p1) + lg(p10) + (m - 2) * lg(p00),  # 100...0
        lg(p1) + half * lg(p10) + (half - 1) * lg(p01),  # 1010...10
        lg(p1) + (m - 1) * lg(p11),  # 11...1
    )
    log2_p_max = max(log2_sequences)
    h = 1.0 if log2_p_max == -math.inf else min(-log2_p_max / m, 1.0)
    details = {
        "p0": p0,
        "p1": p1,
        "p00": p00,
        "p01": p01,
        "p10": p10,
        "p11": p11,
        "log2_p_max": log2_p_max,
    }
    return _result("markov", h, details)


# --------------------------------------------------------------------------
# 6.3.4 Compression estimate (binary only)
# --------------------------------------------------------------------------


def _bisect_decreasing(f: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """Binary search for ``f(p) = target`` with f decreasing and f(lo) > target > f(hi).

    Iterates until the bracket can no longer be split in double precision,
    then returns the *upper* end. A larger probability means less entropy,
    so the last rounding step also errs on the conservative side.
    """
    while True:
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            return hi
        if f(mid) > target:
            lo = mid
        else:
            hi = mid


def _compression_distances(bits: Symbols, block_bits: int, dictionary: int) -> Symbols:
    """Steps 1-4 of 6.3.4: the distances D_i of the test blocks.

    ``D = i - (index of the previous occurrence of the same block)``, or
    ``D = i`` (1-based) when the block value has not occurred. The step 3/4
    dictionary of last positions holds exactly that, so it is computed
    directly: a stable sort by block value puts each block next to its
    previous occurrence.
    """
    n_blocks = bits.size // block_bits
    weights = 1 << np.arange(block_bits - 1, -1, -1, dtype=np.int64)
    values = bits[: n_blocks * block_bits].reshape(n_blocks, block_bits) @ weights
    order = np.argsort(values, kind="stable")
    same = values[order[1:]] == values[order[:-1]]
    previous = np.full(n_blocks, -1, dtype=np.int64)
    previous[order[1:]] = np.where(same, order[:-1], -1)
    index = np.arange(dictionary, n_blocks, dtype=np.int64)
    return index - previous[dictionary:]  # previous == -1 gives the 1-based index i + 1


def _compression_expectation(
    n_blocks: int, dictionary: int, block_bits: int
) -> Callable[[float], float]:
    """Right-hand side of 6.3.4 step 7, ``G(p) + (2^b - 1) G(q)``, as a function of p.

    The step 7 definition is a double sum
    ``G(z) = 1/v sum_{t=d+1}^{L} sum_{u=1}^{t} log2(u) F(z, t, u)`` with
    ``F = z^2 (1-z)^(u-1)`` for u < t and ``F = z (1-z)^(t-1)`` for u = t.
    Swapping the order of summation, the u < t part becomes
    ``z^2 sum_{u<L} (L - max(u, d)) log2(u) (1-z)^(u-1)``. The diagonal is a
    single sum. Both sums use the powers ``(1-z)^0 .. (1-z)^(L-1)``, which one
    vectorised ``exp`` produces per evaluation. With the weights precomputed,
    an evaluation is two weighted sums over L terms, O(L), instead of
    O(L^2). (Plain numpy reductions rather than BLAS ``dot``: their pairwise
    summation is accurate, and they avoid BLAS threading overhead.) The test
    suite checks this against the literal double sum.
    """
    v = n_blocks - dictionary
    u = np.arange(1, n_blocks, dtype=np.float64)
    inner_weight = (n_blocks - np.maximum(u, dictionary)) * np.log2(u)  # u = 1 .. L-1
    diagonal_weight = np.log2(np.arange(dictionary + 1, n_blocks + 1, dtype=np.float64))
    exponents = np.arange(n_blocks, dtype=np.float64)
    others = float(2**block_bits - 1)

    def g(z: float) -> float:
        if z >= 1.0:  # only (1-z)^0 survives, and it is weighted by log2(1) = 0
            return 0.0
        powers = np.exp(exponents * math.log1p(-z))
        inner = float(np.sum(inner_weight * powers[: n_blocks - 1]))
        diagonal = float(np.sum(diagonal_weight * powers[dictionary:]))
        return (z * z * inner + z * diagonal) / v

    def expectation(p: float) -> float:
        return g(p) + others * g((1.0 - p) / others)

    return expectation


def compression(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.4 Compression estimate (binary data only).

    The bits are cut into b = 6 bit blocks. The first d = 1000 blocks prime a
    dictionary, and every later block records the distance D_i back to its
    previous occurrence. This is Maurer's universal statistic: predictable
    data repeats blocks sooner. With
    ``X' = mean(log2 D) - 2.576 * sigma / sqrt(v)``,
    ``sigma = 0.5907 * sqrt(sum(log2(D)^2) / (v - 1) - mean^2)``, a binary
    search finds the p in ``[2^-b, 1]`` whose model distribution (one block
    value with probability p, the rest uniform) has expected statistic X'.
    The estimate is ``H = -log2(p) / b`` bits per bit. When no p fits
    (X' >= E(2^-b)), step 8 gives 1 bit. X' <= 0 is clamped to p = 1.

    Complexity O(L) per evaluation of the expectation, times about 50 bisection
    steps.
    """
    return _compression(_binary(samples, "6.3.4"))


def _compression(bits: Symbols) -> EstimatorResult:
    b, d = COMPRESSION_BLOCK_BITS, COMPRESSION_DICTIONARY_BLOCKS
    n_blocks = int(bits.size) // b
    if n_blocks <= d + 1:
        raise InsufficientDataError(f"compression estimate needs more than {(d + 1) * b} bits")
    log_d = np.log2(_compression_distances(bits, b, d).astype(np.float64))
    v = int(log_d.size)
    mean = float(log_d.mean())
    sigma = COMPRESSION_SIGMA_FACTOR * math.sqrt(
        max(0.0, float(np.sum(log_d * log_d)) / (v - 1) - mean * mean)
    )
    lower = mean - Z_99 * sigma / math.sqrt(v)
    expectation = _compression_expectation(n_blocks, d, b)
    p_min = 2.0**-b
    if lower >= expectation(p_min):
        p, h = p_min, 1.0
    elif lower <= 0.0:  # E(1) = 0: nothing is more predictable than a constant
        p, h = 1.0, 0.0
    else:
        p = _bisect_decreasing(expectation, lower, p_min, 1.0)
        h = -math.log2(p) / b
    details = {
        "blocks": n_blocks,
        "v": v,
        "mean": mean,
        "sigma": sigma,
        "mean_lower": lower,
        "p": p,
    }
    return _result("compression", h, details)


# --------------------------------------------------------------------------
# 6.3.5 t-Tuple and 6.3.6 LRS (shared suffix-array machinery)
# --------------------------------------------------------------------------


def _suffix_array(s: Symbols) -> Symbols:
    """Suffix array by prefix doubling (Manber-Myers), each round one numpy sort.

    A suffix that runs out of symbols sorts before every longer suffix
    sharing its prefix (an implicit smallest sentinel). After the round that
    compares prefixes of length 2h, the order is final once all ranks are
    distinct. That happens after about log2(longest repeat) rounds, each
    O(n log n).
    """
    n = int(s.size)
    rank = s.astype(np.int64)
    h = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        if h < n:
            second[: n - h] = rank[h:]
        key = rank * (n + 1) + (second + 1)
        sa = np.argsort(key).astype(np.int64)
        ordered = key[sa]
        rank = np.empty(n, dtype=np.int64)
        rank[sa] = np.concatenate([[0], np.cumsum(ordered[1:] != ordered[:-1])])
        if n == 0 or int(rank[sa[-1]]) == n - 1 or h >= n:
            return sa
        h *= 2


def _lcp_array(s: Symbols, sa: Symbols) -> list[int]:
    """LCP of adjacent suffixes in suffix-array order (Kasai et al. 2001), O(n).

    Kasai's lemma, ``lcp(i+1) >= lcp(i) - 1`` in text order, bounds the total
    number of symbol comparisons by 2n. The loop is inherently sequential,
    so it runs as plain Python over lists. A trailing -1 sentinel stops every
    comparison without bounds checks.
    """
    n = int(s.size)
    predecessor = np.full(n, -1, dtype=np.int64)
    predecessor[sa[1:]] = sa[:-1]
    text = [*s.tolist(), -1]
    prev = predecessor.tolist()
    plcp = [0] * n
    h = 0
    for i in range(n):
        j = prev[i]
        if j < 0:
            h = 0
            continue
        while text[i + h] == text[j + h]:
            h += 1
        plcp[i] = h
        if h:
            h -= 1
    return [plcp[p] for p in sa.tolist()]


@dataclass(frozen=True, slots=True)
class _TupleProfile:
    """Tuple statistics of one sequence for every tuple length W.

    ``max_count[W]`` is the number of occurrences of the most common W-tuple
    (Q[W] in 6.3.5). ``pair_count[W]`` is ``sum_i C(c_i, 2)`` over the distinct
    W-tuples, the numerator of P_W in 6.3.6. Both arrays are indexed 0..v,
    where v is the length of the longest repeated substring; index 0 is unused.
    """

    n: int
    max_count: npt.NDArray[np.int64]
    pair_count: npt.NDArray[np.int64]

    @property
    def lrs_length(self) -> int:
        return int(self.max_count.size) - 1


def _tuple_profile(s: Symbols) -> _TupleProfile:
    """Q[W] and the pair counts for every W from one bottom-up LCP-interval pass.

    The suffixes sharing a W-prefix form maximal runs of LCP >= W in suffix
    array order. These runs are the lcp-intervals of Abouelhoda et al.
    (2004). An interval with lcp value l, parent value p and c suffixes is
    such a run for every W in (p, l]. Hence Q[W] is the largest interval
    with l >= W, and the pair count gets ``C(c, 2)`` added over (p, l]
    (difference array). One stack pass, O(n), covers all lengths; constant
    or periodic inputs, where W runs up to n, cost no more.
    """
    n = int(s.size)
    if n < 2:
        raise InsufficientDataError("tuple estimates need at least 2 samples")
    lcp = _lcp_array(s, _suffix_array(s))
    v = max(lcp)
    best = [0] * (v + 1)
    diff = [0] * (v + 2)
    stack_lcp = [0]
    stack_left = [0]
    for i in range(1, n + 1):
        cur = lcp[i] if i < n else 0
        left = i - 1
        while cur < stack_lcp[-1]:
            ell = stack_lcp.pop()
            left = stack_left.pop()
            size = i - left
            parent = max(cur, stack_lcp[-1])
            if size > best[ell]:
                best[ell] = size
            pairs = size * (size - 1) // 2
            diff[parent + 1] += pairs
            diff[ell + 1] -= pairs
        if cur > stack_lcp[-1]:
            stack_lcp.append(cur)
            stack_left.append(left)
    best_arr = np.asarray(best, dtype=np.int64)
    max_count = np.maximum(np.maximum.accumulate(best_arr[::-1])[::-1], 1)
    pair_count = np.cumsum(np.asarray(diff, dtype=np.int64))[: v + 1]
    return _TupleProfile(n, max_count, pair_count)


def _t_tuple(profile: _TupleProfile) -> EstimatorResult:
    q = profile.max_count[1:]
    t = int(np.count_nonzero(q >= TUPLE_COUNT_THRESHOLD))  # Q is non-increasing in W
    if t == 0:
        raise InsufficientDataError(
            f"t-tuple estimate needs a symbol occurring {TUPLE_COUNT_THRESHOLD} times"
        )
    n = profile.n
    width = np.arange(1, t + 1, dtype=np.float64)
    p = q[:t] / (n - width + 1.0)
    p_max = float(np.max(p ** (1.0 / width)))
    p_u = _upper_bound(p_max, n)
    return _result("t_tuple", -math.log2(p_u), {"t": t, "p_hat": p_max, "n": n})


def _lrs(profile: _TupleProfile) -> EstimatorResult:
    q = profile.max_count[1:]
    u = int(np.count_nonzero(q >= TUPLE_COUNT_THRESHOLD)) + 1
    v = profile.lrs_length
    if v < u:
        raise InsufficientDataError("LRS estimate needs a repeated tuple of length >= u")
    n = profile.n
    width = np.arange(u, v + 1, dtype=np.float64)
    tuples = n - width + 1.0
    p = profile.pair_count[u : v + 1] / (tuples * (tuples - 1.0) / 2.0)
    p_max = float(np.max(p ** (1.0 / width)))
    p_u = _upper_bound(p_max, n)
    return _result("lrs", -math.log2(p_u), {"u": u, "v": v, "p_hat": p_max, "n": n})


def t_tuple(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.5 t-Tuple estimate.

    t is the largest length for which some t-tuple occurs at least 35 times.
    For i = 1..t, ``P_max[i] = (Q[i] / (L - i + 1)) ** (1 / i)``, where Q[i]
    counts the most common i-tuple (overlapping occurrences). With
    ``p = max_i P_max[i]`` and the usual 99% bound
    ``p_u = min(1, p + 2.576 sqrt(p (1 - p) / (L - 1)))``, ``H = -log2(p_u)``.

    It catches sources that repeat short patterns more often than their
    symbol frequencies predict. Complexity O(L log L) (suffix array).
    """
    return _t_tuple(_tuple_profile(_dense(samples)[0]))


def lrs(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.6 Longest Repeated Substring (LRS) estimate.

    u is the smallest length whose most common tuple occurs fewer than 35
    times, and v is the length of the longest repeated substring. For W in
    u..v, ``P_W = sum_i C(C_i, 2) / C(L - W + 1, 2)``, the probability that
    two random W-tuples are equal. ``P_max,W = P_W ** (1 / W)``, then the 99%
    bound and ``H = -log2(p_u)``. It is the long-range partner of the
    t-tuple estimate: one long block that repeats (a recording looped, a
    buffer replayed) barely moves the histogram but dominates P_W.

    Complexity O(L log L).
    """
    return _lrs(_tuple_profile(_dense(samples)[0]))


# --------------------------------------------------------------------------
# Shared prediction-estimate machinery (6.3.7 steps 4-6, reused by 6.3.8-6.3.10)
# --------------------------------------------------------------------------


def _longest_run(mask: Mask) -> int:
    """Length of the longest run of True values."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)))


def _log_no_run_probability(p: float, r: int, n: int) -> float:
    """Natural log of the step 5 formula for P(no run of r successes in n trials).

    ``(1 - p x) / ((r + 1 - r x) q) / x^(n+1)`` with ``q = 1 - p`` and x from
    ``x_j = 1 + q p^r x_{j-1}^(r+1)``, ``x_0 = 1``, j = 1..10. It is written in
    terms of ``delta = x - 1``, because for realistic n delta is about 1e-8
    and x itself would lose every significant digit. The formula approximates
    the root of ``1 - x + q p^r x^(r+1)`` (Feller). Where it breaks down
    (very large p for the given r), the true probability is negligible, so
    -inf is returned, which tells the search "p is too large".
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf
    q = 1.0 - p
    log_coefficient = math.log(q) + r * math.log(p)
    delta = 0.0
    for _ in range(P_LOCAL_ITERATIONS):
        exponent = log_coefficient + (r + 1) * math.log1p(delta)
        if exponent > 700.0:
            return -math.inf
        delta = math.exp(exponent)
    numerator = q - p * delta  # 1 - p x
    denominator = 1.0 - r * delta  # r + 1 - r x
    if numerator <= 0.0 or denominator <= 0.0:
        return -math.inf
    return math.log(numerator) - math.log(denominator * q) - (n + 1) * math.log1p(delta)


def solve_p_local(n_predictions: int, longest_run: int) -> float:
    """SP 800-90B 6.3.7 step 5: the local predictor performance P_local.

    With ``r = longest_run + 1``, this is the success probability p for which
    n Bernoulli(p) trials have no run of r successes with probability 0.99.
    In words: the largest per-prediction success rate that still makes the
    observed longest run of correct predictions unsurprising at 99%
    confidence. A predictor can be right in a burst while its overall rate
    stays low (e.g. during a frozen frame); P_local turns such a burst into
    credit lost. Solved by binary search on (0, 1); the formula decreases
    in p. Against the exact longest-run distribution it is accurate to about
    1e-11 for N >= 1000, and within 0.003 (always above, i.e. conservative)
    down to N = 2.
    """
    if n_predictions < 1 or not 0 <= longest_run <= n_predictions:
        raise ValueError("need 0 <= longest_run <= n_predictions and n_predictions >= 1")
    if longest_run == n_predictions:
        # Every prediction was right: no p < 1 makes that unlikely, so the step 5
        # equation has no root (Feller's formula is not valid for r > n either).
        return 1.0
    r = longest_run + 1
    target = math.log(P_LOCAL_CONFIDENCE)
    return _bisect_decreasing(
        lambda p: _log_no_run_probability(p, r, n_predictions), target, 0.0, 1.0
    )


def prediction_estimate(
    correct: npt.ArrayLike, alphabet_size: int, name: str = "prediction"
) -> EstimatorResult:
    """Steps 4-6 shared by the four predictors of SP 800-90B 6.3.7-6.3.10.

    For the N predictions with C correct:

    * global performance ``P_global = C / N``. Its 99% upper bound is
      ``1 - 0.01^(1/N)`` if C = 0 and
      ``min(1, P + 2.576 sqrt(P (1 - P) / (N - 1)))`` otherwise;
    * local performance P_local, from the longest run of correct
      predictions (:func:`solve_p_local`);
    * ``H = -log2(max(P_global', P_local, 1/k))``. A predictor worse than
      chance proves nothing, so the 1/k floor keeps H <= log2(k).
    """
    hits = np.asarray(correct, dtype=np.bool_).ravel()
    n = int(hits.size)
    if n < 2:
        raise InsufficientDataError(f"{name} needs at least 2 predictions")
    if alphabet_size < 1:
        raise ValueError("alphabet_size must be >= 1")
    c = int(np.count_nonzero(hits))
    run = _longest_run(hits)
    p_global = c / n
    p_global_upper = 1.0 - 0.01 ** (1.0 / n) if c == 0 else _upper_bound(p_global, n)
    p_local = solve_p_local(n, run)
    p = max(p_global_upper, p_local, 1.0 / alphabet_size)
    details = {
        "n": n,
        "correct": c,
        "longest_run": run,
        "p_global": p_global,
        "p_global_upper": p_global_upper,
        "p_local": p_local,
        "k": alphabet_size,
    }
    return _result(name, -math.log2(p), details)


def _follow_the_leader(chunks: Iterable[Mask]) -> Mask:
    """Scoreboard of 6.3.7-6.3.9: which predictions of the current winner were right.

    ``hits[t, d]`` says whether subpredictor d was right at step t (False for
    a Null prediction). The standard updates the scoreboard sequentially:
    for d = 1..D, if d was right, increment its score, and if
    ``score_d >= score_winner`` set ``winner = d``. That loop has a closed
    form. Let M be the highest score after step t. If some d was right at t
    and now scores M, the new winner is the largest such d. Otherwise the
    winner does not change. (The winner always holds the highest score. An
    earlier-processed d that reaches the top is overtaken by any later one
    that also reaches it, and no d below the top can take over.) So the
    winner series comes from cumulative sums, a row max and a forward fill:
    O(N D) numpy work. ``chunks`` lets the caller stream the hit matrix in
    row blocks (Lag's is N x 128).
    """
    scores: npt.NDArray[np.int64] | None = None
    winner = 0  # 6.3.7 step 2: winner = 1, the first subpredictor
    out: list[Mask] = []
    for hits in chunks:
        rows, width = hits.shape
        if rows == 0:
            continue
        after = np.cumsum(hits, axis=0, dtype=np.int64)
        if scores is not None:
            after += scores
        top = after.max(axis=1)
        reached = hits & (after == top[:, None])
        changed = reached.any(axis=1)
        newest = (width - 1) - np.argmax(reached[:, ::-1], axis=1)
        last_change = np.maximum.accumulate(np.where(changed, np.arange(rows), -1))
        winner_after = np.where(last_change >= 0, newest[np.maximum(last_change, 0)], winner)
        winner_before = np.concatenate([[winner], winner_after[:-1]])
        out.append(hits[np.arange(rows), winner_before])
        scores = after[-1]
        winner = int(winner_after[-1])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.bool_)


def _row_chunks(hits: Mask, rows: int = _SCOREBOARD_CHUNK_ROWS) -> Iterator[Mask]:
    for start in range(0, hits.shape[0], rows):
        yield hits[start : start + rows]


def _context_ids(s: Symbols, k: int, max_order: int) -> Iterator[tuple[int, Symbols]]:
    """Yield ``(d, ids)`` where ``ids[t]`` identifies the context ``s[t-d:t]``.

    ``ids[t] = -1`` for t < d. Contexts are packed base-k integers, each
    built from the previous order with one multiply-add. When the packing
    would overflow (byte alphabets beyond 7 symbols), the previous ids are
    first renumbered densely, which keeps them injective.
    """
    n = int(s.size)
    ids = np.full(n, -1, dtype=np.int64)
    ids[1:] = s[:-1]
    for d in range(1, max_order + 1):
        if d >= n:
            return
        if d > 1:
            body = ids[d:]
            if (int(np.max(body)) + 1) * k >= _PACK_LIMIT:
                body = np.unique(body, return_inverse=True)[1].astype(np.int64).ravel()
            ids = np.full(n, -1, dtype=np.int64)
            ids[d:] = body * k + s[: n - d]
        yield d, ids


def _online_mode(groups: Symbols, outcomes: Symbols, k: int) -> tuple[Symbols, Symbols, Symbols]:
    """For each event, the most frequent outcome among *earlier* events of its group.

    Events are in time order. ``groups[e]`` is the context and ``outcomes[e]``
    the symbol that followed it. Returns ``(prediction, count, first)``:

    * ``prediction`` is the largest outcome among those with the highest
      count (the reference implementation's tie break), or -1;
    * ``count`` is that highest count (0 means the context is new: a Null
      prediction);
    * ``first`` is the index of the group's first event.

    This is the inner loop of MultiMMC and LZ78Y, done for every event at
    once. Counts only grow, so the running mode of a group changes when an
    outcome sets a new maximum (a new segment starts) or ties it (the
    largest tied outcome wins within the segment). After a stable sort by
    group, both are running maxima. O(n log n).
    """
    n = int(groups.size)
    order = np.argsort(_sort_key(groups), kind="stable").astype(np.int64)
    g = groups[order]
    y = outcomes[order]
    pos = np.arange(n, dtype=np.int64)
    starts = np.ones(n, dtype=np.bool_)
    starts[1:] = g[1:] != g[:-1]
    gid = np.cumsum(starts, dtype=np.int64) - 1
    group_start = np.maximum.accumulate(np.where(starts, pos, 0))
    if k <= 2:
        ones = np.cumsum(y)
        ones_in_group = ones - (ones[group_start] - y[group_start])
        count = np.where(y == 1, ones_in_group, pos - group_start + 1 - ones_in_group)
    else:
        pair = gid * k + y
        pair_order = np.argsort(pair, kind="stable")
        sorted_pair = pair[pair_order]
        pair_starts = np.ones(n, dtype=np.bool_)
        pair_starts[1:] = sorted_pair[1:] != sorted_pair[:-1]
        count = np.empty(n, dtype=np.int64)
        count[pair_order] = pos - np.maximum.accumulate(np.where(pair_starts, pos, 0)) + 1
    offset = gid * (n + 1)
    run_max = np.maximum.accumulate(offset + count) - offset
    prev_max = np.zeros(n, dtype=np.int64)
    prev_max[1:] = run_max[:-1]
    prev_max[starts] = 0
    segment_offset = (np.cumsum(count > prev_max, dtype=np.int64) - 1) * (k + 1)
    tied = np.where(count == run_max, y, -1)
    mode = np.maximum.accumulate(segment_offset + tied + 1) - segment_offset - 1
    prev_mode = np.full(n, -1, dtype=np.int64)
    prev_mode[1:] = mode[:-1]
    prev_mode[starts] = -1
    prediction = np.empty(n, dtype=np.int64)
    prediction[order] = prev_mode
    best = np.empty(n, dtype=np.int64)
    best[order] = prev_max
    first = np.empty(n, dtype=np.int64)
    first[order] = order[group_start]
    return prediction, best, first


# --------------------------------------------------------------------------
# 6.3.7 MultiMCW prediction
# --------------------------------------------------------------------------


def _window_modes_binary(s: Symbols, w: int) -> Symbols:
    """Most common bit of ``s[t-w:t]`` for t in [w, L), ties going to the most recent bit."""
    csum = np.concatenate([[0], np.cumsum(s)])
    ones2 = 2 * (csum[w:-1] - csum[: -w - 1])
    return np.where(ones2 > w, 1, np.where(ones2 < w, 0, s[w - 1 : -1]))


def _window_modes_general(s: Symbols, k: int, w: int) -> Symbols:
    """Most common symbol of ``s[t-w:t]`` for t in [w, L), ties going to the most recent.

    Sliding counts with the current mode kept incrementally (the NIST
    reference approach). Adding a symbol can only make *it* the mode, and it
    is then the most recent. Only when the mode itself leaves the window is
    the alphabet rescanned: argmax of (count, last position). That happens
    with probability about max_count / w per step, so the cost is
    O(L + rescans * k) in a tight Python loop.
    """
    seq = s.tolist()
    n = len(seq)
    count = [0] * k
    last = [-1] * k
    mode, top = 0, 0
    for t in range(w):
        x = seq[t]
        count[x] += 1
        last[x] = t
        if count[x] >= top:
            mode, top = x, count[x]
    out = [0] * (n - w)
    for t in range(w, n):
        out[t - w] = mode
        x, gone = seq[t], seq[t - w]
        count[x] += 1
        last[x] = t
        count[gone] -= 1
        if gone != mode:
            if count[x] >= top:
                mode, top = x, count[x]
        elif x != gone:
            top, newest = max(zip(count, last, strict=True))
            mode = seq[newest]
    return np.asarray(out, dtype=np.int64)


def _window_modes(s: Symbols, k: int, w: int) -> Symbols:
    return _window_modes_binary(s, w) if k <= 2 else _window_modes_general(s, k, w)


def _multi_mcw_correct(s: Symbols, k: int, windows: tuple[int, ...]) -> Mask:
    """``correct`` of 6.3.7 steps 1-3 for t = w_1 .. L-1 (0-based)."""
    n = int(s.size)
    w1 = windows[0]
    hits = np.zeros((max(n - w1, 0), len(windows)), dtype=np.bool_)
    for j, w in enumerate(windows):
        if n > w:
            hits[w - w1 :, j] = _window_modes(s, k, w) == s[w:]
    return _follow_the_leader(_row_chunks(hits))


def _multi_mcw(s: Symbols, k: int, windows: tuple[int, ...] = MULTI_MCW_WINDOWS) -> EstimatorResult:
    if s.size < windows[0] + 2:
        raise InsufficientDataError(f"multi_mcw needs at least {windows[0] + 2} samples")
    return prediction_estimate(_multi_mcw_correct(s, k, windows), k, "multi_mcw")


def multi_mcw(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.7 Multi Most Common in Window prediction estimate.

    Four subpredictors predict the most common value in the last
    63, 255, 1023 and 4095 samples; ties go to the most recently seen
    value. A subpredictor is Null until its window is full. A scoreboard
    picks the subpredictor with the most correct guesses so far. This
    catches sources whose bias drifts (slow lighting changes, a fish that
    sulks in one corner): a recent window sees the local bias that a global
    histogram averages away. The N = L - 63 predictions of the winner then go
    through :func:`prediction_estimate`.

    The standard defines the estimate for any L > 64, so it runs there; the
    NIST tool refuses inputs shorter than 4096 samples. Complexity O(L) for
    bits (cumulative sums), O(L + rescans * k) otherwise.
    """
    return _multi_mcw(*_dense(samples))


# --------------------------------------------------------------------------
# 6.3.8 Lag prediction
# --------------------------------------------------------------------------


def _lag_hit_chunks(s: Symbols, depth: int) -> Iterator[Mask]:
    """Row blocks of ``hits[t, d-1] = (s[t] == s[t-d])`` for t = 1..L-1, d = 1..D."""
    n = int(s.size)
    padded = np.concatenate([np.full(depth, -1, dtype=np.int64), s])
    windows = sliding_window_view(padded, depth + 1)  # windows[t, j] = s[t - depth + j]
    for start in range(1, n, _LAG_CHUNK_ROWS):
        block = windows[start : min(n, start + _LAG_CHUNK_ROWS)]
        yield block[:, depth - 1 :: -1] == block[:, depth : depth + 1]


def _lag_correct(s: Symbols, depth: int = LAG_DEPTH) -> Mask:
    """``correct`` of 6.3.8 steps 1-3 for t = 1 .. L-1 (0-based)."""
    return _follow_the_leader(_lag_hit_chunks(s, depth))


def lag(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.8 Lag prediction estimate.

    D = 128 subpredictors; subpredictor d guesses that the next sample
    equals the one d steps back. The scoreboard follows the best lag. It
    catches periodic structure: a light flickering at mains frequency, a
    camera that duplicates frames, a replayed loop shorter than 128 samples.
    The N = L - 1 predictions go through :func:`prediction_estimate`.

    Complexity O(L D), streamed through the vectorised scoreboard 4096 rows
    at a time so memory stays at a few MB.
    """
    return _lag(*_dense(samples))


def _lag(s: Symbols, k: int) -> EstimatorResult:
    if s.size < 3:
        raise InsufficientDataError("lag needs at least 3 samples")
    return prediction_estimate(_lag_correct(s), k, "lag")


# --------------------------------------------------------------------------
# 6.3.9 MultiMMC prediction
# --------------------------------------------------------------------------


def _multi_mmc_correct(s: Symbols, k: int, depth: int = MULTI_MMC_DEPTH) -> Mask:
    """``correct`` of 6.3.9 steps 1-4 for t = 2 .. L-1 (0-based).

    For order d, the event at time t is "context ``s[t-d:t]`` was followed by
    ``s[t]``". The standard adds it to M_d at step t + 1, right after
    predicting ``s[t]`` from the *earlier* events of the same context. That
    is exactly what :func:`_online_mode` returns.
    """
    n = int(s.size)
    hits = np.zeros((max(n - 2, 0), depth), dtype=np.bool_)
    for d, ids in _context_ids(s, k, depth):
        prediction, count, _ = _online_mode(ids[d:], s[d:], k)
        ok = (count > 0) & (prediction == s[d:])
        start = max(d, 2)
        hits[start - 2 :, d - 1] = ok[start - d :]
    return _follow_the_leader(_row_chunks(hits))


def multi_mmc(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.9 Multi Markov Model with Counting prediction estimate.

    D = 16 subpredictors. The order-d one counts the transitions from every
    d-sample context to the next sample and predicts the most frequent
    successor of the current context (Null for an unseen context; ties go to
    the largest symbol, as in the reference implementation). The scoreboard
    follows the best order. It catches any short-memory dependence: a fish's
    heading innovations, quantisation patterns of the sensor. The
    N = L - 2 predictions go through :func:`prediction_estimate`.

    No cap on dictionary entries. The 2018 text has none, while the NIST
    tool keeps a 100 000-entry cap from an earlier draft. Uncapped counting
    can only predict better, so it is the conservative choice. Complexity
    O(D L log L).
    """
    return _multi_mmc(*_dense(samples))


def _multi_mmc(s: Symbols, k: int) -> EstimatorResult:
    if s.size < 4:
        raise InsufficientDataError("multi_mmc needs at least 4 samples")
    return prediction_estimate(_multi_mmc_correct(s, k), k, "multi_mmc")


# --------------------------------------------------------------------------
# 6.3.10 LZ78Y prediction
# --------------------------------------------------------------------------


def _first_occurrences(ids: Symbols) -> npt.NDArray[np.int64]:
    """1 where ``ids[e]`` appears for the first time, else 0."""
    _, first = np.unique(_sort_key(ids), return_index=True)
    mask = np.zeros(ids.size, dtype=np.int64)
    mask[first] = 1
    return mask


def _lz78y_correct(
    s: Symbols, k: int, depth: int = LZ78Y_DEPTH, max_dictionary: int = LZ78Y_MAX_DICTIONARY
) -> Mask:
    """``correct`` of 6.3.10 step 3 for t = B+1 .. L-1 (0-based).

    Event e (time t = B + e) adds "context ``s[t-j:t]`` -> ``s[t]``" for every
    j = B..1. A context enters the dictionary at its first appearance only if
    there is room, and entries never leave. So the admitted contexts are the
    first ``max_dictionary`` distinct (j, context) pairs in the order
    (time, B - j). Each new order-j context also makes every longer context
    at that step new, so the new orders at any step are a top range of j.
    Pass 1 counts the new contexts per step and finds the step where the
    dictionary fills, and how many of that step's new contexts fit. Pass 2
    runs :func:`_online_mode` per order, discards the counts of contexts that
    were never admitted, and keeps the order with the highest count; ties go
    to the longest context, as in step 3b, which scans j from B down with a
    strict ">".
    """
    n = int(s.size)
    outcomes = s[depth:]
    n_events = int(outcomes.size)
    new_per_step = np.zeros(n_events, dtype=np.int64)
    for _, ids in _context_ids(s, k, depth):
        new_per_step += _first_occurrences(ids[depth:])
    total = np.cumsum(new_per_step)
    full_at, cut_order = n_events, 1  # defaults: the dictionary never fills
    if n_events and int(total[-1]) >= max_dictionary:
        full_at = int(np.searchsorted(total, max_dictionary, side="left"))
        room = max_dictionary - (int(total[full_at - 1]) if full_at else 0)
        cut_order = depth - room + 1  # orders >= cut_order still fit at step full_at
    best_count = np.zeros(n_events, dtype=np.int64)
    best_prediction = np.full(n_events, -1, dtype=np.int64)
    for j, ids in _context_ids(s, k, depth):
        prediction, count, first = _online_mode(ids[depth:], outcomes, k)
        admitted = (first < full_at) | ((first == full_at) & (j >= cut_order))
        count = np.where(admitted, count, 0)
        better = (count > 0) & (count >= best_count)  # ascending j: ties go to the longer
        best_count = np.where(better, count, best_count)
        best_prediction = np.where(better, prediction, best_prediction)
    if n <= depth + 1:
        return np.zeros(0, dtype=np.bool_)
    return (best_count[1:] > 0) & (best_prediction[1:] == outcomes[1:])


def lz78y(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.10 LZ78Y prediction estimate.

    A dictionary maps every context of length 1..B = 16 to counts of the
    symbol that followed it. At most 65 536 contexts are kept, the first
    ones seen. The prediction is the most frequent successor of whichever
    current context (longest first) has the highest count. Like LZ78
    compression, it rewards a source that reuses long phrases; unlike
    MultiMMC it trusts long contexts as soon as they reappear, and it has no
    scoreboard. The N = L - B - 1 predictions go through
    :func:`prediction_estimate`.

    Complexity O(B L log L); see :func:`_lz78y_correct`.
    """
    return _lz78y(*_dense(samples))


def _lz78y(s: Symbols, k: int) -> EstimatorResult:
    if s.size < LZ78Y_DEPTH + 3:
        raise InsufficientDataError(f"lz78y needs at least {LZ78Y_DEPTH + 3} samples")
    return prediction_estimate(_lz78y_correct(s, k), k, "lz78y")


# --------------------------------------------------------------------------
# Section 3.1.3: combining the estimators
# --------------------------------------------------------------------------

ESTIMATORS: Final[dict[str, Callable[[npt.ArrayLike], EstimatorResult]]] = {
    "most_common_value": most_common_value,
    "collision": collision,
    "markov": markov,
    "compression": compression,
    "t_tuple": t_tuple,
    "lrs": lrs,
    "multi_mcw": multi_mcw,
    "lag": lag,
    "multi_mmc": multi_mmc,
    "lz78y": lz78y,
}
"""All ten non-IID estimators in section order, keyed by result name."""

BINARY_ONLY: Final = frozenset({"collision", "markov", "compression"})
"""Estimators the 2018 standard restricts to binary data."""


@dataclass(frozen=True, slots=True)
class Assessment:
    """Outcome of the SP 800-90B section 3.1.3 non-IID initial entropy estimate.

    ``min_entropy`` is the assessed min-entropy per sample,
    ``min(h_original, bits_per_symbol * h_bitstring)``. ``h_original`` and
    ``h_bitstring`` are the per-track minima. They are capped at
    ``bits_per_symbol`` and 1 bit, the most the data can carry, the same
    caps as the NIST tool. ``results`` lists every estimator that ran, tagged
    in ``details["track"]``, either ``"original"`` or ``"bitstring"``.
    ``skipped`` maps ``"track/name"`` to the reason an estimator could not run
    on data this short. ``n_bitstring`` is the number of bits the bitstring
    track assessed (0 for binary input, whose bitstring is the input itself).
    """

    min_entropy: float
    h_original: float
    h_bitstring: float
    bits_per_symbol: int
    results: list[EstimatorResult]
    n_samples: int
    n_bitstring: int = 0
    skipped: dict[str, str] = field(default_factory=dict)

    def track(self, name: str) -> dict[str, EstimatorResult]:
        """The results of one track (``"original"`` or ``"bitstring"``), by estimator name."""
        return {r.name: r for r in self.results if r.details.get("track") == name}

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for reports and the API."""
        return {
            "min_entropy": self.min_entropy,
            "h_original": self.h_original,
            "h_bitstring": self.h_bitstring,
            "bits_per_symbol": self.bits_per_symbol,
            "n_samples": self.n_samples,
            "n_bitstring": self.n_bitstring,
            "results": [
                {
                    "name": r.name,
                    "min_entropy": r.min_entropy,
                    "p_upper": r.p_upper,
                    "details": dict(r.details),
                }
                for r in self.results
            ],
            "skipped": dict(self.skipped),
        }


def to_bitstring(samples: npt.ArrayLike, bits_per_symbol: int) -> npt.NDArray[np.uint8]:
    """Write each sample as ``bits_per_symbol`` bits, most significant first (3.1.3).

    The bitstring track lets the binary-only estimators, and dependence
    between bit positions, see what the symbol-level estimators cannot
    (e.g. a constant low bit, or one nibble copying the other).
    """
    if not 1 <= bits_per_symbol <= 63:
        raise ValueError("bits_per_symbol must be in 1..63")
    a = _as_symbols(samples).astype(np.int64)
    if a.size and (int(np.min(a)) < 0 or int(np.max(a)) >= 1 << bits_per_symbol):
        raise ValueError(f"samples must lie in [0, 2**{bits_per_symbol})")
    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    return ((a[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def _assess_track(
    s: Symbols, k: int, *, binary: bool, track: str
) -> tuple[list[EstimatorResult], dict[str, str]]:
    """Run the estimators of one track on dense symbols; inapplicable ones are skipped.

    The suffix array is the most expensive shared structure, so it is built
    once and used for both t-tuple and LRS.
    """
    results: list[EstimatorResult] = []
    skipped: dict[str, str] = {}

    def attempt(name: str, compute: Callable[[], EstimatorResult]) -> None:
        try:
            result = compute()
        except InsufficientDataError as exc:
            skipped[f"{track}/{name}"] = str(exc)
            return
        results.append(replace(result, details={**result.details, "track": track}))

    profile = functools.cache(lambda: _tuple_profile(s))
    attempt("most_common_value", lambda: _most_common_value(s))
    if binary:
        attempt("collision", lambda: _collision(s))
        attempt("markov", lambda: _markov(s))
        attempt("compression", lambda: _compression(s))
    attempt("t_tuple", lambda: _t_tuple(profile()))
    attempt("lrs", lambda: _lrs(profile()))
    attempt("multi_mcw", lambda: _multi_mcw(s, k))
    attempt("lag", lambda: _lag(s, k))
    attempt("multi_mmc", lambda: _multi_mmc(s, k))
    attempt("lz78y", lambda: _lz78y(s, k))
    return results, skipped


def assess_non_iid(
    samples: npt.ArrayLike, bits_per_symbol: int, *, max_samples: int = BITSTRING_LIMIT
) -> Assessment:
    """SP 800-90B section 3.1.3 initial entropy estimate for a non-IID source.

    * Original track: the samples, mapped to a dense alphabet ``0..k-1``.
      If they are binary (k <= 2), all ten estimators run. Otherwise the
      seven defined for any alphabet run (MCV, t-tuple, LRS, MultiMCW,
      Lag, MultiMMC, LZ78Y). ``H_original`` is the minimum.
    * Bitstring track: the samples written MSB-first with
      ``bits_per_symbol`` bits each, truncated to the first ``max_samples``
      bits (10^6 by default, as 3.1.3 specifies). All ten estimators run, and
      ``H_bitstring`` is the minimum, in bits per bit.
    * ``H = min(H_original, bits_per_symbol * H_bitstring)``.

    For ``bits_per_symbol == 1`` the bitstring *is* the original sequence, so
    it is not assessed twice and ``h_bitstring = h_original``. The original
    track uses every sample given. Callers choose the window; the standard
    asks for at least 10^6 samples before certifying anything, and the
    estimators' confidence bounds shrink accordingly on less.
    """
    if not 1 <= bits_per_symbol <= MAX_BITS_PER_SYMBOL:
        raise ValueError(f"bits_per_symbol must be in 1..{MAX_BITS_PER_SYMBOL}")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    raw = _as_symbols(samples)
    n = int(raw.size)
    if n < 2:
        raise InsufficientDataError("assessment needs at least 2 samples")
    if int(np.min(raw)) < 0 or int(np.max(raw)) >= 1 << bits_per_symbol:
        raise ValueError(f"samples must lie in [0, 2**{bits_per_symbol})")
    s, k = _dense(raw)
    results, skipped = _assess_track(s, k, binary=k <= 2, track="original")
    h_original = min([float(bits_per_symbol)] + [r.min_entropy for r in results])
    n_bits = 0
    if bits_per_symbol == 1:
        h_bitstring = min(1.0, h_original)
    else:
        needed = -(-max_samples // bits_per_symbol)  # convert only what is assessed
        bits = to_bitstring(raw[:needed], bits_per_symbol)[:max_samples].astype(np.int64)
        n_bits = int(bits.size)
        bit_results, bit_skipped = _assess_track(bits, 2, binary=True, track="bitstring")
        results += bit_results
        skipped |= bit_skipped
        h_bitstring = min([1.0] + [r.min_entropy for r in bit_results])
    min_entropy = min(h_original, bits_per_symbol * h_bitstring)
    return Assessment(
        min_entropy=min_entropy,
        h_original=h_original,
        h_bitstring=h_bitstring,
        bits_per_symbol=bits_per_symbol,
        results=results,
        n_samples=n,
        n_bitstring=n_bits,
        skipped=skipped,
    )
