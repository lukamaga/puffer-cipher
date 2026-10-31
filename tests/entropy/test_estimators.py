"""SP 800-90B section 6.3 estimators: formulas, worked examples, known distributions.

Expected values are derived independently of the implementation:
closed forms by hand, scipy special functions and root finding, exact
dynamic programming, or brute-force counting. Statistical tolerances
reflect each estimator's documented downward bias (99% confidence bounds,
the max over tuple lengths, p_local's sensitivity to short lucky runs)
and are stated next to each assertion.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import gamma, gammaincc

from puffer.entropy import estimators as est
from puffer.entropy.estimators import (
    BINARY_ONLY,
    ESTIMATORS,
    Assessment,
    EstimatorResult,
    InsufficientDataError,
    assess_non_iid,
    collision,
    compression,
    lag,
    lrs,
    lz78y,
    markov,
    most_common_value,
    multi_mcw,
    multi_mmc,
    prediction_estimate,
    quick_min_entropy,
    solve_p_local,
    t_tuple,
    to_bitstring,
)

NON_BINARY = [name for name in ESTIMATORS if name not in BINARY_ONLY]


def _estimate(data: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: ESTIMATORS[name](data).min_entropy for name in names}


# --------------------------------------------------------------------------
# 6.3.1 MCV
# --------------------------------------------------------------------------


def test_mcv_hand_example() -> None:
    s = [0] * 30 + [1] * 25 + [2] * 25 + [3] * 20  # mode 0 occurs 30 times out of 100
    p_u = 0.3 + 2.576 * math.sqrt(0.3 * 0.7 / 99)
    r = most_common_value(s)
    assert r.min_entropy == pytest.approx(-math.log2(p_u), rel=1e-12)
    assert r.p_upper == pytest.approx(p_u, rel=1e-12)
    assert r.details == {"p_hat": 0.3, "n": 100}
    # With 10 samples the bound 0.6 + 2.576 * sqrt(0.24 / 9) exceeds 1: p_u = 1, H = 0.
    assert most_common_value([0, 1, 1, 2, 1, 0, 1, 1, 3, 1]).min_entropy == 0.0


def test_quick_min_entropy_is_mcv_and_safe_on_tiny_input(rng: np.random.Generator) -> None:
    x = rng.integers(0, 256, 4096).astype(np.uint8)
    assert quick_min_entropy(x) == most_common_value(x).min_entropy
    assert quick_min_entropy(np.zeros(1, dtype=np.uint8)) == 0.0
    assert quick_min_entropy([]) == 0.0


def test_symbol_values_and_dtypes_do_not_matter(rng: np.random.Generator) -> None:
    x = rng.integers(0, 6, 3000)
    # Order-preserving relabel: MultiMMC and LZ78Y break ties towards the larger symbol.
    relabelled = np.array([-7, 3, 42, 1000, 5000, 99_999_999])[x]
    for name in NON_BINARY:
        assert ESTIMATORS[name](x).min_entropy == pytest.approx(
            ESTIMATORS[name](relabelled).min_entropy, abs=1e-12
        ), name
    bits = rng.integers(0, 2, 7000)
    assert markov(bits.astype(bool)).min_entropy == markov(bits).min_entropy
    with pytest.raises(TypeError):
        most_common_value(np.array([0.5, 1.5]))


# --------------------------------------------------------------------------
# 6.3.2 Collision
# --------------------------------------------------------------------------


def _standard_collision_expectation(p: float) -> float:
    """6.3.2 step 7 right-hand side, with F(1/z) = Gamma(3, z) z^-3 e^z via scipy."""
    q = 1.0 - p
    z = 1.0 / q
    f = gammaincc(3, z) * gamma(3) * z**-3 * math.exp(z)
    return p / q**2 * (1 + 0.5 * (1 / p - 1 / q)) * f - p / q * 0.5 * (1 / p - 1 / q)


def test_collision_solves_the_standard_equation(rng: np.random.Generator) -> None:
    bits = (rng.random(20_000) < 0.7).astype(np.int64)
    # Independent collision times: windows of 2 or 3 bits.
    times, i = [], 0
    while i + 1 < bits.size:
        t = 2 if bits[i] == bits[i + 1] else 3
        if i + t > bits.size:
            break
        times.append(t)
        i += t
    mean, sigma = float(np.mean(times)), float(np.std(times, ddof=1))
    lower = mean - 2.576 * sigma / math.sqrt(len(times))
    p = brentq(lambda x: _standard_collision_expectation(x) - lower, 0.5 + 1e-6, 0.95)
    r = collision(bits)
    assert r.details["v"] == len(times)
    assert r.details["mean_lower"] == pytest.approx(lower, rel=1e-12)
    assert r.details["p"] == pytest.approx(p, rel=1e-9)
    assert r.min_entropy == pytest.approx(-math.log2(p), rel=1e-9)
    # Sanity of the model: a 0.7-coin's expected collision time is 2 + 2pq.
    assert _standard_collision_expectation(0.7) == pytest.approx(2 + 2 * 0.7 * 0.3, rel=1e-12)


def test_collision_edge_cases() -> None:
    assert collision(np.zeros(1000, dtype=np.int64)).min_entropy == 0.0  # X' = 2 -> p = 1
    alternating = np.tile([0, 1], 500)  # every window is 3 bits: X' = 3 > 2.5, no solution
    r = collision(alternating)
    assert r.min_entropy == 1.0
    assert r.details["p"] == 0.5
    with pytest.raises(InsufficientDataError):
        collision([0, 1, 1])
    with pytest.raises(ValueError, match="binary"):
        collision([0, 1, 2, 0, 1, 2])


# --------------------------------------------------------------------------
# 6.3.3 Markov
# --------------------------------------------------------------------------


def test_markov_hand_example() -> None:
    s = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    # zeros 6/10; from 0: 0->0 three times, 0->1 twice; from 1: two of each.
    r = markov(s)
    assert (r.details["p0"], r.details["p00"], r.details["p01"]) == pytest.approx((0.6, 0.6, 0.4))
    assert (r.details["p10"], r.details["p11"]) == pytest.approx((0.5, 0.5))
    # 00...0 has probability 0.6 * 0.6^127, the largest of the six sequences.
    assert r.min_entropy == pytest.approx(-math.log2(0.6), rel=1e-12)


def test_markov_alternating_and_never_left_states() -> None:
    # 0101...: P0 = 1/2, P01 = P10 = 1, so 0101...01 has probability 1/2.
    assert markov(np.tile([0, 1], 500)).min_entropy == pytest.approx(1 / 128, rel=1e-12)
    # A single trailing 1 is never left: its transition probabilities are 0.
    r = markov([0] * 99 + [1])
    assert r.details["p10"] == r.details["p11"] == 0.0
    assert r.min_entropy == pytest.approx(-math.log2(0.99) / 128 - 127 / 128 * math.log2(98 / 99))
    assert markov([0, 1]).min_entropy == 1.0  # every 128-bit sequence is ruled out


# --------------------------------------------------------------------------
# 6.3.4 Compression
# --------------------------------------------------------------------------


def _brute_g(z: float, n_blocks: int, d: int) -> float:
    """G(z) exactly as the double sum printed in 6.3.4 step 7."""
    total = 0.0
    for t in range(d + 1, n_blocks + 1):
        u = np.arange(1, t + 1, dtype=np.float64)
        f = np.where(u < t, z * z * (1 - z) ** (u - 1), z * (1 - z) ** (t - 1))
        total += float(np.sum(np.log2(u) * f))
    return total / (n_blocks - d)


@pytest.mark.parametrize("p", [1 / 64, 0.03, 0.2, 0.7, 0.99, 1.0])
def test_compression_expectation_matches_double_sum(p: float) -> None:
    n_blocks, d = 1300, 1000
    expected = _brute_g(p, n_blocks, d) + 63 * _brute_g((1 - p) / 63, n_blocks, d)
    got = est._compression_expectation(n_blocks, d, 6)(p)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_compression_solution_satisfies_step_7(rng: np.random.Generator) -> None:
    bits = (rng.random(60_000) < 0.75).astype(np.int64)
    r = compression(bits)
    expectation = est._compression_expectation(r.details["blocks"], 1000, 6)
    assert expectation(r.details["p"]) == pytest.approx(r.details["mean_lower"], rel=1e-9)
    assert r.min_entropy == pytest.approx(-math.log2(r.details["p"]) / 6, rel=1e-12)
    # Independent statistic: mean of log2 distances to the previous equal block.
    blocks = bits[: r.details["blocks"] * 6].reshape(-1, 6) @ (1 << np.arange(5, -1, -1))
    last: dict[int, int] = {}
    logs = []
    for i, b in enumerate(blocks.tolist(), start=1):
        if i > 1000:
            logs.append(math.log2(i - last.get(b, 0)))
        last[b] = i
    assert r.details["mean"] == pytest.approx(float(np.mean(logs)), rel=1e-12)


def test_compression_edge_cases(rng: np.random.Generator) -> None:
    assert compression(np.zeros(7000, dtype=np.int64)).min_entropy == 0.0
    with pytest.raises(InsufficientDataError):
        compression(rng.integers(0, 2, 6000))


# --------------------------------------------------------------------------
# 6.3.5 t-Tuple and 6.3.6 LRS
# --------------------------------------------------------------------------

STANDARD_EXAMPLE = [2, 2, 0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 1, 2, 1, 0, 0, 1, 0, 0, 0]


def test_t_tuple_counts_standard_example() -> None:
    """The example of 6.3.5 step 2: Q[1] = 9 (zeros) and Q[2] = 4 (the pair 01)."""
    s, _ = est._dense(STANDARD_EXAMPLE)
    profile = est._tuple_profile(s)
    assert profile.max_count[1] == 9
    assert profile.max_count[2] == 4


def test_t_tuple_hand_example() -> None:
    s = [0] * 40 + [1] * 40  # Q[i] = 41 - i: t = 6 is the last length with Q >= 35
    p_max = max(((41 - i) / (81 - i)) ** (1 / i) for i in range(1, 7))
    p_u = p_max + 2.576 * math.sqrt(p_max * (1 - p_max) / 79)
    r = t_tuple(s)
    assert r.details["t"] == 6
    assert r.min_entropy == pytest.approx(-math.log2(p_u), rel=1e-12)


def test_lrs_matches_brute_force(rng: np.random.Generator) -> None:
    x = rng.integers(0, 3, 3000)
    x[2000:2040] = x[100:140]  # one long repeat
    n = x.size
    counts = {w: Counter(tuple(x[i : i + w]) for i in range(n - w + 1)) for w in range(1, 45)}
    u = next(w for w in counts if max(counts[w].values()) < 35)
    v = max(w for w in counts if max(counts[w].values()) >= 2)
    p_max = max(
        (sum(c * (c - 1) / 2 for c in counts[w].values()) / math.comb(n - w + 1, 2)) ** (1 / w)
        for w in range(u, v + 1)
    )
    p_u = p_max + 2.576 * math.sqrt(p_max * (1 - p_max) / (n - 1))
    r = lrs(x)
    assert (r.details["u"], r.details["v"]) == (u, v)
    assert v >= 40
    assert r.min_entropy == pytest.approx(-math.log2(p_u), rel=1e-12)


def test_tuple_estimates_need_enough_repeats(rng: np.random.Generator) -> None:
    short = rng.permutation(256)
    with pytest.raises(InsufficientDataError):
        t_tuple(short)  # no symbol occurs 35 times
    with pytest.raises(InsufficientDataError):
        lrs(short)  # nothing repeats at all


# --------------------------------------------------------------------------
# Shared prediction machinery
# --------------------------------------------------------------------------


def _exact_no_run(p: float, r: int, n: int) -> float:
    """P(no run of r successes in n Bernoulli(p) trials), exact DP over the current run."""
    state = np.zeros(r)
    state[0] = 1.0
    for _ in range(n):
        nxt = np.empty(r)
        nxt[0] = state.sum() * (1 - p)
        nxt[1:] = state[:-1] * p
        state = nxt
    return float(state.sum())


@pytest.mark.parametrize(
    ("n", "run", "rel"),
    [(1000, 0, 1e-9), (1000, 9, 1e-9), (5000, 3, 1e-9), (2000, 20, 1e-9), (30, 29, 3e-3)],
)
def test_p_local_matches_exact_longest_run_distribution(n: int, run: int, rel: float) -> None:
    exact = brentq(lambda p: _exact_no_run(p, run + 1, n) - 0.99, 1e-15, 1 - 1e-15, xtol=1e-16)
    got = solve_p_local(n, run)
    assert got == pytest.approx(exact, rel=rel)
    assert got >= exact * (1 - 1e-9)  # the approximation never under-states p_local


def test_p_local_closed_cases() -> None:
    assert solve_p_local(1, 0) == pytest.approx(0.01, rel=1e-12)  # P(fail once) = 0.99
    assert solve_p_local(1000, 0) == pytest.approx(1 - 0.99 ** (1 / 1000), rel=1e-9)
    assert solve_p_local(10, 10) == 1.0  # all correct: no p < 1 explains it
    with pytest.raises(ValueError, match="longest_run"):
        solve_p_local(10, 11)


def test_prediction_estimate_global_bounds() -> None:
    never = prediction_estimate(np.zeros(1000, dtype=bool), 256)
    assert never.details["p_global_upper"] == pytest.approx(1 - 0.01 ** (1 / 1000))
    assert never.min_entropy == pytest.approx(-math.log2(1 - 0.01 ** (1 / 1000)))
    # With enough evidence the bound drops below 1/k and the floor takes over.
    assert prediction_estimate(np.zeros(10_000, dtype=bool), 256).min_entropy == 8.0
    always = prediction_estimate(np.ones(1000, dtype=bool), 2)
    assert always.min_entropy == 0.0
    half = prediction_estimate(np.tile([True, False], 5000), 2)
    p = 0.5 + 2.576 * math.sqrt(0.25 / 9999)
    assert half.details["longest_run"] == 1
    assert half.min_entropy == pytest.approx(-math.log2(p), rel=1e-12)
    burst = np.zeros(10_000, dtype=bool)
    burst[5000:5030] = True  # a 30-long lucky streak dominates a 0.3% hit rate
    r = prediction_estimate(burst, 2)
    assert r.details["p_local"] > 0.5 > r.details["p_global_upper"]
    assert r.min_entropy == pytest.approx(-math.log2(r.details["p_local"]))


# --------------------------------------------------------------------------
# Known distributions
# --------------------------------------------------------------------------


def test_constant_data_has_zero_entropy_everywhere() -> None:
    bits = np.zeros(20_000, dtype=np.uint8)
    assert _estimate(bits, list(ESTIMATORS)) == dict.fromkeys(ESTIMATORS, 0.0)
    symbols = np.full(20_000, 7, dtype=np.uint8)
    assert _estimate(symbols, NON_BINARY) == dict.fromkeys(NON_BINARY, 0.0)


def test_uniform_bytes_close_to_but_not_above_8_bits(rng: np.random.Generator) -> None:
    # Tolerance: at n = 50 000 the 99% bound alone costs ~0.25 bit for MCV.
    # Predictors lose up to ~0.6 bit to P_local: a chance run of 2 correct
    # guesses (expected ~0.8 times) gives (0.01 / N)^(1/3) = 2^-7.4.
    for name, h in _estimate(rng.integers(0, 256, 50_000), NON_BINARY).items():
        assert 7.25 < h <= 8.0, name


def test_uniform_bits_close_to_but_not_above_1_bit(rng: np.random.Generator) -> None:
    # Floors at n = 200 000 reflect known biases: collision's sqrt makes it
    # ~0.87, compression ~0.8, t-tuple's max over lengths ~0.92.
    floors = {
        "most_common_value": 0.98,
        "collision": 0.85,
        "markov": 0.98,
        "compression": 0.72,
        "t_tuple": 0.9,
        "lrs": 0.95,
        "multi_mcw": 0.97,
        "lag": 0.97,
        "multi_mmc": 0.97,
        "lz78y": 0.97,
    }
    for name, h in _estimate(rng.integers(0, 2, 200_000), list(ESTIMATORS)).items():
        assert floors[name] < h <= 1.0, name


def test_biased_iid_symbols(rng: np.random.Generator) -> None:
    probs = [0.5, 0.2, 0.2, 0.1]  # min-entropy exactly 1 bit
    h = _estimate(rng.choice(4, 50_000, p=probs), NON_BINARY)
    for name in ("most_common_value", "t_tuple", "multi_mcw", "multi_mmc", "lz78y"):
        assert 0.95 < h[name] <= 1.0, name
    # Lag guesses an earlier sample, which is right with the collision
    # probability sum(p^2) = 0.34; LRS also measures collision entropy.
    # Both sit near -log2(0.34) = 1.56 and the minimum is carried by MCV.
    collision_entropy = -math.log2(sum(p * p for p in probs))
    for name in ("lag", "lrs"):
        assert 1.25 < h[name] <= collision_entropy + 0.02, name


def test_biased_iid_bits(rng: np.random.Generator) -> None:
    h_true = -math.log2(0.8)
    h = _estimate((rng.random(100_000) < 0.8).astype(np.int64), list(ESTIMATORS))
    for name in ("most_common_value", "collision", "t_tuple", "multi_mcw", "multi_mmc", "lz78y"):
        assert h_true - 0.02 < h[name] <= h_true, name
    assert h_true - 0.02 < h["markov"] < h_true + 0.01  # no confidence bound in 6.3.3
    assert h["compression"] <= h_true  # very conservative for biased bits
    for name in ("lag", "lrs"):  # collision entropy -log2(0.68) = 0.556
        assert 0.5 < h[name] <= 0.57, name


def test_sticky_markov_chain_is_caught(rng: np.random.Generator) -> None:
    bits = np.cumsum(rng.random(100_000) < 0.1) % 2  # P(stay) = 0.9, balanced marginals
    h = _estimate(bits, list(ESTIMATORS))
    rate = -math.log2(0.9)  # 0.152 bits per bit
    assert h["most_common_value"] > 0.95  # the histogram sees a fair coin
    assert (1 + 127 * rate) / 128 - 0.01 < h["markov"] < (1 + 127 * rate) / 128 + 0.02
    for name in ("lag", "multi_mmc", "lz78y"):
        assert rate - 0.02 < h[name] <= rate + 0.01, name
    for name in ("collision", "compression", "t_tuple"):
        assert h[name] < 0.25, name


def test_periodic_sequence_is_fully_predictable(rng: np.random.Generator) -> None:
    s = np.tile(rng.permutation(5), 4000)
    h = _estimate(s, NON_BINARY)
    assert h["most_common_value"] > 2.2  # ~log2(5): the histogram is flat
    for name in ("lag", "multi_mmc", "lz78y", "t_tuple", "lrs"):
        assert h[name] < 0.01, name


def test_repeated_blocks_are_caught(rng: np.random.Generator) -> None:
    s = np.tile(rng.integers(0, 256, 1000), 40)  # a replayed 1000-sample recording
    h = _estimate(s, NON_BINARY)
    assert h["most_common_value"] > 6.0
    for name in ("t_tuple", "lrs", "multi_mmc"):
        assert h[name] < 0.05, name


def test_minimum_lengths() -> None:
    with pytest.raises(InsufficientDataError):
        most_common_value([1])
    with pytest.raises(InsufficientDataError):
        multi_mcw(np.zeros(64, dtype=np.int64))
    assert multi_mcw(np.zeros(65, dtype=np.int64)).details["n"] == 2
    with pytest.raises(InsufficientDataError):
        lag([0, 1])
    with pytest.raises(InsufficientDataError):
        multi_mmc([0, 1, 0])
    with pytest.raises(InsufficientDataError):
        lz78y(np.zeros(18, dtype=np.int64))
    assert lz78y(np.zeros(19, dtype=np.int64)).details["n"] == 2


def test_registry_names_match_results(rng: np.random.Generator) -> None:
    bits = rng.integers(0, 2, 8000)
    for name, estimator in ESTIMATORS.items():
        r = estimator(bits)
        assert isinstance(r, EstimatorResult)
        assert r.name == name
        assert r.p_upper == pytest.approx(2.0**-r.min_entropy)


# --------------------------------------------------------------------------
# Section 3.1.3 assessment
# --------------------------------------------------------------------------


def test_to_bitstring_msb_first() -> None:
    assert to_bitstring([5, 1, 6], 3).tolist() == [1, 0, 1, 0, 0, 1, 1, 1, 0]
    assert to_bitstring(np.array([0xA5], dtype=np.uint8), 8).tolist() == [1, 0, 1, 0, 0, 1, 0, 1]
    assert to_bitstring([], 4).size == 0
    with pytest.raises(ValueError, match="must lie in"):
        to_bitstring([8], 3)
    with pytest.raises(ValueError, match="must lie in"):
        to_bitstring([-1], 3)
    with pytest.raises(ValueError, match="bits_per_symbol"):
        to_bitstring([1], 0)


def test_assessment_combines_tracks(rng: np.random.Generator) -> None:
    a = assess_non_iid(rng.integers(0, 256, 20_000), 8, max_samples=60_000)
    original, bitstring = a.track("original"), a.track("bitstring")
    assert list(original) == NON_BINARY
    assert list(bitstring) == list(ESTIMATORS)
    assert a.h_original == min(r.min_entropy for r in original.values())
    assert a.h_bitstring == min(r.min_entropy for r in bitstring.values())
    assert a.min_entropy == min(a.h_original, 8 * a.h_bitstring)
    assert (a.n_samples, a.n_bitstring, a.bits_per_symbol) == (20_000, 60_000, 8)
    assert all(r.details["n"] <= 60_000 for r in bitstring.values() if "n" in r.details)
    assert a.skipped == {}
    json.dumps(a.to_dict())  # report/API ready


def test_bitstring_track_can_dominate(rng: np.random.Generator) -> None:
    nibble = rng.integers(0, 16, 20_000)
    a = assess_non_iid((nibble << 4) | nibble, 8)  # the low nibble copies the high one
    # 16 equiprobable symbols and no visible dependence: about 4 bits, minus the
    # 99% bounds and Lag's sensitivity to one lucky run of guesses.
    assert a.h_original > 3.4
    # Bitwise, Lag 4 predicts the low nibble exactly: P(correct) = 3/4 per bit.
    assert a.track("bitstring")["lag"].min_entropy < 0.43
    assert a.min_entropy == 8 * a.h_bitstring < a.h_original


def test_original_track_can_dominate(rng: np.random.Generator) -> None:
    x = np.where(rng.random(20_000) < 0.5, 0x5A, rng.integers(0, 256, 20_000))
    a = assess_non_iid(x, 8)
    assert a.min_entropy == a.h_original  # p_max = 0.5 + 0.5 / 256: just under 1 bit
    assert 0.9 < a.h_original < 1.0 < 8 * a.h_bitstring


def test_binary_input_runs_all_ten_once(rng: np.random.Generator) -> None:
    bits = rng.integers(0, 2, 20_000)
    a = assess_non_iid(bits, 1)
    assert list(a.track("original")) == list(ESTIMATORS)
    assert a.track("bitstring") == {}
    assert a.h_bitstring == a.h_original == a.min_entropy
    assert a.n_bitstring == 0


def test_two_valued_bytes_get_binary_estimators_on_both_tracks(rng: np.random.Generator) -> None:
    a = assess_non_iid(np.where(rng.random(10_000) < 0.5, 0x00, 0xFF), 8)
    assert list(a.track("original")) == list(ESTIMATORS)
    assert list(a.track("bitstring")) == list(ESTIMATORS)
    # Each byte is one coin written 8 times. Aligned 2-bit windows always collide,
    # so the bitstring collision estimate is exactly 0 and so is the assessment:
    # the bitstring track refuses to credit duplicated bits.
    assert a.track("bitstring")["collision"].min_entropy == 0.0
    assert a.min_entropy == 0.0 < 0.9 < a.track("original")["most_common_value"].min_entropy


def test_short_input_skips_instead_of_failing(rng: np.random.Generator) -> None:
    a = assess_non_iid(rng.integers(0, 256, 50), 8)
    # No byte occurs 35 times; MultiMCW needs 65 samples; compression > 6006 bits.
    assert {"original/t_tuple", "original/multi_mcw", "bitstring/compression"} <= set(a.skipped)
    assert "original/most_common_value" not in a.skipped
    assert len(a.results) + len(a.skipped) == 7 + 10
    assert 0.0 <= a.min_entropy <= 8.0


@pytest.mark.parametrize(
    ("samples", "bits", "error", "match"),
    [
        ([0, 1, 2], 0, ValueError, "bits_per_symbol"),
        ([0, 1, 2], 9, ValueError, "bits_per_symbol"),
        ([0, 4], 2, ValueError, "must lie in"),
        ([0, -1], 2, ValueError, "must lie in"),
        ([1], 2, InsufficientDataError, "at least 2"),
        ([0.0, 1.0], 2, TypeError, "integer symbols"),
    ],
)
def test_assessment_validates_input(
    samples: list[float], bits: int, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        assess_non_iid(samples, bits)


def test_assessment_rejects_bad_max_samples() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        assess_non_iid([0, 1, 2], 2, max_samples=0)


@pytest.mark.slow
def test_full_assessment_of_100k_bytes_is_fast_and_sane(rng: np.random.Generator) -> None:
    import time

    start = time.perf_counter()
    a = assess_non_iid(rng.integers(0, 256, 100_000).astype(np.uint8), 8)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0
    assert isinstance(a, Assessment)
    assert a.n_bitstring == 800_000
    assert 7.5 < a.h_original <= 8.0
    # Compression is the binding bitstring estimator for full-entropy data (~0.83 at 10^6 bits).
    assert 0.8 < a.h_bitstring <= 1.0
    assert a.min_entropy == min(a.h_original, 8 * a.h_bitstring)


@pytest.mark.slow
def test_bitstring_track_is_capped_at_one_million_bits(rng: np.random.Generator) -> None:
    a = assess_non_iid(rng.integers(0, 2**6, 200_000), 6)
    assert a.n_bitstring == est.BITSTRING_LIMIT
    assert all(r.details.get("n", 0) <= est.BITSTRING_LIMIT for r in a.track("bitstring").values())
