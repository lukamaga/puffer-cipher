"""Vectorised internals against straightforward reference loops, and behaviour.

Each heavy test is implemented with numpy tricks (window histograms, packed
GF(2) rows, sorted last-occurrence tables, ...). Here every trick is checked
against a literal transcription of the spec / NIST STS algorithm on random
inputs, and the suite's behaviour on random, biased and structured data is
pinned.
"""

from __future__ import annotations

import json
import math
import time
from itertools import product

import numpy as np
import pytest
from scipy import special

from puffer.analysis import sp800_22 as sp


def random_bits(rng: np.random.Generator, n: int, p_one: float = 0.5) -> np.ndarray:
    return (rng.random(n) < p_one).astype(np.uint8)


# --------------------------------------------------------------------------
# Reference loops (transcriptions of NIST STS 2.1.2)
# --------------------------------------------------------------------------


def naive_non_overlapping_counts(eps, template: str, n_blocks: int) -> list[int]:
    m, block = len(template), len(eps) // n_blocks
    t = [int(c) for c in template]
    counts = []
    for i in range(n_blocks):
        w, j = 0, 0
        while j < block - m + 1:
            if list(eps[i * block + j : i * block + j + m]) == t:
                w += 1
                j += m
            else:
                j += 1
        counts.append(w)
    return counts


def naive_longest_run(block) -> int:
    best = run = 0
    for b in block:
        run = run + 1 if b else 0
        best = max(best, run)
    return best


def naive_gf2_rank(matrix) -> int:
    rows = [int("".join(str(int(b)) for b in row), 2) for row in matrix]
    rank = 0
    for bit in reversed(range(len(matrix[0]))):
        pivot = next((r for r in rows if r >> bit & 1), None)
        if pivot is None:
            continue
        rows.remove(pivot)
        rows = [r ^ pivot if r >> bit & 1 else r for r in rows]
        rank += 1
    return rank


def naive_berlekamp_massey(s) -> int:
    n = len(s)
    c, b = [0] * n, [0] * n
    c[0] = b[0] = 1
    length, m = 0, -1
    for k in range(n):
        d = s[k]
        for i in range(1, length + 1):
            d ^= c[i] & s[k - i]
        if d:
            t = c[:]
            for j in range(n - k + m):
                c[k - m + j] ^= b[j]
            if length <= k // 2:
                length, m, b = k + 1 - length, k, t
    return length


def naive_universal_fn(eps, length: int, q: int) -> float:
    k = len(eps) // length - q
    table = [0] * (1 << length)
    value = [int("".join(map(str, eps[i * length : (i + 1) * length])), 2) for i in range(q + k)]
    for i in range(1, q + 1):
        table[value[i - 1]] = i
    total = 0.0
    for i in range(q + 1, q + k + 1):
        total += math.log2(i - table[value[i - 1]])
        table[value[i - 1]] = i
    return total / k


def naive_cyclic_counts(eps, m: int) -> dict[str, int]:
    n = len(eps)
    counts = {"".join(p): 0 for p in product("01", repeat=m)}
    for i in range(n):
        counts["".join(str(eps[(i + j) % n]) for j in range(m))] += 1
    return counts


def naive_excursion_counts(eps) -> tuple[int, dict[int, list[int]]]:
    walk, s = [], 0
    for b in eps:
        s += 2 * b - 1
        walk.append(s)
    cycles: list[list[int]] = []
    current: list[int] = []
    for v in walk:
        if v == 0:
            cycles.append(current)
            current = []
        else:
            current.append(v)
    if walk[-1] != 0:
        cycles.append(current)
    nu = {}
    for x in (-4, -3, -2, -1, 1, 2, 3, 4):
        visits = [min(c.count(x), 5) for c in cycles]
        nu[x] = [visits.count(k) for k in range(6)]
    return len(cycles), nu


# --------------------------------------------------------------------------
# Vectorised == reference
# --------------------------------------------------------------------------


def test_non_overlapping_counts_match_the_scan_and_skip_loop(rng):
    eps = random_bits(rng, 3000)
    r = sp.non_overlapping_template_matching(eps, template_length=4, n_blocks=4)
    assert r.labels == sp.aperiodic_templates(4)
    for template, counts in zip(r.labels, r.statistics["matches"], strict=True):
        assert counts == naive_non_overlapping_counts(eps, template, 4)


def test_non_overlapping_rejects_periodic_templates():
    with pytest.raises(ValueError, match="aperiodic"):
        sp.non_overlapping_template_matching(
            np.zeros(100, np.uint8), template_length=3, templates=["101"]
        )


@pytest.mark.parametrize("n", [128, 6272, 750_000])
def test_longest_run_classes_match_a_loop(rng, n):
    eps = random_bits(rng, n + 37)
    r = sp.longest_run_of_ones(eps)
    m = r.statistics["block_size"]
    lo, hi = r.statistics["class_bounds"]
    blocks = eps[: (len(eps) // m) * m].reshape(-1, m)
    classes = [min(max(naive_longest_run(b), lo), hi) - lo for b in blocks]
    assert r.statistics["counts"] == [classes.count(k) for k in range(hi - lo + 1)]


def test_gf2_rank_matches_textbook_elimination(rng):
    for shape in [(50, 32, 32), (20, 5, 9), (20, 9, 5), (10, 64, 64)]:
        mats = rng.integers(0, 2, size=shape, dtype=np.uint8)
        mats[0] = 0  # rank 0
        mats[1, 1:] = mats[1, 0]  # rank <= 1
        mats[2, -1] = mats[2, 0] ^ mats[2, 1]  # a dependent row
        expected = [naive_gf2_rank(m) for m in mats]
        assert sp.gf2_rank(mats).tolist() == expected


def test_rank_probabilities_for_32x32_match_sts():
    r = sp.binary_matrix_rank(np.zeros(1024, dtype=np.uint8))
    assert r.statistics["probabilities"] == pytest.approx([0.2888, 0.5776, 0.1336], abs=1e-4)


def test_berlekamp_massey_matches_textbook_version(rng):
    for n in (1, 2, 17, 100, 500):
        for _ in range(5):
            s = random_bits(rng, n).tolist()
            assert sp.berlekamp_massey(s) == naive_berlekamp_massey(s)


def test_berlekamp_massey_known_complexities():
    # A maximal-length LFSR s_k = s_{k-3} ^ s_{k-5} has linear complexity 5.
    s = [1, 0, 0, 1, 1]
    while len(s) < 62:
        s.append(s[-3] ^ s[-5])
    assert sp.berlekamp_massey(s) == 5
    assert sp.berlekamp_massey([0] * 40) == 0
    assert sp.berlekamp_massey([0] * 39 + [1]) == 40


def test_universal_fn_matches_the_sts_table_loop(rng):
    eps = random_bits(rng, 3 * (80 + 600) + 2)
    r = sp.universal(eps, block_length=3, init_blocks=80)
    assert r.statistics["fn"] == pytest.approx(naive_universal_fn(eps.tolist(), 3, 80), rel=1e-12)


def test_dft_counts_match_a_direct_fourier_sum(rng):
    eps = random_bits(rng, 1000)
    n = len(eps)
    x = 2.0 * eps - 1.0
    k = np.arange(n // 2)[:, np.newaxis]
    direct = np.abs(np.sum(x * np.exp(-2j * np.pi * k * np.arange(n) / n), axis=1))
    r = sp.discrete_fourier_transform(eps)
    assert r.statistics["observed"] == int(np.count_nonzero(direct < math.sqrt(math.log(20) * n)))


@pytest.mark.parametrize("m", [2, 3, 5])
def test_serial_and_apen_use_cyclic_pattern_counts(rng, m):
    eps = random_bits(rng, 300)
    n = len(eps)
    eps_list = eps.tolist()

    def psi2(k: int) -> float:
        if k <= 0:
            return 0.0
        counts = naive_cyclic_counts(eps_list, k).values()
        return 2**k / n * sum(c * c for c in counts) - n

    def phi(k: int) -> float:
        return sum(c / n * math.log(c / n) for c in naive_cyclic_counts(eps_list, k).values() if c)

    r = sp.serial(eps, block_length=m)
    assert r.statistics["psi_squared"] == pytest.approx([psi2(m), psi2(m - 1), psi2(m - 2)])
    a = sp.approximate_entropy(eps, block_length=m)
    assert a.statistics["apen"] == pytest.approx(phi(m) - phi(m + 1))


def test_random_excursion_counts_match_cycle_splitting(rng):
    for n in (50, 999, 5000):
        eps = random_bits(rng, n)
        r = sp.random_excursions(eps, min_cycles=1)
        cycles, nu = naive_excursion_counts(eps.tolist())
        assert r.statistics["cycles"] == cycles
        assert r.statistics["counts"] == {f"x={x:+d}": v for x, v in nu.items()}


def test_random_excursion_cycles_when_the_walk_ends_at_zero():
    eps = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)  # S = 1, 0, -1, 0, 1, 0
    r = sp.random_excursions_variant(eps, min_cycles=1)
    assert r.statistics["cycles"] == 3
    assert r.statistics["visits"][8:10] == [1, 2]  # x = -1, +1


def test_overlapping_probabilities_match_brute_force_enumeration():
    m, block, k = 3, 12, 3
    counts = np.zeros(k + 1)
    for word in product((0, 1), repeat=block):
        hits = sum(all(word[j : j + m]) for j in range(block - m + 1))
        counts[min(hits, k)] += 1
    expected = counts / 2**block
    assert sp.overlapping_template_probabilities(m, block, k) == pytest.approx(expected.tolist())
    assert sp.overlapping_template_probabilities(1, 5, 2) == pytest.approx(
        [1 / 32, 5 / 32, 26 / 32]
    )


def test_overlapping_uses_exact_probabilities_for_non_reference_parameters(rng):
    eps = random_bits(rng, 20_000)
    r = sp.overlapping_template_matching(
        eps, template_length=4, block_size=100, degrees_of_freedom=4
    )
    assert r.statistics["probabilities"] == pytest.approx(
        list(sp.overlapping_template_probabilities(4, 100, 4))
    )
    with pytest.raises(ValueError, match="class probabilities"):
        sp.overlapping_template_matching(eps, probabilities=[0.5, 0.5])


def test_cusum_summation_bounds_match_real_valued_ceil_and_floor():
    def reference(n: int, z: int) -> float:
        s1 = sum(
            special.ndtr((4 * k + 1) * z / math.sqrt(n))
            - special.ndtr((4 * k - 1) * z / math.sqrt(n))
            for k in range(math.ceil((-n / z + 1) / 4), math.floor((n / z - 1) / 4) + 1)
        )
        s2 = sum(
            special.ndtr((4 * k + 3) * z / math.sqrt(n))
            - special.ndtr((4 * k + 1) * z / math.sqrt(n))
            for k in range(math.ceil((-n / z - 3) / 4), math.floor((n / z - 1) / 4) + 1)
        )
        return 1.0 - s1 + s2

    for n, z in [(10, 4), (100, 16), (100, 19), (1000, 1), (1000, 37), (1000, 1000), (12345, 250)]:
        assert sp._cusum_p_value(n, z) == pytest.approx(reference(n, z), abs=1e-12)


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


def test_random_megabit_runs_the_full_suite_quickly(rng):
    eps = random_bits(rng, 1_000_000)
    start = time.perf_counter()
    suite = sp.run_suite(eps)
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0  # ~1 s on a laptop; generous for CI
    assert {r.name for r in suite.applicable} >= set(sp.SUITE_TESTS) - {
        "random_excursions",
        "random_excursions_variant",  # need J >= 500 cycles: absent for ~1/3 of random inputs
    }
    assert suite.p_value_count >= 162
    assert suite.failed_count <= 6
    assert suite.failures_tail_probability() > 0.01
    assert suite.expected_false_failures == pytest.approx(0.01 * suite.p_value_count)


@pytest.mark.parametrize(
    ("name", "eps"),
    [
        ("biased", (np.random.default_rng(7).random(200_000) < 0.51).astype(np.uint8)),
        ("periodic", np.tile(np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8), 40_000)),
        ("zeros", np.zeros(200_000, dtype=np.uint8)),
    ],
)
def test_obviously_non_random_input_fails(name, eps):
    suite = sp.run_suite(eps)
    assert not suite.passed, name
    assert not suite["frequency"].passed or not suite["runs"].passed, name
    assert suite.failures_tail_probability() < 0.01, name


def test_biased_input_fails_frequency_and_the_runs_prerequisite():
    eps = (np.random.default_rng(3).random(100_000) < 0.53).astype(np.uint8)
    assert not sp.frequency(eps).passed
    r = sp.runs(eps)
    assert r.p_values == [0.0]
    assert r.statistics["prerequisite_passed"] is False
    assert not r.passed


def test_alternating_bits_fail_runs_and_spectral():
    eps = np.tile(np.array([0, 1], dtype=np.uint8), 5000)
    assert sp.frequency(eps).passed  # perfectly balanced ...
    assert not sp.runs(eps).passed  # ... but far too many runs
    assert not sp.discrete_fourier_transform(eps).passed


def test_run_suite_skips_tests_below_their_recommended_size(rng):
    suite = sp.run_suite(random_bits(rng, 20_000))
    skipped = {r.name: r.skipped for r in suite.skipped}
    assert set(skipped) == {
        "binary_matrix_rank",
        "approximate_entropy",
        "overlapping_template_matching",
        "universal",
        "random_excursions",
        "random_excursions_variant",
        "serial",
        "linear_complexity",
    }
    assert "SP 800-22 2.8.7" in str(skipped["overlapping_template_matching"])
    assert "387,840" in str(skipped["universal"])
    for r in suite.skipped:
        assert r.p_values == []
        assert not r.passed
        assert not r.applicable


def test_run_suite_without_size_enforcement_runs_everything_computable(rng):
    suite = sp.run_suite(random_bits(rng, 20_000), enforce_min_length=False)
    assert {r.name for r in suite.skipped} <= {
        "universal",
        "random_excursions",
        "random_excursions_variant",
    }
    assert "387,840" in str(suite["universal"].skipped)


def test_run_suite_subset_order_and_validation(rng):
    suite = sp.run_suite(random_bits(rng, 1000), tests=["runs", "frequency"])
    assert [r.name for r in suite.results] == ["frequency", "runs"]
    with pytest.raises(KeyError):
        suite["serial"]
    with pytest.raises(ValueError, match="unknown SP 800-22 tests"):
        sp.run_suite(random_bits(rng, 1000), tests=["dieharder"])
    with pytest.raises(ValueError, match="alpha"):
        sp.run_suite(random_bits(rng, 1000), alpha=0.0)


def test_results_are_json_serialisable(rng):
    suite = sp.run_suite(random_bits(rng, 1_000_000))
    decoded = json.loads(json.dumps(suite.to_dict()))
    assert decoded["n_bits"] == 1_000_000
    assert len(decoded["results"]) == 15
    assert decoded["results"][2]["labels"] == ["forward", "backward"]


def test_subtests_are_qualified_by_label(pi_100):
    assert sp.frequency(pi_100).subtests() == [("frequency", sp.frequency(pi_100).p_values[0])]
    names = [name for name, _ in sp.cumulative_sums(pi_100).subtests()]
    assert names == ["cumulative_sums[forward]", "cumulative_sums[backward]"]


def test_alpha_only_changes_the_verdict(pi_100):
    strict = sp.frequency(pi_100, alpha=0.2)
    lenient = sp.frequency(pi_100, alpha=0.01)
    assert strict.p_values == lenient.p_values
    assert lenient.passed
    assert not strict.passed


def test_tests_skip_when_undefined():
    empty = np.zeros(0, dtype=np.uint8)
    for name in sp.SUITE_TESTS:
        result = sp.run_suite(empty, tests=[name], enforce_min_length=False)[name]
        assert result.skipped, name
    few = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
    assert sp.random_excursions(few).skipped is not None
    assert "cycles" in str(sp.random_excursions_variant(few).skipped)
    assert sp.longest_run_of_ones(few).skipped == "needs n >= 128 bits, got n=12"


def test_invalid_parameters_raise():
    eps = np.zeros(2000, dtype=np.uint8)
    with pytest.raises(ValueError, match="block_size must be positive"):
        sp.block_frequency(eps, block_size=0)
    with pytest.raises(ValueError, match="block_length must be 2"):
        sp.serial(eps, block_length=1)
    with pytest.raises(ValueError, match=r"block_length must be 1\.\.16"):
        sp.universal(eps, block_length=17)
    with pytest.raises(ValueError, match="dimensions must be at least 2"):
        sp.binary_matrix_rank(eps, rows=1)
    with pytest.raises(ValueError, match="expected 7 class probabilities"):
        sp.linear_complexity(eps, probabilities=[1.0])
    with pytest.raises(ValueError, match="min_cycles must be positive"):
        sp.random_excursions(eps, min_cycles=0)
    with pytest.raises(ValueError, match="alpha"):
        sp.frequency(eps, alpha=1.5)
