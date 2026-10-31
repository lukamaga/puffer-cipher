"""Vectorised SP 800-90B machinery vs literal transcriptions of the standard.

The production estimators reorganise the standard's per-sample loops into
numpy operations (closed-form scoreboard, sort-based online counting,
suffix arrays). These tests keep a deliberately naive, line-by-line
transcription of each algorithm from the 2018 text (1-based indices exactly
as printed) and demand *identical* outputs on many seeded random inputs:
binary and larger alphabets, structured and unstructured, with small
parameters so every branch (Null predictions, ties, dictionary overflow) is
exercised quickly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from puffer.entropy import estimators as est

# --------------------------------------------------------------------------
# Literal transcriptions (slow, obviously correct)
# --------------------------------------------------------------------------


def ref_scoreboard(hits: list[list[bool]]) -> list[bool]:
    """6.3.7 step 3c-3d: predict with the winner, then update scores in order."""
    width = len(hits[0])
    scores = [0] * width
    winner = 0
    out = []
    for row in hits:
        out.append(row[winner])
        for d in range(width):
            if row[d]:
                scores[d] += 1
                if scores[d] >= scores[winner]:
                    winner = d
    return out


def ref_multi_mcw(s: list[int], windows: tuple[int, ...]) -> list[bool]:
    """6.3.7 steps 1-3."""
    length = len(s)
    scoreboard = [0] * len(windows)
    winner = 0
    correct = []
    for i in range(windows[0] + 1, length + 1):
        frequent: list[int | None] = []
        for w in windows:
            if i > w:
                window = s[i - w - 1 : i - 1]  # s_{i-w} .. s_{i-1}
                counts = Counter(window)
                top = max(counts.values())
                frequent.append(next(x for x in reversed(window) if counts[x] == top))
            else:
                frequent.append(None)
        correct.append(frequent[winner] == s[i - 1])
        for j, f in enumerate(frequent):
            if f == s[i - 1]:
                scoreboard[j] += 1
                if scoreboard[j] >= scoreboard[winner]:
                    winner = j
    return correct


def ref_lag(s: list[int], depth: int) -> list[bool]:
    """6.3.8 steps 1-3."""
    scoreboard = [0] * depth
    winner = 0
    correct = []
    for i in range(2, len(s) + 1):
        lags = [s[i - d - 1] if d < i else None for d in range(1, depth + 1)]  # s_{i-d}
        correct.append(lags[winner] == s[i - 1])
        for d, guess in enumerate(lags):
            if guess == s[i - 1]:
                scoreboard[d] += 1
                if scoreboard[d] >= scoreboard[winner]:
                    winner = d
    return correct


def _most_frequent(counts: dict[int, int]) -> int:
    """Highest count; ties to the largest symbol (NIST reference tie break)."""
    return max(counts, key=lambda y: (counts[y], y))


def ref_multi_mmc(s: list[int], depth: int) -> list[bool]:
    """6.3.9 steps 1-4."""
    models: list[dict[tuple[int, ...], dict[int, int]]] = [{} for _ in range(depth)]
    scoreboard = [0] * depth
    winner = 0
    correct = []
    for i in range(3, len(s) + 1):
        for d in range(1, depth + 1):
            if d < i - 1:
                ctx = tuple(s[i - d - 2 : i - 2])  # s_{i-d-1} .. s_{i-2}
                successors = models[d - 1].setdefault(ctx, {})
                successors[s[i - 2]] = successors.get(s[i - 2], 0) + 1  # s_{i-1}
        subpredict: list[int | None] = []
        for d in range(1, depth + 1):
            ctx = tuple(s[i - d - 1 : i - 1])  # s_{i-d} .. s_{i-1}
            seen = models[d - 1].get(ctx) if i - d - 1 >= 0 else None
            subpredict.append(_most_frequent(seen) if seen else None)
        correct.append(subpredict[winner] == s[i - 1])
        for d, guess in enumerate(subpredict):
            if guess == s[i - 1]:
                scoreboard[d] += 1
                if scoreboard[d] >= scoreboard[winner]:
                    winner = d
    return correct


def ref_lz78y(s: list[int], depth: int, max_dictionary: int) -> list[bool]:
    """6.3.10 step 3."""
    dictionary: dict[tuple[int, ...], dict[int, int]] = {}
    correct = []
    for i in range(depth + 2, len(s) + 1):
        for j in range(depth, 0, -1):
            ctx = tuple(s[i - j - 2 : i - 2])  # s_{i-j-1} .. s_{i-2}
            if ctx not in dictionary and len(dictionary) < max_dictionary:
                dictionary[ctx] = {s[i - 2]: 0}
            if ctx in dictionary:
                dictionary[ctx][s[i - 2]] = dictionary[ctx].get(s[i - 2], 0) + 1
        prediction, max_count = None, 0
        for j in range(depth, 0, -1):
            prev = tuple(s[i - j - 1 : i - 1])  # s_{i-j} .. s_{i-1}
            if prev in dictionary:
                y = _most_frequent(dictionary[prev])
                if dictionary[prev][y] > max_count:
                    prediction, max_count = y, dictionary[prev][y]
        correct.append(prediction == s[i - 1])
    return correct


def ref_collision_times(s: list[int]) -> list[int]:
    """6.3.2 steps 1-5, for any alphabet."""
    times: list[int] = []
    index = 0
    while True:
        seen: set[int] = set()
        j = index
        while j < len(s) and s[j] not in seen:
            seen.add(s[j])
            j += 1
        if j >= len(s):
            return times
        times.append(j - index + 1)
        index = j + 1


def ref_compression_distances(bits: list[int], b: int, d: int) -> list[int]:
    """6.3.4 steps 1-4."""
    n_blocks = len(bits) // b
    blocks = [int("".join(map(str, bits[i * b : (i + 1) * b])), 2) for i in range(n_blocks)]
    dictionary = [0] * (1 << b)
    for i in range(1, d + 1):
        dictionary[blocks[i - 1]] = i
    out = []
    for i in range(d + 1, n_blocks + 1):
        x = blocks[i - 1]
        out.append(i - dictionary[x] if dictionary[x] else i)
        dictionary[x] = i
    return out


def ref_tuple_profile(s: list[int]) -> tuple[list[int], list[int]]:
    """Q[W] and sum C(c, 2) for W = 0..v by explicit tuple counting."""
    q, pairs = [0], [0]
    width = 1
    while True:
        counts = Counter(tuple(s[i : i + width]) for i in range(len(s) - width + 1))
        if max(counts.values(), default=0) < 2:
            return q, pairs
        q.append(max(counts.values()))
        pairs.append(sum(c * (c - 1) // 2 for c in counts.values()))
        width += 1


def ref_online_mode(groups: list[int], outcomes: list[int]) -> tuple[list[int], list[int]]:
    seen: dict[int, dict[int, int]] = {}
    predictions, counts = [], []
    for g, y in zip(groups, outcomes, strict=True):
        successors = seen.setdefault(g, {})
        if successors:
            best = _most_frequent(successors)
            predictions.append(best)
            counts.append(successors[best])
        else:
            predictions.append(-1)
            counts.append(0)
        successors[y] = successors.get(y, 0) + 1
    return predictions, counts


# --------------------------------------------------------------------------
# Inputs: unstructured and structured sequences over several alphabets
# --------------------------------------------------------------------------


def _sequences(rng: np.random.Generator, length: int) -> list[np.ndarray]:
    flips = rng.random(length) < 0.15
    return [
        rng.integers(0, 2, length),
        (rng.random(length) < 0.8).astype(np.int64),
        np.cumsum(flips) % 2,  # sticky binary Markov chain
        rng.integers(0, 3, length),
        rng.choice(5, length, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
        np.tile(rng.integers(0, 4, 7), length // 7 + 1)[:length],  # periodic
        rng.integers(0, 40, length),
    ]


def _dense(x: np.ndarray) -> tuple[np.ndarray, int]:
    return est._dense(x)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_follow_the_leader_matches_sequential_scoreboard(rng: np.random.Generator) -> None:
    for width in (1, 2, 3, 8):
        for density in (0.05, 0.5, 0.9):
            hits = rng.random((700, width)) < density
            expected = ref_scoreboard(hits.tolist())
            # Uneven chunks exercise the carried scores and winner.
            cuts = np.sort(rng.choice(np.arange(1, 700), size=5, replace=False))
            chunks = np.split(hits, cuts)
            assert est._follow_the_leader(iter(chunks)).tolist() == expected


def test_online_mode_matches_dictionary_counting(rng: np.random.Generator) -> None:
    for k in (2, 3, 9):
        groups = rng.integers(0, 6, 800)
        outcomes = rng.integers(0, k, 800)
        prediction, count, first = est._online_mode(groups, outcomes, k)
        ref_prediction, ref_count = ref_online_mode(groups.tolist(), outcomes.tolist())
        assert count.tolist() == ref_count
        assert prediction.tolist() == ref_prediction
        first_seen = {g: i for i, g in reversed(list(enumerate(groups.tolist())))}
        assert first.tolist() == [first_seen[g] for g in groups.tolist()]


@pytest.mark.parametrize("windows", [(3, 7, 15, 31), (4, 6, 10), (63, 255)])
def test_multi_mcw_matches_reference(rng: np.random.Generator, windows: tuple[int, ...]) -> None:
    for x in _sequences(rng, 600):
        s, k = _dense(x)
        expected = ref_multi_mcw(s.tolist(), windows)
        assert est._multi_mcw_correct(s, k, windows).tolist() == expected


def test_multi_mcw_standard_windows_matches_reference(rng: np.random.Generator) -> None:
    s, k = _dense(rng.choice(4, 4400, p=[0.4, 0.3, 0.2, 0.1]))
    expected = ref_multi_mcw(s.tolist(), est.MULTI_MCW_WINDOWS)
    assert est._multi_mcw_correct(s, k, est.MULTI_MCW_WINDOWS).tolist() == expected


@pytest.mark.parametrize("w", [1, 2, 5, 6, 63])
def test_window_mode_paths_agree_on_bits(rng: np.random.Generator, w: int) -> None:
    for x in _sequences(rng, 2000)[:3]:
        s = x.astype(np.int64)
        assert np.array_equal(est._window_modes_binary(s, w), est._window_modes_general(s, 2, w))


@pytest.mark.parametrize("depth", [1, 3, 9])
def test_lag_matches_reference(rng: np.random.Generator, depth: int) -> None:
    for x in _sequences(rng, 500):
        s, _ = _dense(x)
        assert est._lag_correct(s, depth).tolist() == ref_lag(s.tolist(), depth)


def test_lag_standard_depth_streams_chunks(rng: np.random.Generator) -> None:
    s, _ = _dense(rng.integers(0, 3, 9000))  # > 2 chunks of 4096 rows
    assert est._lag_correct(s).tolist() == ref_lag(s.tolist(), est.LAG_DEPTH)


@pytest.mark.parametrize("depth", [1, 2, 5, 16])
def test_multi_mmc_matches_reference(rng: np.random.Generator, depth: int) -> None:
    for x in _sequences(rng, 400):
        s, k = _dense(x)
        assert est._multi_mmc_correct(s, k, depth).tolist() == ref_multi_mmc(s.tolist(), depth)


@pytest.mark.parametrize(("depth", "cap"), [(1, 65_536), (4, 65_536), (4, 25), (5, 3), (16, 400)])
def test_lz78y_matches_reference(rng: np.random.Generator, depth: int, cap: int) -> None:
    for x in _sequences(rng, 400):
        s, k = _dense(x)
        expected = ref_lz78y(s.tolist(), depth, cap)
        assert est._lz78y_correct(s, k, depth, cap).tolist() == expected


def test_lz78y_dictionary_fills_in_the_middle_of_a_step(rng: np.random.Generator) -> None:
    """Caps chosen so the dictionary fills part-way through one step's new contexts."""
    s, k = _dense(rng.integers(0, 3, 300))
    for cap in range(1, 60, 3):
        assert est._lz78y_correct(s, k, 6, cap).tolist() == ref_lz78y(s.tolist(), 6, cap)


def test_collision_times_match_reference(rng: np.random.Generator) -> None:
    for x in _sequences(rng, 3001)[:3]:
        s = x.astype(np.int64)
        for cut in (2, 3, 4, 5, 3001):
            prefix = s[:cut]
            assert est._collision_times(prefix).tolist() == ref_collision_times(prefix.tolist())


def test_compression_distances_match_reference(rng: np.random.Generator) -> None:
    for x in _sequences(rng, 6 * 1400)[:3]:
        bits = x.astype(np.int64)
        expected = ref_compression_distances(bits.tolist(), 6, 1000)
        assert est._compression_distances(bits, 6, 1000).tolist() == expected


def test_suffix_array_and_tuple_profile_match_brute_force(rng: np.random.Generator) -> None:
    for x in [*_sequences(rng, 300), np.zeros(50, dtype=np.int64), np.arange(30)]:
        s, _ = _dense(x)
        values = s.tolist()
        assert est._suffix_array(s).tolist() == sorted(range(len(values)), key=lambda i: values[i:])
        q, pairs = ref_tuple_profile(values)
        profile = est._tuple_profile(s)
        assert profile.max_count[1:].tolist() == q[1:]
        assert profile.pair_count[1:].tolist() == pairs[1:]
        assert profile.lrs_length == len(q) - 1
