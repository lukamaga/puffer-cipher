"""ChallengeSchedule: line code exactness, memory, determinism, thread safety."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from puffer.liveness import ChallengeSchedule, ExpiredSymbolError
from puffer.types import LedSchedule

from .helpers import seeded_schedule


class CountingSource:
    """Deterministic byte source that records how often it was called."""

    def __init__(self, seed: int = 7) -> None:
        self._gen = np.random.default_rng(seed)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, n: int) -> bytes:
        with self._lock:
            self.calls += 1
            return self._gen.bytes(n)


def fixed_bytes(data: bytes) -> Callable[[int], bytes]:
    """random_bytes that returns ``data`` repeated (bit-exact expectations)."""

    def source(n: int) -> bytes:
        return (data * (n // len(data) + 1))[:n]

    return source


def test_implements_led_schedule_protocol():
    assert isinstance(ChallengeSchedule(), LedSchedule)


def test_manchester_chips_exact():
    # 0xA5 = 1010 0101 -> symbols 0..7 carry bits 1,0,1,0,0,1,0,1 (MSB first).
    sch = ChallengeSchedule(0.2, random_bytes=fixed_bytes(b"\xa5"))
    assert sch.bits(0, 8).tolist() == [1, 0, 1, 0, 0, 1, 0, 1]
    for i, bit in enumerate(sch.bits(0, 8)):
        first = sch.state_at(i * 0.2 + 0.05)
        second = sch.state_at(i * 0.2 + 0.15)
        # bit 1 = on -> off, bit 0 = off -> on
        assert (first, second) == ((True, False) if bit else (False, True))
    assert sch.chip_states(0, 4).tolist() == [True, False, False, True]


def test_manchester_is_balanced_and_always_transitions_mid_symbol():
    sch = seeded_schedule(1)
    states = sch.chip_states(0, 2000)
    assert states.sum() == 1000  # exactly half the time on, whatever the bits
    assert np.all(states[0::2] != states[1::2])


def test_nrz_encoding():
    sch = ChallengeSchedule(0.1, "nrz", random_bytes=fixed_bytes(b"\xa5"))
    assert sch.chip_s == pytest.approx(0.1)
    assert sch.chips_per_symbol == 1
    assert [sch.state_at(0.05 + 0.1 * i) for i in range(8)] == [
        bool(b) for b in [1, 0, 1, 0, 0, 1, 0, 1]
    ]


def test_before_t0_is_off_and_has_no_bits():
    sch = ChallengeSchedule(0.2, t0=100.0, random_bytes=fixed_bytes(b"\xff"))
    assert not sch.state_at(99.99)
    assert not sch.state_at(-1e6)
    assert sch.state_at(100.0)  # bit 1 starts on
    assert sch.symbol_index(99.99) == -1
    assert sch.states_at([50.0, 99.0, 100.05]).tolist() == [False, False, True]
    with pytest.raises(ValueError, match="invalid symbol range"):
        sch.bits(-1, 2)


def test_index_helpers_and_chips_between():
    sch = ChallengeSchedule(0.2, t0=10.0, random_bytes=fixed_bytes(b"\xa5"))
    assert sch.encoding == "manchester"
    assert sch.t0 == 10.0
    assert sch.chip_s == pytest.approx(0.1)
    assert sch.chip_states(-5, -1).tolist() == [False] * 4  # before t0
    with pytest.raises(ValueError, match="invalid chip range"):
        sch.chip_states(3, 2)
    assert sch.symbol_index(10.0) == 0
    assert sch.symbol_index(10.39) == 1
    assert sch.chip_index(10.39) == 3
    assert sch.symbol_start(3) == pytest.approx(10.6)
    chips = sch.chips_between(10.05, 10.35)
    assert chips.chip.tolist() == [0, 1, 2, 3]
    assert np.allclose(chips.start, [10.0, 10.1, 10.2, 10.3])
    assert chips.state.tolist() == [True, False, False, True]  # bits 1, 0
    assert sch.chips_between(10.2, 10.2).chip.size == 0
    # consistent with state_at even where float rounding moves an edge
    for ta, tb in [(10.2, 10.25), (10.0, 10.2), (10.6, 10.6000001)]:
        chips = sch.chips_between(ta, tb)
        assert chips.chip[0] == sch.chip_index(ta)
        assert chips.chip[-1] == sch.chip_index(np.nextafter(tb, -np.inf))


def test_continuity_state_is_constant_within_each_chip(rng):
    sch = seeded_schedule(3)
    k = np.arange(0, 500)
    starts = k * sch.chip_s
    for offset in (1e-6, 0.25, 0.5, 0.75, 1 - 1e-6):
        assert np.array_equal(sch.states_at(starts + offset * sch.chip_s), sch.chip_states(0, 500))
    t = rng.uniform(0.0, 100.0, size=5000)
    t.sort()
    s = sch.states_at(t)
    changes = np.flatnonzero(s[1:] != s[:-1])
    # every state change happens across a chip boundary
    assert np.all(np.floor(t[changes] / sch.chip_s) != np.floor(t[changes + 1] / sch.chip_s))


def test_deterministic_for_a_given_random_source(rng):
    times = rng.uniform(0.0, 300.0, size=2000)
    a, b = seeded_schedule(42), seeded_schedule(42)
    assert np.array_equal(a.states_at(times), b.states_at(times))
    assert seeded_schedule(43).bits(0, 256).tolist() != a.bits(0, 256).tolist()


def test_bits_are_remembered(rng):
    sch = ChallengeSchedule(random_bytes=CountingSource())
    first = [sch.state_at(t) for t in np.linspace(0, 60, 601)]
    sch.states_at(rng.uniform(0, 120, size=1000))  # touch other batches in between
    assert [sch.state_at(t) for t in np.linspace(0, 60, 601)] == first
    assert sch.bit(17) == sch.bits(10, 20)[7]
    assert sch.bits(5, 5).size == 0


def test_vectorised_and_scalar_agree_exactly_on_boundaries():
    sch = seeded_schedule(5)
    t = np.arange(0, 3000) * (sch.chip_s / 3)  # many times land exactly on chip edges
    assert sch.states_at(t).tolist() == [sch.state_at(float(x)) for x in t]


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=-10.0, max_value=5000.0, allow_nan=False))
def test_property_states_at_matches_state_at(t):
    sch = seeded_schedule(11, t0=1.5)
    assert sch.states_at([t]).tolist() == [sch.state_at(t)]


def test_lazy_batched_generation():
    src = CountingSource()
    sch = ChallengeSchedule(0.2, random_bytes=src, batch_bits=64)
    assert src.calls == 0
    sch.state_at(0.0)
    sch.state_at(64 * 0.2 - 0.01)  # still batch 0
    assert src.calls == 1
    sch.state_at(64 * 0.2)  # first symbol of batch 1
    assert src.calls == 2


def test_retention_prunes_and_never_regenerates():
    src = CountingSource()
    sch = ChallengeSchedule(0.2, random_bytes=src, batch_bits=64, retention_s=30.0)
    early = sch.bits(0, 10).copy()
    sch.state_at(10_000.0)  # far ahead: old batches fall behind the horizon
    with pytest.raises(ExpiredSymbolError):
        sch.state_at(0.5)
    with pytest.raises(ExpiredSymbolError):
        sch.bits(0, 10)
    assert sch.held_symbols <= sch.retention_symbols + 2 * 64  # bounded memory
    assert early.size == 10
    calls = src.calls
    with pytest.raises(ValueError, match="retention"):
        sch.bits(0, sch.retention_symbols + 1)
    assert src.calls == calls


def test_memory_stays_bounded_while_time_advances():
    sch = ChallengeSchedule(0.2, random_bytes=CountingSource(), batch_bits=64, retention_s=20.0)
    for t in np.arange(0.0, 2000.0, 0.5):
        sch.state_at(float(t))
        sch.state_at(float(t) - 10.0)  # the verifier looks back within the horizon
    assert sch.held_symbols <= sch.retention_symbols + 2 * 64


def test_thread_safety_one_consistent_history():
    src = CountingSource()
    sch = ChallengeSchedule(0.2, random_bytes=src, batch_bits=32)
    times = np.random.default_rng(9).uniform(0.0, 200.0, size=4000)
    barrier = threading.Barrier(12)
    results: dict[int, list[bool]] = {}

    def worker(i: int) -> None:
        order = np.random.default_rng(i).permutation(times.size)
        barrier.wait()
        out = [False] * times.size
        for j in order:
            out[j] = sch.state_at(float(times[j]))
        results[i] = out

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    reference = results[0]
    assert all(r == reference for r in results.values())
    assert reference == sch.states_at(times).tolist()
    # every batch was drawn exactly once, despite the races
    n_batches = len({sch.symbol_index(float(t)) // 32 for t in times})
    assert src.calls == n_batches


def test_repr_does_not_leak_bits():
    sch = ChallengeSchedule(random_bytes=fixed_bytes(b"\x5a"))
    sch.bits(0, 64)
    text = repr(sch)
    assert "symbol_s=0.2" in text
    assert "0101" not in text
    assert "1010" not in text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symbol_s": 0.0}, "symbol_s"),
        ({"symbol_s": float("nan")}, "symbol_s"),
        ({"encoding": "ook"}, "encoding"),
        ({"t0": float("inf")}, "t0"),
        ({"batch_bits": 12}, "batch_bits"),
        ({"retention_s": -1.0}, "retention_s"),
    ],
)
def test_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ChallengeSchedule(**kwargs)


def test_short_random_source_is_rejected():
    sch = ChallengeSchedule(random_bytes=lambda n: b"\x00" * (n - 1))
    with pytest.raises(ValueError, match="returned"):
        sch.state_at(0.0)


def test_non_finite_times_are_rejected():
    sch = ChallengeSchedule()
    with pytest.raises(ValueError, match="finite"):
        sch.state_at(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        sch.states_at([0.0, float("inf")])


def test_default_source_is_fresh_os_randomness():
    a, b = ChallengeSchedule(), ChallengeSchedule()
    bits_a, bits_b = a.bits(0, 256), b.bits(0, 256)
    assert not np.array_equal(bits_a, bits_b)  # 2^-256 chance of a false failure
    assert 64 < int(bits_a.sum()) < 192
