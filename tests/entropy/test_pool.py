"""Entropy pool: accumulators, credit ledger, attribution, ratchet, thread safety."""

from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest

from puffer.entropy.pool import EntropyPool, pool_person
from puffer.types import ChannelOutput


def output(
    channel: str,
    absorb: bytes = b"frame",
    n_samples: int = 0,
    owners: tuple[str | None, ...] | None = None,
    frame_index: int = 0,
) -> ChannelOutput:
    return ChannelOutput(
        channel=channel,
        frame_index=frame_index,
        t=frame_index / 30,
        absorb=absorb,
        samples=np.zeros(n_samples, dtype=np.uint8),
        sample_owner=owners,
    )


def accumulate(channel: str, absorbs: list[bytes], seed: bytes | None = None) -> bytes:
    """Independent re-implementation of one accumulator (DESIGN.md section 4.4)."""
    h = hashlib.blake2b(digest_size=64, person=pool_person(channel))
    if seed is not None:
        h.update(seed)
    for data in absorbs:
        h.update(len(data).to_bytes(8, "big") + data)
    return h.digest()


def test_personalization_is_truncated_to_16_bytes():
    assert pool_person("sensor") == b"puffer-pool-sens"
    assert pool_person("fish") == b"puffer-pool-fish"


def test_credit_saturates_per_channel():
    pool = EntropyPool(capacity_bits=1000)
    for _ in range(3):
        pool.absorb(output("sensor"), 400.0)
    pool.absorb(output("fish"), 10.0)
    assert pool.credits() == {"sensor": 1000.0, "fish": 10.0}
    assert pool.camera_credit() == 1010.0


def test_fish_attribution_is_proportional_and_sums_to_channel_credit():
    pool = EntropyPool()
    pool.absorb(output("fish", n_samples=4, owners=("a", "b", "a", "a")), 8.0)
    pool.absorb(output("fish", n_samples=2, owners=("c", "b")), 2.0)
    fish = pool.fish_credits()
    assert fish == pytest.approx({"a": 6.0, "b": 3.0, "c": 1.0})
    assert sum(fish.values()) == pytest.approx(pool.credits()["fish"])


def test_attribution_uses_credit_actually_added_after_saturation():
    pool = EntropyPool(capacity_bits=10)
    pool.absorb(output("fish", n_samples=1, owners=("a",)), 8.0)
    pool.absorb(output("fish", n_samples=2, owners=("b", "c")), 8.0)  # only 2 bits fit
    assert pool.fish_credits() == pytest.approx({"a": 8.0, "b": 1.0, "c": 1.0})
    assert sum(pool.fish_credits().values()) == pytest.approx(pool.credits()["fish"])


def test_unknown_owners_stay_unattributed():
    pool = EntropyPool()
    pool.absorb(output("fish", n_samples=4, owners=("a", None, None, None)), 4.0)
    assert pool.fish_credits() == {"a": 1.0}
    assert pool.credits()["fish"] == 4.0


def test_draw_material_matches_reference_and_resets_credit():
    pool = EntropyPool()
    pool.absorb(output("sensor", b"s1", frame_index=10), 100.0)
    pool.absorb(output("fish", b"f1", n_samples=1, owners=("nemo",), frame_index=11), 5.0)
    pool.absorb(output("sensor", b"s2", frame_index=12), 50.0)
    draw = pool.draw()
    assert draw.draw_index == 0
    assert draw.channels == ("fish", "sensor")  # sorted
    assert draw.material == accumulate("fish", [b"f1"]) + accumulate("sensor", [b"s1", b"s2"])
    assert draw.credits == {"sensor": 150.0, "fish": 5.0}
    assert draw.camera_bits == 155.0
    assert draw.fish_credits == {"nemo": 5.0}
    assert draw.frames == (10, 12)

    assert pool.credits() == {"sensor": 0.0, "fish": 0.0}
    assert pool.fish_credits() == {}
    assert pool.draw_index == 1
    second = pool.draw()
    assert second.draw_index == 1
    assert second.frames is None
    assert second.credits == {"sensor": 0.0, "fish": 0.0}


def test_ratchet_changes_material_and_is_one_way():
    pool = EntropyPool()
    pool.absorb(output("sensor", b"only once"), 1.0)
    first = pool.draw().material
    second = pool.draw().material
    assert first != second
    ratchet = hashlib.blake2b(first, digest_size=64, person=b"puffer-ratchet").digest()
    assert second == accumulate("sensor", [], seed=ratchet)
    pool.absorb(output("sensor", b"more"), 1.0)
    third = pool.draw().material
    ratchet = hashlib.blake2b(second, digest_size=64, person=b"puffer-ratchet").digest()
    assert third == accumulate("sensor", [b"more"], seed=ratchet)


def test_empty_pool_draws_empty_material():
    draw = EntropyPool().draw()
    assert (draw.material, draw.channels, draw.credits, draw.frames) == (b"", (), {}, None)


def test_framing_makes_absorb_boundaries_matter():
    a, b = EntropyPool(), EntropyPool()
    a.absorb(output("sensor", b"ab"), 0.0)
    a.absorb(output("sensor", b"c"), 0.0)
    b.absorb(output("sensor", b"a"), 0.0)
    b.absorb(output("sensor", b"bc"), 0.0)
    assert a.draw().material != b.draw().material


def test_snapshot_digest_consumes_nothing_and_tracks_state():
    pool = EntropyPool()
    empty = pool.snapshot_digest()
    pool.absorb(output("sensor", b"x"), 10.0)
    snap = pool.snapshot_digest()
    assert len(snap) == 64
    assert snap != empty
    assert pool.snapshot_digest() == snap
    assert pool.credits() == {"sensor": 10.0}
    draw = pool.draw()
    assert snap not in draw.material
    assert pool.snapshot_digest() != snap


def test_invalid_inputs():
    pool = EntropyPool()
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="credit_bits"):
            pool.absorb(output("sensor"), bad)
    with pytest.raises(ValueError, match="one entry per sample"):
        pool.absorb(output("fish", n_samples=2, owners=("a",)), 1.0)
    with pytest.raises(ValueError, match="capacity_bits"):
        EntropyPool(capacity_bits=0)


def test_concurrent_absorb_and_draw_conserve_credit():
    pool = EntropyPool(capacity_bits=1e12)
    n_threads, n_absorbs = 8, 500
    drawn: list[float] = []
    stop = threading.Event()

    def producer(k: int) -> None:
        for i in range(n_absorbs):
            pool.absorb(output("sensor" if k % 2 else "fish", bytes([k, i % 256])), 1.0)

    def consumer() -> None:
        while not stop.is_set():
            drawn.append(pool.draw().camera_bits)

    threads = [threading.Thread(target=producer, args=(k,)) for k in range(n_threads)]
    drawer = threading.Thread(target=consumer)
    drawer.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    drawer.join()
    assert sum(drawn) + pool.camera_credit() == n_threads * n_absorbs


def test_wait_for_credit_wakes_on_absorb():
    pool = EntropyPool()
    assert not pool.wait_for_credit(1.0, timeout=0.0)
    timer = threading.Timer(0.05, lambda: pool.absorb(output("sensor"), 64.0))
    timer.start()
    try:
        assert pool.wait_for_credit(64.0, timeout=5.0)
    finally:
        timer.cancel()
    assert not pool.wait_for_credit(65.0, timeout=0.01)
    assert pool.wait_for_credit(64.0, timeout=0.0, channels=["sensor"])
    assert not pool.wait_for_credit(1.0, timeout=0.0, channels=["fish"])
