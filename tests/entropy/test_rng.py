"""PufferRNG: exact construction, independence from the pool, display bits, waiting."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable

import nacl.utils
import numpy as np
import pytest
from scipy.stats import chisquare

from puffer.entropy.pool import EntropyPool
from puffer.entropy.receipt import ReceiptContext
from puffer.entropy.rng import CHUNK_BYTES, PufferRNG, derive_seed
from puffer.types import ChannelOutput, HealthState, LivenessState, LivenessStatus


def sensor_output(absorb: bytes, index: int = 0) -> ChannelOutput:
    return ChannelOutput("sensor", index, index / 30, absorb, np.zeros(0, np.uint8))


def filled_pool(*absorbs: bytes, credit: float = 100.0) -> EntropyPool:
    pool = EntropyPool()
    for i, data in enumerate(absorbs):
        pool.absorb(sensor_output(data, i), credit)
    return pool


def fixed(key: bytes) -> Callable[[int], bytes]:
    def system_random(n: int) -> bytes:
        assert n == 32
        return key

    return system_random


KEY_A = bytes(range(32))
KEY_B = bytes(range(1, 33))


def test_output_matches_the_specified_construction():
    pool = filled_pool(b"frame-1", b"frame-2")
    expected_material = filled_pool(b"frame-1", b"frame-2").draw().material
    rng = PufferRNG(pool, system_random=fixed(KEY_A))
    out, _ = rng.generate(100, "salt")
    seed = derive_seed(KEY_A, b"salt", 0, expected_material)
    assert out == nacl.utils.randombytes_deterministic(100, seed)


def test_derive_seed_layout():
    import hashlib

    h = hashlib.blake2b(digest_size=32, key=KEY_A, person=b"puffer-seed-v1")
    h.update((3).to_bytes(4, "big") + b"key" + (7).to_bytes(8, "big") + b"material")
    assert derive_seed(KEY_A, b"key", 7, b"material") == h.digest()
    with pytest.raises(ValueError, match="32 bytes"):
        derive_seed(b"short", b"key", 0, b"")


def test_deterministic_with_injected_system_random():
    a = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_A))
    b = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_A))
    assert [a.random_bytes(48) for _ in range(3)] == [b.random_bytes(48) for _ in range(3)]


def test_system_randomness_alone_changes_output_for_an_identical_pool():
    a = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_A))
    b = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_B))
    assert a.random_bytes(32) != b.random_bytes(32)


def test_pool_alone_changes_output_for_identical_system_bytes():
    """If the OS RNG were stuck, the fish would still make outputs differ."""
    a = PufferRNG(filled_pool(b"nemo left"), system_random=fixed(KEY_A))
    b = PufferRNG(filled_pool(b"nemo right"), system_random=fixed(KEY_A))
    assert a.random_bytes(32) != b.random_bytes(32)


def test_repeated_system_bytes_never_repeat_output():
    rng = PufferRNG(EntropyPool(), system_random=fixed(KEY_A))
    outputs = {rng.random_bytes(32) for _ in range(50)}
    assert len(outputs) == 50  # counter + ratchet keep every seed unique


def test_purpose_is_domain_separated():
    a = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_A))
    b = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_A))
    assert a.generate(32, "key")[0] != b.generate(32, "nonce")[0]


def test_constant_pool_output_is_uniform_looking(rng):
    """No camera entropy at all: output quality comes from the system key alone."""
    pool = EntropyPool()
    generator = PufferRNG(pool, system_random=lambda n: rng.bytes(n))
    data = b"".join(generator.random_bytes(4096) for _ in range(64))
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    assert chisquare(counts).pvalue > 1e-4
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    assert abs(bits.mean() - 0.5) < 5 * 0.5 / np.sqrt(bits.size)


def test_large_requests_are_chunked_with_consecutive_counters():
    pool = filled_pool(b"tank")
    material = filled_pool(b"tank").draw().material
    n = 2 * CHUNK_BYTES + 5
    out, receipt = PufferRNG(pool, system_random=fixed(KEY_A)).generate(n, "file")
    assert len(out) == n == receipt.n_bytes
    expected = b"".join(
        nacl.utils.randombytes_deterministic(size, derive_seed(KEY_A, b"file", i, material))
        for i, size in enumerate((CHUNK_BYTES, CHUNK_BYTES, 5))
    )
    assert out == expected


def test_convenience_methods_and_validation():
    rng = PufferRNG(filled_pool(b"tank"))
    key, receipt = rng.generate_key()
    assert len(key) == 32
    assert receipt.purpose == "key"
    assert len(rng.nonce()) == 24
    assert len(rng.random_bytes(7)) == 7
    assert rng.random_bytes(0) == b""
    with pytest.raises(ValueError, match="non-negative"):
        rng.generate(-1)
    with pytest.raises(TypeError):
        rng.generate(1.5)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="wrong number of bytes"):
        PufferRNG(EntropyPool(), system_random=lambda n: b"\x00" * 16).generate(1)


def test_generate_consumes_credit_and_fills_the_receipt():
    pool = filled_pool(b"a", b"b", credit=300.0)
    context = ReceiptContext(
        mode="night",
        liveness=LivenessStatus(LivenessState.VERIFIED, confidence_bits=40.5),
        health={"sensor": HealthState.OK, "fish": HealthState.STARTUP},
    )
    rng = PufferRNG(pool, clock=lambda: 1_793_454_000.123, context=lambda: context)
    _, receipt = rng.generate(16, "beacon")
    assert receipt.timestamp_ms == 1_793_454_000_123
    assert receipt.channels["sensor"].credited_bits == 600.0
    assert receipt.mode == "night"
    assert receipt.liveness_state is LivenessState.VERIFIED
    assert receipt.frames == (0, 1)
    assert pool.camera_credit() == 0.0
    assert rng.generate(16)[1].camera_bits == 0.0


def test_display_bits_are_public_and_consume_no_credit():
    pool = filled_pool(b"tank", credit=50.0)
    rng = PufferRNG(pool)
    bits = rng.display_bits(256)
    assert len(bits) == 256
    assert set(bits) <= {"0", "1"}
    assert rng.display_bits(256) != bits  # counter advances
    assert pool.camera_credit() == 50.0
    assert len(rng.display_bits(1000)) == 1000
    assert rng.display_bits(0) == ""
    # Reproducible from the public snapshot: no OS randomness involved.
    twin = PufferRNG(filled_pool(b"tank"), system_random=fixed(KEY_A))
    assert twin.display_bits(256) == bits


def test_wait_for_credit_times_out_and_wakes_up():
    pool = EntropyPool()
    rng = PufferRNG(pool)
    assert not rng.wait_for_credit(10.0, timeout=0.05, poll_s=0.01)
    timer = threading.Timer(0.05, lambda: pool.absorb(sensor_output(b"x"), 10.0))
    timer.start()
    try:
        assert rng.wait_for_credit(10.0, timeout=5.0)
    finally:
        timer.cancel()


def test_wait_for_credit_pumps_a_single_threaded_engine():
    pool = EntropyPool()
    frames = itertools.count()
    rng = PufferRNG(pool, pump=lambda: pool.absorb(sensor_output(b"f", next(frames)), 4.0))
    assert rng.wait_for_credit(20.0, timeout=None, poll_s=0.001)
    assert next(frames) == 5  # pumped exactly until 5 * 4 bits were credited
    with pytest.raises(ValueError, match="poll_s"):
        rng.wait_for_credit(1.0, timeout=0.0, poll_s=0.0)


# -- public values: keyed by the OS, mixed with the pool, no credit consumed ---


def public_seed(key: bytes, purpose: bytes, counter: int, snapshot: bytes) -> bytes:
    """DESIGN.md 4.4 public variant, written out with hashlib only."""
    import hashlib

    h = hashlib.blake2b(digest_size=32, key=key, person=b"puffer-public-v1")
    h.update(len(purpose).to_bytes(4, "big") + purpose + counter.to_bytes(8, "big") + snapshot)
    return h.digest()


def test_public_output_matches_the_specified_construction():
    pool = filled_pool(b"frame-1", b"frame-2")
    snapshot = pool.snapshot_digest()
    rng = PufferRNG(pool, system_random=fixed(KEY_A))
    out, _ = rng.generate(100, "salt", consume=False)
    assert out == nacl.utils.randombytes_deterministic(
        100, public_seed(KEY_A, b"salt", 0, snapshot)
    )
    assert derive_seed(KEY_A, b"salt", 0, snapshot, person=b"puffer-public-v1") == public_seed(
        KEY_A, b"salt", 0, snapshot
    )
    # The counter is shared with the consuming path, chunk by chunk.
    n = CHUNK_BYTES + 3
    big, _ = rng.generate(n, "x", consume=False)
    assert big == b"".join(
        nacl.utils.randombytes_deterministic(size, public_seed(KEY_A, b"x", i, snapshot))
        for i, size in ((1, CHUNK_BYTES), (2, 3))
    )


def test_public_output_consumes_no_credit_and_its_receipt_says_so():
    pool = filled_pool(b"a", b"b", credit=300.0)
    context = ReceiptContext(mode="day", health={"sensor": HealthState.OK})
    rng = PufferRNG(pool, clock=lambda: 1_793_454_000.0, context=lambda: context)
    draws_before = pool.draw_index
    out, receipt = rng.generate(24, "nonce", consume=False)
    assert len(out) == 24
    assert pool.credits() == {"sensor": 600.0}
    assert pool.draw_index == draws_before  # nothing was drawn or ratcheted
    d = receipt.to_dict()
    assert d["purpose"] == "nonce"
    assert d["frames"] == []
    assert d["channels"]["sensor"] == {"credited_bits": 0.0, "share": None, "health": "ok"}
    assert d["channels"]["fish"]["credited_bits"] == 0.0
    assert d["channels"]["system"] == {"credited_bits": 256.0, "share": None, "health": "ok"}
    assert d["fish"] == []
    assert receipt.camera_bits == 0.0


def test_nonce_and_salt_leave_camera_credit_for_the_next_key():
    pool = filled_pool(b"tank", credit=500.0)
    rng = PufferRNG(pool)
    assert len(rng.nonce()) == 24
    assert len(rng.salt()) == 16
    assert len(rng.salt(32)) == 32
    _, receipt = rng.generate_key()
    assert receipt.camera_bits == 500.0


def test_public_output_is_keyed_by_the_os_and_mixes_the_pool():
    def nonce(pool: EntropyPool, key: bytes) -> bytes:
        return PufferRNG(pool, system_random=fixed(key)).nonce()

    assert nonce(filled_pool(b"tank"), KEY_A) == nonce(filled_pool(b"tank"), KEY_A)
    assert nonce(filled_pool(b"tank"), KEY_A) != nonce(filled_pool(b"tank"), KEY_B)
    assert nonce(filled_pool(b"nemo left"), KEY_A) != nonce(filled_pool(b"nemo right"), KEY_A)
    calls: list[int] = []

    def counting(n: int) -> bytes:
        calls.append(n)
        return KEY_A

    rng = PufferRNG(filled_pool(b"tank"), system_random=counting)
    rng.nonce()
    rng.salt()
    assert calls == [32, 32]  # one fresh OS key per public value


def test_public_and_secret_outputs_never_share_a_seed():
    """Same OS key, same pool, interleaved calls: every output is distinct."""
    rng = PufferRNG(EntropyPool(), system_random=fixed(KEY_A))
    outputs = []
    for _ in range(20):
        outputs += [rng.nonce()[:16], rng.salt(), rng.random_bytes(16), rng.salt()]
    assert len(set(outputs)) == len(outputs)
    # Personalisation separates the domains even for identical inputs.
    assert derive_seed(KEY_A, b"n", 0, b"m") != derive_seed(
        KEY_A, b"n", 0, b"m", person=b"puffer-public-v1"
    )


def test_salt_plugs_into_password_encryption():
    import io

    from nacl.pwhash import argon2id

    from puffer.crypto import encrypt_stream

    pool = filled_pool(b"tank", credit=100.0)
    header = encrypt_stream(
        io.BytesIO(b"secret"),
        io.BytesIO(),
        password="pw",
        opslimit=argon2id.OPSLIMIT_MIN,
        memlimit=argon2id.MEMLIMIT_MIN,
        random_bytes=PufferRNG(pool).salt,
    )
    assert header.argon2 is not None
    assert len(header.argon2.salt) == 16
    assert pool.camera_credit() == 100.0


def test_failing_os_randomness_consumes_no_credit():
    pool = filled_pool(b"tank", credit=100.0)

    def broken(n: int) -> bytes:
        raise OSError("getrandom() failed")

    with pytest.raises(OSError, match="getrandom"):
        PufferRNG(pool, system_random=broken).generate_key()
    with pytest.raises(RuntimeError, match="wrong number of bytes"):
        PufferRNG(pool, system_random=lambda n: b"\x00" * 16).generate_key()
    assert pool.camera_credit() == 100.0
    assert pool.draw_index == 0


def test_concurrent_requests_never_share_a_counter():
    """Stuck OS key + empty pool: only the counter separates seeds, across threads too."""
    rng = PufferRNG(EntropyPool(), system_random=fixed(KEY_A))
    results: list[bytes] = []
    lock = threading.Lock()

    def worker() -> None:
        mine = [rng.random_bytes(16) for _ in range(50)] + [rng.salt() for _ in range(50)]
        with lock:
            results.extend(mine)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == len(results) == 800


def test_display_bits_never_touch_os_randomness_or_credit():
    def forbidden(n: int) -> bytes:
        raise AssertionError("display bits must not use the system key")

    pool = filled_pool(b"tank", credit=50.0)
    rng = PufferRNG(pool, system_random=forbidden)
    assert len(rng.display_bits(512)) == 512
    assert (pool.camera_credit(), pool.draw_index) == (50.0, 0)


def test_nonce_fits_the_random_bytes_hook_of_aead():
    from puffer.crypto import NONCE_BYTES, decrypt_message, encrypt_message

    pool = filled_pool(b"tank", credit=50.0)
    rng = PufferRNG(pool)
    key = bytes(range(32))
    blob = encrypt_message(b"fish", key, random_bytes=rng.nonce)
    assert len(rng.nonce()) == NONCE_BYTES
    assert decrypt_message(blob, key) == b"fish"
    assert pool.camera_credit() == 50.0


def test_wait_for_unreachable_credit_fails_fast():
    rng = PufferRNG(EntropyPool(capacity_bits=10))
    with pytest.raises(ValueError, match="never"):
        rng.wait_for_credit(21.0, timeout=None)
