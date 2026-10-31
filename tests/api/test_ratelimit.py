"""Token buckets (puffer.api.ratelimit) on a manual clock."""

from __future__ import annotations

import pytest

from puffer.api.ratelimit import TokenBucketLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_burst_then_sustained_rate() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(rate=2.0, burst=3, clock=clock)
    assert [limiter.acquire("a") for _ in range(3)] == [None, None, None]
    denied = limiter.acquire("a")
    assert denied is not None
    assert denied.retry_after_s == pytest.approx(0.5)
    assert denied.retry_after_header == "1"
    clock.now += 0.5
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is not None


def test_buckets_are_per_client_and_refill_caps_at_burst() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(rate=1.0, burst=2, clock=clock)
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is not None
    assert limiter.acquire("b") is None  # another client is unaffected
    clock.now += 1000.0
    assert [limiter.acquire("a") is None for _ in range(3)] == [True, True, False]


def test_denied_requests_do_not_drain_the_bucket_further() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(rate=1.0, burst=1, clock=clock)
    assert limiter.acquire("a") is None
    for _ in range(100):
        assert limiter.acquire("a") is not None
    clock.now += 1.0
    assert limiter.acquire("a") is None


def test_memory_is_bounded() -> None:
    limiter = TokenBucketLimiter(rate=1.0, burst=1, max_clients=10, clock=Clock())
    for i in range(1000):
        limiter.acquire(f"10.0.{i // 256}.{i % 256}")
    assert len(limiter) == 10


def test_large_wait_header_rounds_up() -> None:
    limiter = TokenBucketLimiter(rate=0.1, burst=1, clock=Clock())
    limiter.acquire("a")
    denied = limiter.acquire("a")
    assert denied is not None
    assert denied.retry_after_header == "10"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate": 0, "burst": 1}, "rate"),
        ({"rate": 1, "burst": 0.5}, "burst"),
        ({"rate": float("nan"), "burst": 1}, "rate"),
        ({"rate": 1, "burst": 1, "max_clients": 0}, "max_clients"),
    ],
)
def test_invalid_parameters(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TokenBucketLimiter(**kwargs)
