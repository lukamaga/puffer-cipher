"""Per-client token buckets for the endpoints that spend entropy or CPU.

Why ``/api/random`` and ``/api/demo/*`` are limited: every call draws the
pool (``PufferRNG.generate`` consumes *all* camera credit), so a client
hammering ``/random`` would leave every other consumer - the beacon, a
``puffer keygen`` - with keys that carry no fish entropy at all. Output is
never weaker than ``os.urandom`` either way; what is protected is the
*credit*, and the server's CPU.

A token bucket allows short bursts (the dashboard's encrypt, decrypt and
tamper buttons fire three requests in a row) while capping the sustained
rate. Memory is bounded: at most ``max_clients`` buckets are kept, least
recently seen first out, so a flood of spoofed source addresses costs a
fixed amount of RAM (an evicted client simply starts with a full bucket).

Clients are identified by the peer address. Behind a reverse proxy that is
the proxy, unless uvicorn is told to trust ``X-Forwarded-For``
(``--forwarded-allow-ips``); the limiter deliberately never reads that
header itself, because any client could forge it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RateLimited", "TokenBucketLimiter"]


@dataclass(frozen=True, slots=True)
class RateLimited:
    """A rejected request: how long until one token is available again."""

    retry_after_s: float

    @property
    def retry_after_header(self) -> str:
        """``Retry-After`` value: whole seconds, at least 1 (RFC 9110 10.2.3)."""
        return str(max(1, math.ceil(self.retry_after_s)))


class TokenBucketLimiter:
    """``rate`` tokens per second, ``burst`` tokens at most, one bucket per client key.

    Thread-safe; ``clock`` is injectable so tests can be deterministic.
    """

    def __init__(
        self,
        rate: float,
        burst: float,
        *,
        max_clients: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not (math.isfinite(rate) and rate > 0):
            raise ValueError("rate must be a positive number")
        if not (math.isfinite(burst) and burst >= 1):
            raise ValueError("burst must be at least 1")
        if max_clients < 1:
            raise ValueError("max_clients must be positive")
        self.rate = float(rate)
        self.burst = float(burst)
        self.max_clients = max_clients
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, t)
        self._lock = threading.Lock()

    def acquire(self, key: str, cost: float = 1.0) -> RateLimited | None:
        """Take ``cost`` tokens from ``key``'s bucket; ``None`` if allowed, else the wait."""
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self.burst, now))
            tokens = min(self.burst, tokens + max(0.0, now - last) * self.rate)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                allowed = True
            else:
                self._buckets[key] = (tokens, now)
                allowed = False
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        if allowed:
            return None
        return RateLimited(retry_after_s=(cost - tokens) / self.rate)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
