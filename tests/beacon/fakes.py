"""Deterministic test doubles for the beacon: a manual clock and seeded entropy."""

from __future__ import annotations

from typing import Any

import numpy as np

from puffer.beacon.vectors import VECTOR_T0_MS


class FakeClock:
    """Manually advanced millisecond clock; optionally auto-advances on every read."""

    def __init__(self, now_ms: int = VECTOR_T0_MS, step_ms: int = 0) -> None:
        self.now_ms = now_ms
        self.step_ms = step_ms

    def __call__(self) -> int:
        now = self.now_ms
        self.now_ms += self.step_ms
        return now

    def advance(self, ms: int) -> None:
        self.now_ms += ms


class FakeEntropy:
    """Seeded stand-in for PufferRNG + summary; records every value it hands out.

    Set :attr:`fail_next` to make that many upcoming calls raise, as a dead
    camera or an exhausted pool would.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.draws: list[tuple[bytes, dict[str, Any]]] = []
        self.fail_next = 0

    def __call__(self, n: int, purpose: str) -> tuple[bytes, dict[str, Any]]:
        assert purpose == "beacon"
        if self.fail_next:
            self.fail_next -= 1
            raise OSError("camera unplugged")
        value = self._rng.bytes(n)
        summary = {"draw": len(self.draws), "fish_bits": int(self._rng.integers(0, 4096))}
        self.draws.append((value, summary))
        return value, summary
