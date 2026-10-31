"""Fixtures for beacon tests: deterministic keys, entropy, clock and stores."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from nacl.signing import SigningKey

from puffer.beacon.store import BeaconStore
from puffer.beacon.vectors import vector_signing_key

from .fakes import FakeClock, FakeEntropy


@pytest.fixture
def signing_key() -> SigningKey:
    """The public, deterministic test-vector key (never a real beacon key)."""
    return vector_signing_key()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def entropy(rng: np.random.Generator) -> FakeEntropy:
    return FakeEntropy(rng)


@pytest.fixture
def memory_store() -> Iterator[BeaconStore]:
    with BeaconStore() as store:
        yield store
