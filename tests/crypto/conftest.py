"""Fixtures for the cryptography tests.

Keys and salts come from the seeded ``rng`` fixture so failures reproduce;
real code draws them from PufferRNG / libsodium. Argon2id runs with the
libsodium minimum cost so password-mode tests stay fast - the cost
parameters are stored in (and authenticated by) every file header, so the
code paths are identical to production.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from nacl.pwhash import argon2id

from puffer.crypto import RandomBytes


@pytest.fixture
def key(rng: np.random.Generator) -> bytes:
    return rng.bytes(32)


@pytest.fixture
def other_key(rng: np.random.Generator, key: bytes) -> bytes:
    other = rng.bytes(32)
    assert other != key
    return other


@pytest.fixture
def det_random(rng: np.random.Generator) -> RandomBytes:
    """A deterministic stand-in for PufferRNG (never for real secrets)."""
    return rng.bytes


@pytest.fixture
def fast_kdf(det_random: RandomBytes) -> dict[str, Any]:
    """encrypt_* keyword arguments for cheap, reproducible password mode."""
    return {
        "opslimit": argon2id.OPSLIMIT_MIN,
        "memlimit": argon2id.MEMLIMIT_MIN,
        "random_bytes": det_random,
    }
