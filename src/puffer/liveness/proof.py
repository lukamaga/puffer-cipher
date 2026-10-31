"""Compact, auditable record of one liveness window.

The beacon publishes, with every pulse, *how* it knew the camera was live. A
:class:`LivenessProof` summarises the verifier's latest full window: which
symbols were checked, how many matched, at which camera lag, and how unlikely
that is by chance. Two BLAKE2b-256 commitments bind the record to the exact
challenge bits and the exact decoded observation, so anyone holding the
server's logs (or a recording of the LED) can later check that the published
numbers describe those bits and were not invented.

The commitments are *binding, not hiding*: challenge bits are public the
moment the LED displays them, so there is nothing to hide, and no salt is
used. Each commitment is domain separated by the personalisation string and a
one-byte kind tag, and covers the first symbol index and the bit count, so a
commitment for one window can never be mistaken for another window's.

Hashed data (big-endian; ``packbits`` puts bit 0 in the most significant bit
of byte 0 and zero-pads the last byte)::

    challenge   = 0x43 ("C") || u64(first_symbol) || u32(n_bits) || packbits(bits)
    observation = 0x4F ("O") || u64(first_symbol) || u32(n_bits)
                  || packbits(observed == 1) || packbits(observed is decodable)

``observed`` holds 1, 0, or -1 for a symbol that could not be decoded (no
usable frames or both halves alike); the second bitmap keeps "undecodable"
distinct from "decoded as 0".
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

PERSON: Final = b"puffer-led"
_KIND_CHALLENGE: Final = b"C"
_KIND_OBSERVATION: Final = b"O"


def challenge_commitment(first_symbol: int, bits: npt.ArrayLike) -> bytes:
    """BLAKE2b-256 commitment to the challenge bits of a window."""
    b = _bits(bits)
    return _commit(_KIND_CHALLENGE, first_symbol, b.size, np.packbits(b).tobytes())


def observation_commitment(first_symbol: int, observed: npt.ArrayLike) -> bytes:
    """BLAKE2b-256 commitment to the decoded observation (1 / 0 / -1 = undecodable)."""
    o = np.asarray(observed, dtype=np.int8).ravel()
    if np.any((o < -1) | (o > 1)):
        raise ValueError("observed symbols must be 1, 0 or -1")
    payload = np.packbits(o == 1).tobytes() + np.packbits(o >= 0).tobytes()
    return _commit(_KIND_OBSERVATION, first_symbol, o.size, payload)


@dataclass(frozen=True, slots=True)
class LivenessProof:
    """Summary of one fully evaluated verification window.

    Times are on the shared monotonic timeline (frame timestamps, i.e. the
    camera side: symbol start plus the estimated lag).
    """

    window_start: float
    window_end: float
    first_symbol: int  # schedule index of the first checked symbol
    n_bits: int
    matches: int
    lag_s: float
    confidence_bits: float
    challenge_commitment: bytes
    observation_commitment: bytes

    @classmethod
    def build(
        cls,
        *,
        window_start: float,
        window_end: float,
        first_symbol: int,
        challenge: npt.ArrayLike,
        observed: npt.ArrayLike,
        lag_s: float,
        confidence_bits: float,
    ) -> LivenessProof:
        """Create a proof from the raw window, computing matches and commitments."""
        c = _bits(challenge)
        o = np.asarray(observed, dtype=np.int8).ravel()
        if c.size != o.size:
            raise ValueError("challenge and observation lengths differ")
        return cls(
            window_start=float(window_start),
            window_end=float(window_end),
            first_symbol=int(first_symbol),
            n_bits=int(c.size),
            matches=int(np.count_nonzero(o == c)),
            lag_s=float(lag_s),
            confidence_bits=float(confidence_bits),
            challenge_commitment=challenge_commitment(first_symbol, c),
            observation_commitment=observation_commitment(first_symbol, o),
        )

    def to_dict(self) -> dict[str, int | str]:
        """JSON-ready form for beacon summaries: integers and strings only.

        Beacon summaries contain no floats (DESIGN.md 3.4), so times become
        integer milliseconds and the confidence is rounded *down* - a
        published claim must never overstate the evidence.
        """
        return {
            "window_start_ms": math.floor(self.window_start * 1000.0),
            "window_end_ms": math.floor(self.window_end * 1000.0),
            "first_symbol": self.first_symbol,
            "n_bits": self.n_bits,
            "matches": self.matches,
            "lag_ms": round(self.lag_s * 1000.0),
            "confidence_bits": math.floor(self.confidence_bits),
            "challenge_commitment": self.challenge_commitment.hex(),
            "observation_commitment": self.observation_commitment.hex(),
        }


def _bits(bits: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    b = np.asarray(bits).ravel()
    if np.any((b != 0) & (b != 1)):
        raise ValueError("bits must be 0 or 1")
    return b.astype(np.uint8)


def _commit(kind: bytes, first_symbol: int, n_bits: int, payload: bytes) -> bytes:
    if first_symbol < 0:
        raise ValueError("first_symbol must be non-negative")
    h = hashlib.blake2b(digest_size=32, person=PERSON)
    h.update(kind + struct.pack(">QI", first_symbol, n_bits) + payload)
    return h.digest()
