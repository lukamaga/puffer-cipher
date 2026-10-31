"""LivenessProof: commitments are exact, domain separated and beacon friendly."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest

from puffer.liveness import LivenessProof, challenge_commitment, observation_commitment


def test_challenge_commitment_matches_hashlib_directly():
    bits = [1, 0, 1, 1, 0, 0, 0, 1, 1]  # 9 bits -> 0xB1 0x80 (zero padded)
    expected = hashlib.blake2b(
        b"C" + struct.pack(">QI", 42, 9) + bytes([0xB1, 0x80]),
        digest_size=32,
        person=b"puffer-led",
    ).digest()
    assert challenge_commitment(42, bits) == expected


def test_observation_commitment_keeps_undecodable_distinct():
    observed = [1, 0, -1, 1]
    expected = hashlib.blake2b(
        b"O" + struct.pack(">QI", 7, 4) + bytes([0b1001_0000]) + bytes([0b1101_0000]),
        digest_size=32,
        person=b"puffer-led",
    ).digest()
    assert observation_commitment(7, observed) == expected
    assert observation_commitment(7, [1, 0, 0, 1]) != expected


def test_commitments_are_domain_separated_and_bound_to_the_window():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    c = challenge_commitment(10, bits)
    assert c != observation_commitment(10, bits)  # same bits, different kind
    assert c != challenge_commitment(11, bits)  # different first symbol
    assert c != challenge_commitment(10, bits[:7])  # different length, same packed byte prefix
    assert c != challenge_commitment(10, 1 - bits)
    assert len(c) == 32


@pytest.mark.parametrize(
    ("fn", "first", "data", "message"),
    [
        (challenge_commitment, 0, [0, 2], "0 or 1"),
        (observation_commitment, 0, [1, -2], "1, 0 or -1"),
        (challenge_commitment, -1, [1], "non-negative"),
    ],
)
def test_invalid_inputs(fn, first, data, message):
    with pytest.raises(ValueError, match=message):
        fn(first, data)


def make_proof(confidence: float = 31.97) -> LivenessProof:
    challenge = np.array([1, 0, 1, 1] * 16, dtype=np.uint8)
    observed = challenge.astype(np.int8)
    observed[:3] = [0, -1, 1]  # one wrong, one undecodable, one right
    return LivenessProof.build(
        window_start=1234.5678,
        window_end=1247.3678,
        first_symbol=6172,
        challenge=challenge,
        observed=observed,
        lag_s=0.0834,
        confidence_bits=confidence,
    )


def test_build_counts_matches_and_commits():
    proof = make_proof()
    assert proof.n_bits == 64
    assert proof.matches == 62
    assert proof.challenge_commitment == challenge_commitment(6172, [1, 0, 1, 1] * 16)
    with pytest.raises(ValueError, match="lengths"):
        LivenessProof.build(
            window_start=0.0,
            window_end=1.0,
            first_symbol=0,
            challenge=[1, 0],
            observed=[1],
            lag_s=0.0,
            confidence_bits=0.0,
        )


def test_to_dict_has_only_ints_and_strings_and_never_overstates():
    proof = make_proof(confidence=31.97)
    d = proof.to_dict()
    assert all(type(v) in (int, str) for v in d.values())
    assert d["confidence_bits"] == 31  # rounded down
    assert d["window_start_ms"] == 1234567
    assert d["window_end_ms"] == 1247367
    assert d["lag_ms"] == 83
    assert d["first_symbol"] == 6172
    assert d["matches"] == 62
    assert d["n_bits"] == 64
    assert bytes.fromhex(str(d["challenge_commitment"])) == proof.challenge_commitment
    assert bytes.fromhex(str(d["observation_commitment"])) == proof.observation_commitment
    # canonical beacon summary encoding (DESIGN.md 3.4) accepts it as is
    json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
