"""Deterministic cross-language test vectors for the beacon pulse format.

``python -m puffer.beacon.vectors --out tests/vectors/beacon_vectors.json``
writes the file that both pytest and the dashboard's vitest suite consume, so
the Python signer and the browser verifier are tested against the very same
bytes. ``--check`` verifies instead that the file on disk is up to date.

Everything is a pure function of fixed constants: the signing key comes from
a fixed 32-byte seed, every ``local_random`` is a BLAKE2b counter output and
timestamps are fixed. Ed25519 signatures are deterministic (RFC 8032), so
the file regenerates byte-identically. **The seeds are public: these keys
are for tests only and must never sign a real beacon.**

Negative vectors
----------------
Each invalid vector targets one rule and, wherever the rule can be isolated,
breaks *only* that rule: chain-rule vectors are re-signed with the real test
key (modelling a key holder who rewrites history, the very threat the chain
checks exist for), and a tampered signature gets a recomputed ``output``.
Verifiers that stop at the first failure and verifiers that collect every
failure therefore agree on ``expected_error``. The one inherent exception is
``wrong_public_key`` (a pulse signed by another key), whose signature also
fails under the trusted key; ``expected_error`` is still the first code in
check order.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from nacl.signing import SigningKey

from puffer.beacon.pulse import (
    GENESIS_PREV_OUTPUT,
    ErrorCode,
    Pulse,
    canonical_summary,
    compute_output,
    create_pulse,
    hash_summary,
)

VECTOR_SEED: Final = bytes(range(32))
"""Ed25519 seed of the vector signing key (0x00..0x1f). PUBLIC - tests only."""

FOREIGN_SEED: Final = bytes(range(32, 64))
"""Seed of an unrelated key, used for the ``wrong_public_key`` vectors."""

VECTOR_T0_MS: Final = 1_793_454_000_000  # 2026-10-31T13:40:00Z, a minute boundary
VECTOR_PERIOD_MS: Final = 60_000
VECTOR_CHAIN_LENGTH: Final = 5
DEFAULT_OUT: Final = Path("tests/vectors/beacon_vectors.json")

_LOCAL_RANDOM_PERSON: Final = b"puffer-vector-lr"  # 16 bytes, BLAKE2b's maximum
_FISH: Final = (("f1", "Nemo"), ("f2", "Dory"), ("f3", "Zoë"), ("f4", "Puffy"))
_TARGET: Final = 2  # invalid vectors tamper with pulse 2 and use pulse 1 as prev


def vector_signing_key() -> SigningKey:
    """The deterministic test-vector signing key."""
    return SigningKey(VECTOR_SEED)


def vector_local_random(counter: int) -> bytes:
    """Deterministic stand-in for PufferRNG: ``BLAKE2b-512(u64(counter))``, personalized."""
    return hashlib.blake2b(
        struct.pack(">Q", counter), digest_size=64, person=_LOCAL_RANDOM_PERSON
    ).digest()


def vector_summary(index: int) -> dict[str, Any]:
    """A realistic summary for pulse ``index``.

    It deliberately includes a non-ASCII fish name, so the canonical
    ``\\uXXXX`` escaping is exercised by every verifier that hashes it.
    """
    top = [
        {"fish_id": fish_id, "name": name, "bits": 120 - 17 * rank + index}
        for rank, (fish_id, name) in enumerate(_FISH[:3])
    ]
    return {
        "fish_count": len(_FISH),
        "fish_bits": 812 + 3 * index,
        "sensor_bits": 1530 + 5 * index,
        "system_bits": 256,
        "mode": "day" if index % 4 else "night",
        "liveness": "verified",
        "liveness_confidence_bits": 54 + index,
        "health": {"fish": "ok", "sensor": "ok" if index != 3 else "degraded", "system": "ok"},
        "top_fish": top,
        "frames": [1200 + 90 * index, 1289 + 90 * index],
    }


def build_chain(
    signing_key: SigningKey | None = None, length: int = VECTOR_CHAIN_LENGTH
) -> list[Pulse]:
    """The valid vector chain: pulse ``i`` reveals ``vector_local_random(i)``."""
    key = vector_signing_key() if signing_key is None else signing_key
    chain: list[Pulse] = []
    for index in range(length):
        chain.append(
            create_pulse(
                key,
                index=index,
                timestamp_ms=VECTOR_T0_MS + index * VECTOR_PERIOD_MS,
                period_ms=VECTOR_PERIOD_MS,
                local_random=vector_local_random(index),
                next_local_random=vector_local_random(index + 1),
                prev_output=chain[-1].output if chain else GENESIS_PREV_OUTPUT,
                summary=vector_summary(index),
            )
        )
    return chain


def build_vectors() -> dict[str, Any]:
    """The complete vector document: ``{public_key, valid_chain, invalid}``."""
    key = vector_signing_key()
    chain = build_chain(key)
    target, prev = chain[_TARGET], chain[_TARGET - 1]

    tampered_summary = vector_summary(_TARGET)
    tampered_summary["fish_bits"] += 1_000_000
    tampered_summary_json = canonical_summary(tampered_summary)

    flipped_signature = _flip(target.signature, 10)
    foreign = _resign(target, SigningKey(FOREIGN_SEED))

    cases: list[tuple[str, Pulse, Pulse | None, ErrorCode]] = [
        (
            "tampered_signature",
            dataclasses.replace(
                target,
                signature=flipped_signature,
                output=compute_output(target.message(), flipped_signature),
            ),
            prev,
            ErrorCode.BAD_SIGNATURE,
        ),
        (
            "tampered_summary",
            dataclasses.replace(target, summary=tampered_summary_json),
            prev,
            ErrorCode.BAD_SUMMARY_HASH,
        ),
        (
            "tampered_summary_hash",
            _resign(target, key, summary_hash=hash_summary(tampered_summary_json)),
            prev,
            ErrorCode.BAD_SUMMARY_HASH,
        ),
        (
            "tampered_output",
            dataclasses.replace(target, output=_flip(target.output, 0)),
            prev,
            ErrorCode.BAD_OUTPUT,
        ),
        (
            "tampered_prev_output",
            _resign(target, key, prev_output=_flip(target.prev_output, 63)),
            prev,
            ErrorCode.BROKEN_LINK,
        ),
        (
            "tampered_local_random",
            _resign(target, key, local_random=vector_local_random(1_000_000)),
            prev,
            ErrorCode.BAD_PRECOMMITMENT,
        ),
        (
            "index_gap",
            _resign(target, key, index=prev.index + 2),
            prev,
            ErrorCode.BAD_INDEX,
        ),
        (
            "timestamp_regression",
            _resign(target, key, timestamp_ms=prev.timestamp_ms - VECTOR_PERIOD_MS),
            prev,
            ErrorCode.BAD_TIMESTAMP,
        ),
        (
            "timestamp_repeated",
            _resign(target, key, timestamp_ms=prev.timestamp_ms),
            prev,
            ErrorCode.BAD_TIMESTAMP,
        ),
        ("wrong_public_key", foreign, prev, ErrorCode.WRONG_PUBLIC_KEY),
        (
            "relabelled_public_key",
            dataclasses.replace(target, public_key=foreign.public_key),
            prev,
            ErrorCode.WRONG_PUBLIC_KEY,
        ),
        (
            "bad_genesis",
            _resign(chain[0], key, prev_output=_flip(GENESIS_PREV_OUTPUT, 0)),
            None,
            ErrorCode.BAD_GENESIS,
        ),
    ]
    return {
        "public_key": bytes(key.verify_key).hex(),
        "valid_chain": [pulse.to_json() for pulse in chain],
        "invalid": [
            {
                "name": name,
                "pulse": pulse.to_json(),
                "prev": None if before is None else before.to_json(),
                "expected_error": str(code),
            }
            for name, pulse, before, code in cases
        ],
    }


def render_vectors() -> str:
    """The exact text of the vector file (stable key order, ASCII, trailing newline)."""
    return json.dumps(build_vectors(), indent=2, ensure_ascii=True) + "\n"


def _resign(pulse: Pulse, signing_key: SigningKey, **changes: Any) -> Pulse:
    """Apply ``changes`` and sign the result again, so only the changed rule breaks."""
    edited = dataclasses.replace(pulse, **changes)
    message = edited.message()
    signature = signing_key.sign(message).signature
    return dataclasses.replace(
        edited,
        signature=signature,
        output=compute_output(message, signature),
        public_key=bytes(signing_key.verify_key),
    )


def _flip(data: bytes, position: int) -> bytes:
    """``data`` with the lowest bit of byte ``position`` inverted."""
    mutable = bytearray(data)
    mutable[position] ^= 0x01
    return bytes(mutable)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m puffer.beacon.vectors",
        description="Write (or check) the deterministic beacon test vectors.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"output file (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the file differs instead of writing it"
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    text = render_vectors()
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else None
        if current != text:
            print(f"{out} is out of date; regenerate it without --check", file=sys.stderr)
            return 1
        print(f"{out} is up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
