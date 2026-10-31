"""Beacon pulse: the signed, hash-chained unit of public randomness (DESIGN.md 3.4).

A pulse is the fish-powered analogue of a NIST Randomness Beacon 2.0 record
or a drand round. This module defines its byte format, its JSON form, how it
is created and how it is verified. Everything here is a *protocol* built on
two primitives: Ed25519 (libsodium via ``nacl.signing``) and BLAKE2b
(``hashlib.blake2b``). No cryptographic primitive is implemented here.

Design decisions
----------------
* **A fixed-layout binary message is signed, never JSON.** ``M`` is 260 bytes
  with fixed offsets, so a verifier in any language (the browser verifier
  uses ``@noble/curves``) rebuilds it byte-for-byte from the JSON fields.
  Signing JSON would make every verifier depend on one serializer's quirks.
* **Domain separation.** ``M`` starts with :data:`DOMAIN`, so a beacon
  signature can never be confused with a signature over another protocol's
  message made with the same key, and a future format gets a new domain.
* **Hash chain.** ``prev_output`` links each pulse to its predecessor.
  Rewriting one old pulse forces re-signing every later pulse, which anyone
  who archived a single later pulse can detect.
* **Precommitment.** Pulse *i* publishes ``BLAKE2b-512(local_random_{i+1})``.
  The operator fixes its next contribution one period before revealing it,
  so it cannot choose ``local_random`` after seeing anything published in
  between; its only remaining power is to withhold a pulse, which is visible.
  ``local_random`` is 64 bytes of full-entropy output, so a plain hash is a
  hiding commitment.
* **The output covers the signature.** Ed25519 signatures are deterministic
  (RFC 8032), so ``output = BLAKE2b-512(M || signature)`` can be computed
  only by the key holder, even by someone who learned every field of ``M``
  in advance (for example from a leaked pending reveal).
* **The summary is hashed as an exact string.** Verifiers hash
  ``summary.encode("utf-8")`` and never re-serialize it. The server produces
  the canonical form once (:func:`canonical_summary`) and allows integers but
  no floats, because float formatting differs between languages.
* **JSON integers stay below 2**53.** ``index`` and ``timestamp_ms`` are u64
  in ``M`` but travel as JSON numbers; JavaScript parses those as IEEE-754
  doubles, so larger values could not be rebuilt exactly in the browser.

Every value compared during verification is public, so plain ``==`` is used
instead of constant-time comparison.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from nacl.exceptions import CryptoError
from nacl.signing import SigningKey, VerifyKey

DOMAIN: Final = b"PUFFER-BEACON-V1"
"""Domain-separation prefix of every signed beacon message (16 ASCII bytes)."""

VERSION: Final = 1
"""Pulse format version carried in the JSON form."""

LOCAL_RANDOM_BYTES: Final = 64
PRECOMMITMENT_BYTES: Final = 64
OUTPUT_BYTES: Final = 64
SUMMARY_HASH_BYTES: Final = 32
SIGNATURE_BYTES: Final = 64
PUBLIC_KEY_BYTES: Final = 32
MESSAGE_BYTES: Final = 260
"""``len(M)``: 16 domain + 8 index + 8 timestamp + 4 period + 3*64 + 32."""

GENESIS_PREV_OUTPUT: Final = bytes(OUTPUT_BYTES)
"""``prev_output`` of pulse 0: there is no predecessor to link to."""

MAX_JSON_INT: Final = 2**53 - 1
"""Largest integer that JavaScript's ``JSON.parse`` represents exactly."""

MAX_PERIOD_MS: Final = 2**32 - 1
"""``period_ms`` is a u32 in ``M``."""

_HEADER: Final = struct.Struct(">QQI")  # index, timestamp_ms, period_ms
_LOWER_HEX: Final = re.compile(r"[0-9a-f]*")
_MAX_SUMMARY_DEPTH: Final = 32

_HEX_FIELDS: Final[Mapping[str, int]] = {
    "local_random": LOCAL_RANDOM_BYTES,
    "precommitment": PRECOMMITMENT_BYTES,
    "prev_output": OUTPUT_BYTES,
    "summary_hash": SUMMARY_HASH_BYTES,
    "signature": SIGNATURE_BYTES,
    "output": OUTPUT_BYTES,
    "public_key": PUBLIC_KEY_BYTES,
}
_INT_FIELDS: Final = ("version", "index", "timestamp_ms", "period_ms")
_JSON_FIELDS: Final = frozenset((*_INT_FIELDS, "summary", *_HEX_FIELDS))


class BeaconFormatError(ValueError):
    """A pulse, summary or message does not match the DESIGN.md 3.4 format."""


class ErrorCode(enum.StrEnum):
    """Stable verification error codes, shared with the browser verifier.

    They are a wire contract (the dashboard and the test vectors use the
    string values), so existing values must never be renamed. Members compare
    equal to their string values. The declaration order is the order in which
    :func:`verify_pulse` runs its checks and reports failures.
    """

    WRONG_PUBLIC_KEY = "wrong_public_key"
    BAD_SUMMARY_HASH = "bad_summary_hash"
    BAD_SIGNATURE = "bad_signature"
    BAD_OUTPUT = "bad_output"
    BAD_GENESIS = "bad_genesis"
    BAD_INDEX = "bad_index"
    BAD_TIMESTAMP = "bad_timestamp"
    BROKEN_LINK = "broken_link"
    BAD_PRECOMMITMENT = "bad_precommitment"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of :func:`verify_pulse`: every failed check, in check order."""

    errors: tuple[ErrorCode, ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when no check failed."""
        return not self.errors


# --------------------------------------------------------------------------
# Hashes used by the format
# --------------------------------------------------------------------------


def hash_summary(summary: str) -> bytes:
    """``BLAKE2b-256(summary.encode("utf-8"))``: the exact string, never re-serialized."""
    return hashlib.blake2b(summary.encode("utf-8"), digest_size=SUMMARY_HASH_BYTES).digest()


def precommit(local_random: bytes) -> bytes:
    """``BLAKE2b-512(local_random)``: the commitment published one pulse early."""
    return hashlib.blake2b(local_random, digest_size=PRECOMMITMENT_BYTES).digest()


def compute_output(message: bytes, signature: bytes) -> bytes:
    """``BLAKE2b-512(M || signature)``: the pulse's public random output."""
    return hashlib.blake2b(message + signature, digest_size=OUTPUT_BYTES).digest()


# --------------------------------------------------------------------------
# Canonical summary JSON
# --------------------------------------------------------------------------


def canonical_summary(obj: Mapping[str, Any]) -> str:
    """Serialize a pulse summary to its one canonical JSON string.

    ``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)``
    over a value tree that may contain only objects with string keys, lists
    (or tuples), strings, booleans and integers in the JavaScript-safe range.
    Floats (including NaN and infinities) and ``null`` are rejected rather
    than coerced: entropy amounts are integer bits by contract, and refusing
    early keeps a bad summary from ever being signed. The output is pure
    ASCII, so its UTF-8 encoding is identical in every language.

    Raises:
        BeaconFormatError: the value tree contains a disallowed type.
    """
    if not isinstance(obj, Mapping):
        raise BeaconFormatError("summary must be a JSON object (a mapping)")
    plain = _plain_json(obj, "summary", 0)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def check_canonical_summary(summary: str) -> None:
    """Require ``summary`` to be exactly what :func:`canonical_summary` would emit.

    Used wherever the server is about to commit to a summary string it did not
    serialize itself in this call (a stored pending reveal, a caller-supplied
    string), so a non-canonical summary is refused before it is signed.

    Raises:
        BeaconFormatError: not valid JSON, not canonical, or disallowed types.
    """
    if not isinstance(summary, str):
        raise BeaconFormatError("summary must be a str")
    try:
        parsed = json.loads(summary)
    except (ValueError, RecursionError) as exc:
        # ValueError covers JSONDecodeError and the int digit limit;
        # RecursionError a hostile nesting depth. All are format errors.
        raise BeaconFormatError(f"summary is not valid JSON: {exc!s:.200}") from exc
    if canonical_summary(parsed) != summary:
        raise BeaconFormatError("summary string is not in canonical form")


def _plain_json(value: object, path: str, depth: int) -> object:
    """Validate a summary value tree and convert it to plain dict/list/scalars."""
    if depth > _MAX_SUMMARY_DEPTH:
        raise BeaconFormatError(f"{path}: nested deeper than {_MAX_SUMMARY_DEPTH} levels")
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_JSON_INT:
            raise BeaconFormatError(f"{path}: integer outside the JavaScript-safe range")
        return value
    if isinstance(value, float):
        raise BeaconFormatError(f"{path}: floats are not allowed (use integer bits)")
    if isinstance(value, Mapping):
        plain: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BeaconFormatError(f"{path}: object keys must be str, got {type(key)!r}")
            plain[key] = _plain_json(item, f"{path}.{key}", depth + 1)
        return plain
    if isinstance(value, list | tuple):
        return [_plain_json(item, f"{path}[{i}]", depth + 1) for i, item in enumerate(value)]
    raise BeaconFormatError(f"{path}: unsupported summary value type {type(value).__name__}")


# --------------------------------------------------------------------------
# Signed message
# --------------------------------------------------------------------------


def build_message(
    *,
    index: int,
    timestamp_ms: int,
    period_ms: int,
    local_random: bytes,
    precommitment: bytes,
    prev_output: bytes,
    summary_hash: bytes,
) -> bytes:
    """Assemble the 260-byte signed message ``M`` of DESIGN.md 3.4.

    Raises:
        BeaconFormatError: a field is out of range or has the wrong length.
    """
    _check_uint("index", index, MAX_JSON_INT)
    _check_uint("timestamp_ms", timestamp_ms, MAX_JSON_INT)
    _check_uint("period_ms", period_ms, MAX_PERIOD_MS)
    _check_bytes("local_random", local_random, LOCAL_RANDOM_BYTES)
    _check_bytes("precommitment", precommitment, PRECOMMITMENT_BYTES)
    _check_bytes("prev_output", prev_output, OUTPUT_BYTES)
    _check_bytes("summary_hash", summary_hash, SUMMARY_HASH_BYTES)
    message = b"".join(
        (
            DOMAIN,
            _HEADER.pack(index, timestamp_ms, period_ms),
            local_random,
            precommitment,
            prev_output,
            summary_hash,
        )
    )
    assert len(message) == MESSAGE_BYTES
    return message


def _check_uint(name: str, value: object, maximum: int) -> None:
    # ``type(...) is int`` rejects bools (an int subclass) and int-like objects.
    if type(value) is not int or not 0 <= value <= maximum:
        raise BeaconFormatError(f"{name} must be an int in [0, {maximum}], got {value!r}")


def _check_bytes(name: str, value: object, length: int) -> None:
    if type(value) is not bytes or len(value) != length:
        raise BeaconFormatError(f"{name} must be exactly {length} bytes")


# --------------------------------------------------------------------------
# Pulse
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pulse:
    """One beacon pulse, field-for-field the JSON form of DESIGN.md 3.4.

    Construction validates the *format* (types, ranges, lengths, version) so
    a malformed pulse cannot exist in memory; it does not validate the
    *cryptography* - that is :func:`verify_pulse`'s job, and a well-formed
    but forged pulse must remain representable so it can be reported.
    """

    version: int
    index: int
    timestamp_ms: int
    period_ms: int
    local_random: bytes
    precommitment: bytes
    prev_output: bytes
    summary: str
    summary_hash: bytes
    signature: bytes
    output: bytes
    public_key: bytes

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != VERSION:
            raise BeaconFormatError(f"unsupported pulse version {self.version!r}")
        _check_uint("index", self.index, MAX_JSON_INT)
        _check_uint("timestamp_ms", self.timestamp_ms, MAX_JSON_INT)
        _check_uint("period_ms", self.period_ms, MAX_PERIOD_MS)
        for name, length in _HEX_FIELDS.items():
            _check_bytes(name, getattr(self, name), length)
        if type(self.summary) is not str:
            raise BeaconFormatError("summary must be a str")
        try:
            # Lone surrogates have no UTF-8 encoding; JavaScript's TextEncoder
            # would silently replace them, so verifiers would disagree.
            self.summary.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BeaconFormatError("summary is not encodable as UTF-8") from exc

    def message(self) -> bytes:
        """The 260-byte message ``M`` that :attr:`signature` signs."""
        return build_message(
            index=self.index,
            timestamp_ms=self.timestamp_ms,
            period_ms=self.period_ms,
            local_random=self.local_random,
            precommitment=self.precommitment,
            prev_output=self.prev_output,
            summary_hash=self.summary_hash,
        )

    def to_json(self) -> dict[str, int | str]:
        """JSON-ready dict in DESIGN.md field order; bytes as lowercase hex."""
        return {
            "version": self.version,
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            "period_ms": self.period_ms,
            "local_random": self.local_random.hex(),
            "precommitment": self.precommitment.hex(),
            "prev_output": self.prev_output.hex(),
            "summary": self.summary,
            "summary_hash": self.summary_hash.hex(),
            "signature": self.signature.hex(),
            "output": self.output.hex(),
            "public_key": self.public_key.hex(),
        }

    @classmethod
    def from_json(cls, data: object) -> Pulse:
        """Parse the JSON form strictly.

        Pulses arrive from untrusted places (files, other servers), so this
        accepts exactly the documented field set: integers must be real JSON
        integers (not bools or floats), byte fields must be lowercase hex of
        the exact length, and the version must be known. Being stricter than
        necessary keeps every implementation agreeing on what a pulse is.

        Raises:
            BeaconFormatError: anything deviates from the format.
        """
        if not isinstance(data, Mapping):
            raise BeaconFormatError("pulse must be a JSON object")
        keys = set(data)
        if keys != _JSON_FIELDS:
            missing = sorted(_JSON_FIELDS - keys)
            extra = sorted(str(k) for k in keys - _JSON_FIELDS)
            raise BeaconFormatError(f"pulse fields mismatch: missing={missing} extra={extra}")
        ints: dict[str, int] = {}
        for name in _INT_FIELDS:
            value = data[name]
            if type(value) is not int:
                raise BeaconFormatError(f"{name} must be a JSON integer")
            ints[name] = value
        summary = data["summary"]
        if type(summary) is not str:
            raise BeaconFormatError("summary must be a JSON string")
        raw = {name: _parse_hex(name, data[name], length) for name, length in _HEX_FIELDS.items()}
        return cls(
            version=ints["version"],
            index=ints["index"],
            timestamp_ms=ints["timestamp_ms"],
            period_ms=ints["period_ms"],
            local_random=raw["local_random"],
            precommitment=raw["precommitment"],
            prev_output=raw["prev_output"],
            summary=summary,
            summary_hash=raw["summary_hash"],
            signature=raw["signature"],
            output=raw["output"],
            public_key=raw["public_key"],
        )


def _parse_hex(name: str, value: object, length: int) -> bytes:
    # bytes.fromhex alone would also accept uppercase digits and whitespace.
    if type(value) is not str or len(value) != 2 * length or not _LOWER_HEX.fullmatch(value):
        raise BeaconFormatError(f"{name} must be {2 * length} lowercase hex characters")
    return bytes.fromhex(value)


# --------------------------------------------------------------------------
# Create and verify
# --------------------------------------------------------------------------


def create_pulse(
    signing_key: SigningKey,
    *,
    index: int,
    timestamp_ms: int,
    period_ms: int,
    local_random: bytes,
    next_local_random: bytes,
    prev_output: bytes,
    summary: Mapping[str, Any] | str,
) -> Pulse:
    """Sign a new pulse that reveals ``local_random`` and precommits to the next one.

    ``summary`` is either a mapping (serialized with :func:`canonical_summary`)
    or an already-canonical string (checked with
    :func:`check_canonical_summary`). Chain consistency with the previous
    pulse is the caller's responsibility (the store re-verifies it on append);
    this function only refuses inputs that could never verify on their own.

    Raises:
        BeaconFormatError: a field is malformed or the summary is not canonical.
        ValueError: ``index == 0`` with a non-zero ``prev_output``.
    """
    if isinstance(summary, str):
        check_canonical_summary(summary)
        summary_json = summary
    else:
        summary_json = canonical_summary(summary)
    _check_bytes("next_local_random", next_local_random, LOCAL_RANDOM_BYTES)
    if index == 0 and prev_output != GENESIS_PREV_OUTPUT:
        raise ValueError("the genesis pulse (index 0) must have an all-zero prev_output")
    precommitment = precommit(next_local_random)
    summary_digest = hash_summary(summary_json)
    message = build_message(
        index=index,
        timestamp_ms=timestamp_ms,
        period_ms=period_ms,
        local_random=local_random,
        precommitment=precommitment,
        prev_output=prev_output,
        summary_hash=summary_digest,
    )
    signature = signing_key.sign(message).signature
    return Pulse(
        version=VERSION,
        index=index,
        timestamp_ms=timestamp_ms,
        period_ms=period_ms,
        local_random=local_random,
        precommitment=precommitment,
        prev_output=prev_output,
        summary=summary_json,
        summary_hash=summary_digest,
        signature=signature,
        output=compute_output(message, signature),
        public_key=bytes(signing_key.verify_key),
    )


def verify_pulse(
    pulse: Pulse, *, public_key: bytes | None = None, prev: Pulse | None = None
) -> VerificationResult:
    """Run every DESIGN.md 3.4 check and report all failures, in check order.

    The signature is verified only against the *trusted* key, never blindly
    against the key embedded in the pulse: ``public_key`` when given,
    otherwise ``prev.public_key`` (so a chain cannot silently switch keys),
    otherwise the embedded key (which proves only internal consistency). An
    embedded key that differs from the trusted one is ``wrong_public_key``.

    ``prev`` is the pulse with index ``pulse.index - 1``; it is assumed to be
    verified already and is used only for the chain checks. All checks run
    even after a failure, so a report lists every problem; the documented
    order (see :class:`ErrorCode`) keeps ``errors[0]`` stable across
    implementations.

    Raises:
        ValueError: ``public_key`` is given but is not 32 bytes.
    """
    if public_key is not None:
        if type(public_key) is not bytes or len(public_key) != PUBLIC_KEY_BYTES:
            raise ValueError(f"public_key must be exactly {PUBLIC_KEY_BYTES} bytes")
        trusted = public_key
    elif prev is not None:
        trusted = prev.public_key
    else:
        trusted = pulse.public_key

    errors: list[ErrorCode] = []
    if pulse.public_key != trusted:
        errors.append(ErrorCode.WRONG_PUBLIC_KEY)
    if hash_summary(pulse.summary) != pulse.summary_hash:
        errors.append(ErrorCode.BAD_SUMMARY_HASH)
    message = pulse.message()
    if not _signature_valid(trusted, message, pulse.signature):
        errors.append(ErrorCode.BAD_SIGNATURE)
    if compute_output(message, pulse.signature) != pulse.output:
        errors.append(ErrorCode.BAD_OUTPUT)
    if pulse.index == 0 and pulse.prev_output != GENESIS_PREV_OUTPUT:
        errors.append(ErrorCode.BAD_GENESIS)
    if prev is not None:
        if pulse.index != prev.index + 1:
            errors.append(ErrorCode.BAD_INDEX)
        if pulse.timestamp_ms <= prev.timestamp_ms:
            errors.append(ErrorCode.BAD_TIMESTAMP)
        if pulse.prev_output != prev.output:
            errors.append(ErrorCode.BROKEN_LINK)
        if precommit(pulse.local_random) != prev.precommitment:
            errors.append(ErrorCode.BAD_PRECOMMITMENT)
    return VerificationResult(tuple(errors))


def _signature_valid(public_key: bytes, message: bytes, signature: bytes) -> bool:
    try:
        VerifyKey(public_key).verify(message, signature)
    except CryptoError:  # BadSignatureError, or a key libsodium refuses outright
        return False
    return True
