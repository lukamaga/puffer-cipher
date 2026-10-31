"""Pulse format, canonical summaries, signing and single-pulse verification."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import Any

import nacl.bindings
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from nacl.signing import SigningKey

from puffer.beacon.pulse import (
    DOMAIN,
    GENESIS_PREV_OUTPUT,
    MAX_JSON_INT,
    MESSAGE_BYTES,
    BeaconFormatError,
    ErrorCode,
    Pulse,
    build_message,
    canonical_summary,
    check_canonical_summary,
    create_pulse,
    verify_pulse,
)
from puffer.types import HealthState

SUMMARY = {"fish_bits": 812, "mode": "day", "top_fish": [{"fish_id": "f1", "bits": 3}]}


def _pulse(key: SigningKey, index: int = 7, **overrides: Any) -> Pulse:
    fields: dict[str, Any] = {
        "index": index,
        "timestamp_ms": 1_793_454_000_000,
        "period_ms": 60_000,
        "local_random": bytes([0xA1]) * 64,
        "next_local_random": bytes([0xB2]) * 64,
        "prev_output": GENESIS_PREV_OUTPUT if index == 0 else bytes([0xC3]) * 64,
        "summary": SUMMARY,
    }
    fields.update(overrides)
    return create_pulse(key, **fields)


# --------------------------------------------------------------------------
# Message layout
# --------------------------------------------------------------------------


def test_domain_and_message_size() -> None:
    assert DOMAIN == b"PUFFER-BEACON-V1"
    assert len(DOMAIN) == 16
    assert MESSAGE_BYTES == 260


def test_message_layout_has_exact_offsets() -> None:
    local_random, precommitment = bytes([1]) * 64, bytes([2]) * 64
    prev_output, summary_hash = bytes([3]) * 64, bytes([4]) * 32
    message = build_message(
        index=0x0102030405060708 & MAX_JSON_INT,
        timestamp_ms=1_793_454_000_000,
        period_ms=60_000,
        local_random=local_random,
        precommitment=precommitment,
        prev_output=prev_output,
        summary_hash=summary_hash,
    )
    assert len(message) == 260
    assert message[0:16] == b"PUFFER-BEACON-V1"
    assert message[16:24] == (0x0102030405060708 & MAX_JSON_INT).to_bytes(8, "big")
    assert message[24:32] == (1_793_454_000_000).to_bytes(8, "big")
    assert message[32:36] == (60_000).to_bytes(4, "big")
    assert message[36:100] == local_random
    assert message[100:164] == precommitment
    assert message[164:228] == prev_output
    assert message[228:260] == summary_hash


def test_pulse_message_matches_independent_assembly(signing_key: SigningKey) -> None:
    pulse = _pulse(signing_key)
    expected = (
        b"PUFFER-BEACON-V1"
        + struct.pack(">Q", pulse.index)
        + struct.pack(">Q", pulse.timestamp_ms)
        + struct.pack(">I", pulse.period_ms)
        + pulse.local_random
        + pulse.precommitment
        + pulse.prev_output
        + pulse.summary_hash
    )
    assert pulse.message() == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index", -1),
        ("index", MAX_JSON_INT + 1),
        ("timestamp_ms", True),
        ("period_ms", 2**32),
        ("local_random", bytes(63)),
        ("precommitment", bytearray(64)),
        ("prev_output", bytes(65)),
        ("summary_hash", bytes(64)),
    ],
)
def test_build_message_rejects_out_of_range_fields(field: str, value: object) -> None:
    fields: dict[str, Any] = {
        "index": 1,
        "timestamp_ms": 2,
        "period_ms": 3,
        "local_random": bytes(64),
        "precommitment": bytes(64),
        "prev_output": bytes(64),
        "summary_hash": bytes(32),
        field: value,
    }
    with pytest.raises(BeaconFormatError, match=field):
        build_message(**fields)


# --------------------------------------------------------------------------
# Creation and verification
# --------------------------------------------------------------------------


def test_create_pulse_derives_every_field_per_spec(signing_key: SigningKey) -> None:
    pulse = _pulse(signing_key)
    assert pulse.version == 1
    assert pulse.summary == canonical_summary(SUMMARY)
    assert pulse.summary_hash == hashlib.blake2b(pulse.summary.encode(), digest_size=32).digest()
    assert pulse.precommitment == hashlib.blake2b(bytes([0xB2]) * 64, digest_size=64).digest()
    assert pulse.public_key == bytes(signing_key.verify_key)
    message = pulse.message()
    # Independent check through the raw libsodium binding.
    assert nacl.bindings.crypto_sign_open(pulse.signature + message, pulse.public_key) == message
    assert pulse.output == hashlib.blake2b(message + pulse.signature, digest_size=64).digest()


def test_sign_verify_round_trip(signing_key: SigningKey) -> None:
    pulse = _pulse(signing_key)
    result = verify_pulse(pulse, public_key=bytes(signing_key.verify_key))
    assert result.ok
    assert result.errors == ()


def test_create_pulse_accepts_canonical_string_summary(signing_key: SigningKey) -> None:
    text = canonical_summary(SUMMARY)
    assert _pulse(signing_key, summary=text).summary == text


@pytest.mark.parametrize(
    "summary",
    ['{"b":1,"a":2}', '{"a": 1}', '{"a":1.5}', "not json", "[1,2]"],
)
def test_create_pulse_rejects_non_canonical_string_summary(
    signing_key: SigningKey, summary: str
) -> None:
    with pytest.raises(BeaconFormatError, match="summary"):
        _pulse(signing_key, summary=summary)


def test_create_pulse_rejects_genesis_with_link(signing_key: SigningKey) -> None:
    with pytest.raises(ValueError, match="genesis"):
        _pulse(signing_key, index=0, prev_output=bytes([1]) * 64)


def test_create_pulse_rejects_short_next_local_random(signing_key: SigningKey) -> None:
    with pytest.raises(BeaconFormatError, match="next_local_random"):
        _pulse(signing_key, next_local_random=bytes(32))


def test_verify_detects_each_tampered_field(signing_key: SigningKey) -> None:
    pulse = _pulse(signing_key)
    pk = bytes(signing_key.verify_key)
    flip = bytes([0x80]) + bytes(63)
    cases = {
        "summary": (dataclasses.replace(pulse, summary='{"x":1}'), ErrorCode.BAD_SUMMARY_HASH),
        "signature": (
            dataclasses.replace(pulse, signature=bytes(64)),
            ErrorCode.BAD_SIGNATURE,
        ),
        "output": (dataclasses.replace(pulse, output=bytes(64)), ErrorCode.BAD_OUTPUT),
        "local_random": (
            dataclasses.replace(pulse, local_random=bytes(64)),
            ErrorCode.BAD_SIGNATURE,
        ),
        "precommitment": (
            dataclasses.replace(pulse, precommitment=flip),
            ErrorCode.BAD_SIGNATURE,
        ),
    }
    for name, (tampered, code) in cases.items():
        assert code in verify_pulse(tampered, public_key=pk).errors, name


def test_genesis_rule_is_enforced(signing_key: SigningKey) -> None:
    genesis = _pulse(signing_key, index=0)
    assert verify_pulse(genesis).ok
    linked = dataclasses.replace(genesis, prev_output=bytes([9]) * 64)
    assert ErrorCode.BAD_GENESIS in verify_pulse(linked).errors


def test_prev_pins_the_key_when_no_public_key_is_given(signing_key: SigningKey) -> None:
    prev = _pulse(signing_key, index=0)
    other = SigningKey(bytes([7]) * 32)
    pulse = create_pulse(
        other,
        index=1,
        timestamp_ms=prev.timestamp_ms + 1,
        period_ms=60_000,
        local_random=bytes([0xB2]) * 64,
        next_local_random=bytes(64),
        prev_output=prev.output,
        summary=SUMMARY,
    )
    assert verify_pulse(pulse).ok  # self-consistent under its own key ...
    errors = verify_pulse(pulse, prev=prev).errors  # ... but not a continuation of prev
    assert errors == (ErrorCode.WRONG_PUBLIC_KEY, ErrorCode.BAD_SIGNATURE)


def test_verify_rejects_malformed_public_key(signing_key: SigningKey) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        verify_pulse(_pulse(signing_key), public_key=bytes(31))


def test_small_order_public_key_is_reported_not_raised(signing_key: SigningKey) -> None:
    pulse = dataclasses.replace(_pulse(signing_key), public_key=bytes(32))
    errors = verify_pulse(pulse).errors
    assert errors == (ErrorCode.BAD_SIGNATURE,)


# --------------------------------------------------------------------------
# JSON form
# --------------------------------------------------------------------------


def test_json_round_trip_is_lossless_and_lowercase(signing_key: SigningKey) -> None:
    pulse = _pulse(signing_key)
    data = pulse.to_json()
    assert list(data) == [
        "version",
        "index",
        "timestamp_ms",
        "period_ms",
        "local_random",
        "precommitment",
        "prev_output",
        "summary",
        "summary_hash",
        "signature",
        "output",
        "public_key",
    ]
    for name in ("local_random", "signature", "output", "public_key", "summary_hash"):
        value = data[name]
        assert isinstance(value, str)
        assert value == value.lower()
    assert Pulse.from_json(json.loads(json.dumps(data))) == pulse


def _mutated(pulse: Pulse, **changes: object) -> dict[str, object]:
    data: dict[str, object] = dict(pulse.to_json())
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"version": 2}, "version"),
        ({"version": True}, "version"),
        ({"index": True}, "index"),
        ({"index": 7.0}, "index"),
        ({"index": "7"}, "index"),
        ({"index": -1}, "index"),
        ({"timestamp_ms": MAX_JSON_INT + 1}, "timestamp_ms"),
        ({"period_ms": 2**32}, "period_ms"),
        ({"summary": {"fish_bits": 1}}, "summary"),
        ({"summary": "\ud800"}, "UTF-8"),
        ({"output": "AB" * 64}, "output"),
        ({"output": "ab" * 63}, "output"),
        ({"signature": "zz" * 64}, "signature"),
        ({"public_key": " " + "ab" * 31 + "a"}, "public_key"),
        ({"local_random": None}, "local_random"),
    ],
)
def test_from_json_is_strict(
    signing_key: SigningKey, changes: dict[str, object], match: str
) -> None:
    with pytest.raises(BeaconFormatError, match=match):
        Pulse.from_json(_mutated(_pulse(signing_key), **changes))


def test_from_json_requires_exact_field_set(signing_key: SigningKey) -> None:
    data = _mutated(_pulse(signing_key))
    with pytest.raises(BeaconFormatError, match="extra"):
        Pulse.from_json({**data, "comment": "hi"})
    del data["precommitment"]
    with pytest.raises(BeaconFormatError, match="missing"):
        Pulse.from_json(data)


@pytest.mark.parametrize("value", [[], "pulse", None, 42])
def test_from_json_requires_an_object(value: object) -> None:
    with pytest.raises(BeaconFormatError, match="object"):
        Pulse.from_json(value)


def test_direct_construction_validates_format(signing_key: SigningKey) -> None:
    pulse = _pulse(signing_key)
    with pytest.raises(BeaconFormatError, match="output"):
        dataclasses.replace(pulse, output=bytes(10))
    with pytest.raises(BeaconFormatError, match="version"):
        dataclasses.replace(pulse, version=0)
    with pytest.raises(BeaconFormatError, match="summary"):
        dataclasses.replace(pulse, summary=b"{}")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Canonical summary
# --------------------------------------------------------------------------


def test_canonical_summary_is_sorted_compact_ascii() -> None:
    obj = {"b": [1, True, "x"], "a": {"z": 0, "é": "Zoë"}, "c": False}
    text = canonical_summary(obj)
    assert text == json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert text.startswith('{"a":{"z":0,"\\u00e9":"Zo\\u00eb')
    assert text.isascii()


def test_canonical_summary_normalizes_tuples_and_str_enums() -> None:
    text = canonical_summary({"frames": (1, 2), "health": {"fish": HealthState.OK}})
    assert text == '{"frames":[1,2],"health":{"fish":"ok"}}'


@pytest.mark.parametrize(
    "obj",
    [
        {"fish_bits": 812.0},
        {"x": float("nan")},
        {"x": float("inf")},
        {"top_fish": [{"bits": 1.5}]},
        {"nested": {"deeper": [[0.1]]}},
        {"x": None},
        {"x": b"bytes"},
        {"x": {1, 2}},
        {1: "int key"},
        {"x": {("tuple",): 1}},
        {"x": 2**53},
        {"x": -(2**53)},
    ],
)
def test_canonical_summary_rejects_disallowed_values(obj: dict[Any, Any]) -> None:
    with pytest.raises(BeaconFormatError):
        canonical_summary(obj)


def test_canonical_summary_requires_an_object() -> None:
    with pytest.raises(BeaconFormatError, match="object"):
        canonical_summary([1, 2])  # type: ignore[arg-type]


def test_canonical_summary_rejects_runaway_nesting() -> None:
    obj: dict[str, Any] = {}
    cursor = obj
    for _ in range(40):
        cursor["n"] = {}
        cursor = cursor["n"]
    with pytest.raises(BeaconFormatError, match="nested"):
        canonical_summary(obj)


def test_check_canonical_summary() -> None:
    check_canonical_summary('{"a":1,"b":[true,"x"]}')
    for bad in ['{"b":1,"a":1}', '{"a":1} ', '{"a":1.0}', '{"a":"\\u00e9"', '"str"']:
        with pytest.raises(BeaconFormatError):
            check_canonical_summary(bad)
    with pytest.raises(BeaconFormatError, match="must be a str"):
        check_canonical_summary(b"{}")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

_json_scalars = st.booleans() | st.integers(-MAX_JSON_INT, MAX_JSON_INT) | st.text(max_size=12)
_json_values = st.recursive(
    _json_scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)
_summaries = st.dictionaries(st.text(max_size=8), _json_values, max_size=6)


@settings(max_examples=60, deadline=None)
@given(summary=_summaries)
def test_canonical_summary_is_a_fixed_point(summary: dict[str, Any]) -> None:
    text = canonical_summary(summary)
    assert text.isascii()
    check_canonical_summary(text)
    assert canonical_summary(json.loads(text)) == text


_BYTE_FIELDS = (
    "local_random",
    "precommitment",
    "prev_output",
    "summary_hash",
    "signature",
    "output",
)


@settings(max_examples=60, deadline=None)
@given(
    index=st.integers(1, MAX_JSON_INT),
    timestamp_ms=st.integers(0, MAX_JSON_INT),
    summary=_summaries,
    field=st.sampled_from(_BYTE_FIELDS),
    bit=st.integers(0, 32 * 8 - 1),
)
def test_any_single_bit_flip_is_detected(
    index: int, timestamp_ms: int, summary: dict[str, Any], field: str, bit: int
) -> None:
    key = SigningKey(bytes(range(32)))
    pulse = _pulse(key, index=index, timestamp_ms=timestamp_ms, summary=summary)
    assert verify_pulse(pulse, public_key=bytes(key.verify_key)).ok
    raw = bytearray(getattr(pulse, field))
    raw[bit // 8] ^= 1 << (bit % 8)
    changes: dict[str, Any] = {field: bytes(raw)}
    tampered = dataclasses.replace(pulse, **changes)
    assert not verify_pulse(tampered, public_key=bytes(key.verify_key)).ok
