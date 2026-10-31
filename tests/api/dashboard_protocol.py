"""A line-by-line Python port of the dashboard's runtime decoders.

Source: ``dashboard/src/lib/protocol.ts``. The dashboard *rejects* a message
whose known fields have the wrong type (and counts it as malformed), so the
API tests run every message through these decoders. Keep this file in step
with protocol.ts; ``test_dashboard_decoder.py`` additionally runs the real
TypeScript decoders under Node when it is available.

JavaScript semantics that matter and are reproduced here: numbers must be
finite, integers must be *safe* integers (``3.0`` counts, ``2**53`` does not),
``null`` and a missing key are the same for nullable fields, unknown fields
are ignored.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

MISSING: Any = object()
MAX_SAFE_INTEGER = 2**53 - 1

Decoder = Callable[[Any, str], Any]


class DecodeError(ValueError):
    def __init__(self, path: str, expected: str, value: Any) -> None:
        super().__init__(f"{path or '<root>'}: expected {expected}, got {value!r:.60}")
        self.path = path


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def string(v: Any, path: str = "") -> str:
    if not isinstance(v, str):
        raise DecodeError(path, "string", v)
    return v


def number(v: Any, path: str = "") -> float:
    if not _is_number(v) or not math.isfinite(v):
        raise DecodeError(path, "finite number", v)
    return v


def integer(v: Any, path: str = "") -> int:
    if not _is_number(v) or not math.isfinite(v) or v != int(v) or abs(v) > MAX_SAFE_INTEGER:
        raise DecodeError(path, "integer", v)
    return int(v)


def boolean(v: Any, path: str = "") -> bool:
    if not isinstance(v, bool):
        raise DecodeError(path, "boolean", v)
    return v


def literal(*values: str) -> Decoder:
    def decode(v: Any, path: str = "") -> Any:
        if v not in values:
            raise DecodeError(path, " | ".join(values), v)
        return v

    return decode


def nullable(inner: Decoder) -> Decoder:
    def decode(v: Any, path: str = "") -> Any:
        return None if v is None or v is MISSING else inner(v, path)

    return decode


def array(item: Decoder) -> Decoder:
    def decode(v: Any, path: str = "") -> list[Any]:
        if not isinstance(v, list):
            raise DecodeError(path, "array", v)
        return [item(x, f"{path}[{i}]") for i, x in enumerate(v)]

    return decode


def record(value: Decoder) -> Decoder:
    def decode(v: Any, path: str = "") -> dict[str, Any]:
        if not isinstance(v, dict):
            raise DecodeError(path, "object", v)
        return {k: value(x, f"{path}.{k}") for k, x in v.items()}

    return decode


def obj(**shape: Decoder) -> Decoder:
    def decode(v: Any, path: str = "") -> dict[str, Any]:
        if not isinstance(v, dict):
            raise DecodeError(path, "object", v)
        return {
            key: dec(v.get(key, MISSING), f"{path}.{key}" if path else key)
            for key, dec in shape.items()
        }

    return decode


def tup(*items: Decoder) -> Decoder:
    def decode(v: Any, path: str = "") -> list[Any]:
        if not isinstance(v, list) or len(v) != len(items):
            raise DecodeError(path, f"array of length {len(items)}", v)
        return [d(x, f"{path}[{i}]") for i, (d, x) in enumerate(zip(items, v, strict=True))]

    return decode


def bit_string(v: Any, path: str = "") -> str:
    s = string(v, path)
    if not re.fullmatch(r"[01]*", s):
        raise DecodeError(path, "string of 0/1", v)
    return s


def observed_bit_string(v: Any, path: str = "") -> str:
    s = string(v, path)
    if not re.fullmatch(r"[01?]*", s):
        raise DecodeError(path, "string of 0/1/?", v)
    return s


def version_string(v: Any, path: str = "") -> str:
    return str(v) if _is_number(v) else string(v, path)


# -- shared vocabulary ----------------------------------------------------------

health_state = literal("ok", "startup", "degraded", "failed")
liveness_state = literal("disabled", "calibrating", "verified", "failed")
light_mode = literal("day", "night")
bbox = tup(number, number, number, number)
point = tup(number, number)

liveness_status = obj(
    state=liveness_state,
    match_ratio=number,
    lag_s=number,
    confidence_bits=number,
    expected=bit_string,
    observed=observed_bit_string,
    window_start=number,
    window_end=number,
)
channel_health = obj(state=health_state, reasons=array(string), checks=record(health_state))
health_report = obj(
    channel=nullable(string),
    state=health_state,
    reasons=array(string),
    checks=record(health_state),
)
fish_card = obj(
    fish_id=string,
    name=string,
    color=string,
    present=boolean,
    track_id=nullable(integer),
    stats=obj(
        speed_avg=number,
        speed_max=number,
        activity=number,
        chaos_score=number,
        bits_total=number,
        bits_today=number,
        distance_px=number,
        seen_s=number,
    ),
)
receipt_channel = obj(credited_bits=number, share=nullable(number), health=health_state)
entropy_receipt = obj(
    receipt_version=integer,
    purpose=string,
    bytes=integer,
    timestamp_ms=integer,
    frames=nullable(tup(integer, integer)),
    mode=light_mode,
    liveness=obj(state=liveness_state, confidence_bits=number),
    channels=obj(sensor=receipt_channel, fish=receipt_channel, system=receipt_channel),
    fish=array(obj(fish_id=string, name=nullable(string), bits=number, share=number)),
)
beacon_pulse = obj(
    version=integer,
    index=integer,
    timestamp_ms=integer,
    period_ms=integer,
    local_random=string,
    precommitment=string,
    prev_output=string,
    summary=string,
    summary_hash=string,
    signature=string,
    output=string,
    public_key=string,
)
pulse_summary = obj(
    fish_count=integer,
    fish_bits=integer,
    sensor_bits=integer,
    system_bits=integer,
    mode=light_mode,
    liveness=liveness_state,
    liveness_confidence_bits=integer,
    health=record(health_state),
    top_fish=array(obj(fish_id=string, name=string, bits=integer)),
    frames=nullable(tup(integer, integer)),
)
channel_entropy = obj(
    bits_per_s=number, h_per_sample=nullable(number), credited_total=number, health=health_state
)

MESSAGE_DECODERS: dict[str, Decoder] = {
    "hello": obj(
        version=version_string,
        source=string,
        width=integer,
        height=integer,
        fps=number,
        beacon_public_key=nullable(string),
    ),
    "frame": obj(
        index=integer,
        t=number,
        width=integer,
        height=integer,
        jpeg=nullable(string),
        tracks=array(
            obj(
                track_id=integer,
                fish_id=nullable(string),
                name=nullable(string),
                color=nullable(string),
                bbox=bbox,
                trail=array(point),
                speed=number,
                heading=number,
            )
        ),
    ),
    "entropy": obj(
        t=number,
        mode=light_mode,
        channels=obj(sensor=channel_entropy, fish=channel_entropy, system=channel_entropy),
        pool=obj(sensor=number, fish=number),
    ),
    "bits": obj(t=number, bits=bit_string),
    "health": obj(overall=health_state, channels=record(channel_health)),
    "led": liveness_status,
    "fish": obj(fish=array(fish_card)),
    "beacon": beacon_pulse,
}

REST_DECODERS: dict[str, Decoder] = {
    "status": obj(
        source=string,
        mode=light_mode,
        fps=number,
        uptime_s=number,
        frames=integer,
        pool=record(number),
        liveness=liveness_status,
        health=record(health_report),
    ),
    "random": obj(data=string, receipt=entropy_receipt),
    "encrypt": obj(ciphertext=string, key=string, key_id=string, receipt=entropy_receipt),
    "decrypt": obj(plaintext=string),
    "fish": array(fish_card),
    "leaderboard": array(
        obj(rank=integer, fish_id=string, name=string, chaos_score=number, bits=number)
    ),
    "publicKey": obj(algorithm=literal("Ed25519"), public_key=string),
    "pulse": beacon_pulse,
    "pulses": array(beacon_pulse),
}


def parse_server_message(raw: str) -> tuple[str, Any]:
    """``parseServerMessage`` for a known type: ``(type, decoded data)`` or DecodeError.

    Unknown types are reported as ``("unknown", None)``, like the dashboard,
    which ignores them.
    """
    envelope = json.loads(raw)
    if not isinstance(envelope, dict):
        raise DecodeError("", "object", envelope)
    message_type = envelope.get("type")
    if not isinstance(message_type, str):
        raise DecodeError("type", "string", message_type)
    decoder = MESSAGE_DECODERS.get(message_type)
    if decoder is None:
        return "unknown", None
    return message_type, decoder(envelope.get("data", MISSING), "data")
