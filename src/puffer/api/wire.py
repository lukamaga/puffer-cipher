"""JSON shapes of the wire protocol (DESIGN.md 7) - the single place they are built.

Every REST body and WebSocket message is produced by a function in this
module, so the contract with the dashboard (``dashboard/src/lib/protocol.ts``,
which *rejects* any message whose known fields have the wrong type) is
checked in one place by ``tests/api/test_wire.py``.

Rules enforced here rather than trusted upstream:

* **Finite numbers only.** ``json.dumps`` would emit ``NaN``/``Infinity``,
  which is not JSON and makes ``JSON.parse`` throw for the whole message.
  Required numbers fall back to ``0.0``; nullable ones become ``null``.
* **Enums as their string values**, nullable fields as ``null`` (never a
  missing key), and only the documented fields (no accidental leaks of new
  engine attributes to the public).
* **One encoder.** :func:`dumps` uses exactly the settings of FastAPI's
  ``JSONResponse``, so a beacon pulse is the same bytes over REST and over
  the socket.

Two notes on upstream values:

* ``LivenessStatus.observed`` may mark undecodable bits with ``?`` (DESIGN.md
  5); it is sent unchanged (the dashboard renders ``?`` as an undecodable
  cell), but any other character fails the whole string closed to ``""``.
* ``EntropyReceipt.to_dict()`` encodes "no frames" as ``[]``; the dashboard
  expects ``[first, last]`` or ``null``, so the API sends ``null``.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, Final

from puffer.api.ports import ChannelStatsLike, ReceiptLike, SnapshotLike
from puffer.beacon.pulse import Pulse
from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    HealthReport,
    HealthState,
    LightMode,
    LivenessState,
    LivenessStatus,
    TrackSnapshot,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROTOCOL_CHANNELS",
    "bits_message",
    "describe_source",
    "dumps",
    "entropy_message",
    "envelope",
    "fish_card",
    "frame_message",
    "health_message",
    "health_report",
    "hello_message",
    "identities_from_cards",
    "identities_from_labels",
    "leaderboard_entry",
    "liveness_status",
    "pulse_text",
    "pulses_text",
    "raw_envelope",
    "receipt",
    "status_message",
]

PROTOCOL_CHANNELS: Final = (CHANNEL_SENSOR, CHANNEL_FISH, CHANNEL_SYSTEM)
"""Channels every ``entropy`` message carries (the dashboard requires all three)."""

_HEALTH_VALUES: Final = frozenset(s.value for s in HealthState)
_LIVENESS_VALUES: Final = frozenset(s.value for s in LivenessState)
_MODE_VALUES: Final = frozenset(m.value for m in LightMode)
_COLOR: Final = re.compile(r"#[0-9a-fA-F]{6}")
_BITS: Final = re.compile(r"[01]*")
_OBSERVED: Final = re.compile(r"[01?]*")
_FALLBACK_COLOR: Final = "#8a8a8a"
_COORD_DIGITS: Final = 1  # 0.1 px is far below what the overlay can show
_RATE_DIGITS: Final = 3


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def dumps(obj: object) -> str:
    """Compact JSON with FastAPI ``JSONResponse`` settings; refuses NaN/Infinity."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def envelope(message_type: str, data: object) -> str:
    """A WebSocket text frame ``{"type": ..., "data": ...}`` (DESIGN.md 7.2)."""
    return dumps({"type": message_type, "data": data})


def raw_envelope(message_type: str, data_json: str) -> str:
    """Like :func:`envelope` around already-encoded JSON, keeping its bytes as they are."""
    return f'{{"type":{dumps(message_type)},"data":{data_json}}}'


def pulse_text(pulse: Pulse) -> str:
    """The pulse JSON (DESIGN.md 3.4) as served by REST and pushed over the socket."""
    return dumps(pulse.to_json())


def pulses_text(pulses: Iterable[Pulse]) -> str:
    """A JSON array of pulses, each byte-identical to :func:`pulse_text`."""
    return "[" + ",".join(pulse_text(p) for p in pulses) + "]"


# --------------------------------------------------------------------------
# Scalars
# --------------------------------------------------------------------------


def _finite(value: object, default: float = 0.0) -> float:
    """``value`` as a finite float, or ``default`` (also for bools and non-numbers)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _optional_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _round(value: object, digits: int) -> float:
    return round(_finite(value), digits)


def _enum_value(value: object) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _health(value: object, default: HealthState = HealthState.FAILED) -> str:
    """A health state string; anything unknown is reported as ``default`` (fail closed)."""
    text = _enum_value(value)
    return text if text in _HEALTH_VALUES else default.value


def _mode(value: object) -> str:
    text = _enum_value(value)
    return text if text in _MODE_VALUES else LightMode.DAY.value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _color(value: object) -> str | None:
    return value if isinstance(value, str) and _COLOR.fullmatch(value) else None


def describe_source(source_id: str) -> str:
    """Human-readable data source (DESIGN.md rule 4: always say where data came from).

    Public, so it never echoes a stream URL: ``camera:rtsp://user:pw@host``
    may carry credentials and names internal hosts.
    """
    kind, _, detail = source_id.partition(":")
    if "://" in detail or "@" in detail:
        detail = "(network stream)"
    if kind == "synthetic":
        return "synthetic simulator"
    if kind == "video":
        return f"recorded video {detail}".rstrip()
    if kind == "camera":
        return f"live camera {detail}".rstrip()
    if kind == "picamera":
        return "live camera (Pi camera)"
    if "://" in source_id or "@" in source_id:
        return "network stream"
    return source_id or "unknown"


# --------------------------------------------------------------------------
# Shared payloads
# --------------------------------------------------------------------------


def liveness_status(status: LivenessStatus) -> dict[str, Any]:
    """``types.LivenessStatus`` as JSON: the ``led`` message and ``/status.liveness``."""
    state = _enum_value(status.state)
    expected = status.expected if _BITS.fullmatch(status.expected) else ""
    return {
        "state": state if state in _LIVENESS_VALUES else LivenessState.FAILED.value,
        "match_ratio": _finite(status.match_ratio),
        "lag_s": _finite(status.lag_s),
        "confidence_bits": _finite(status.confidence_bits),
        "expected": expected,
        "observed": status.observed if _OBSERVED.fullmatch(status.observed) else "",
        "window_start": _finite(status.window_start),
        "window_end": _finite(status.window_end),
    }


def health_report(report: HealthReport, *, with_channel: bool) -> dict[str, Any]:
    """One channel's health; ``/status`` includes the channel name, the socket does not."""
    data: dict[str, Any] = {}
    if with_channel:
        data["channel"] = str(report.channel)
    data["state"] = _health(report.state)
    data["reasons"] = [str(r) for r in report.reasons]
    data["checks"] = {str(name): _health(state) for name, state in report.checks.items()}
    return data


def receipt(value: ReceiptLike) -> dict[str, Any]:
    """An entropy receipt (DESIGN.md 4.5) for REST, with ``frames: null`` when empty."""
    data = dict(value.to_dict())
    if not data.get("frames"):
        data["frames"] = None
    return data


def fish_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """A ``FishCard`` (DESIGN.md 7.1) restricted to the documented fields."""
    stats: Mapping[str, Any] = card.get("stats") or {}
    return {
        "fish_id": str(card.get("fish_id", "")),
        "name": str(card.get("name") or card.get("fish_id", "")),
        "color": _color(card.get("color")) or _FALLBACK_COLOR,
        "present": bool(card.get("present", False)),
        "track_id": _optional_int(card.get("track_id")),
        "stats": {
            "speed_avg": _finite(stats.get("speed_avg")),
            "speed_max": _finite(stats.get("speed_max")),
            "activity": min(1.0, max(0.0, _finite(stats.get("activity")))),
            "chaos_score": min(1.0, max(0.0, _finite(stats.get("chaos_score")))),
            "bits_total": _finite(stats.get("bits_total")),
            "bits_today": _finite(stats.get("bits_today")),
            "distance_px": _finite(stats.get("distance_px")),
            "seen_s": _finite(stats.get("seen_s")),
        },
    }


def leaderboard_entry(entry: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """``{rank, fish_id, name, chaos_score, bits}``; ``rank`` defaults to list position."""
    value = _optional_int(entry.get("rank"))
    return {
        "rank": value if value is not None else rank,
        "fish_id": str(entry.get("fish_id", "")),
        "name": str(entry.get("name") or entry.get("fish_id", "")),
        "chaos_score": min(1.0, max(0.0, _finite(entry.get("chaos_score")))),
        "bits": _finite(entry.get("bits")),
    }


# --------------------------------------------------------------------------
# WebSocket messages (DESIGN.md 7.2)
# --------------------------------------------------------------------------


def hello_message(
    *,
    version: str,
    source: str,
    width: int,
    height: int,
    fps: float,
    beacon_public_key: str | None,
) -> dict[str, Any]:
    """``hello``: the first message on every connection."""
    return {
        "version": version,
        "source": source,
        "width": int(width),
        "height": int(height),
        "fps": _finite(fps),
        "beacon_public_key": beacon_public_key,
    }


def _track(
    track: TrackSnapshot, identities: Mapping[str, tuple[str, str | None]]
) -> dict[str, Any]:
    vx, vy = (_finite(v) for v in track.velocity)
    name, color = identities.get(track.fish_id, (None, None)) if track.fish_id else (None, None)
    return {
        "track_id": int(track.track_id),
        "fish_id": track.fish_id,
        "name": name,
        "color": color,
        "bbox": [_round(v, _COORD_DIGITS) for v in track.bbox],
        "trail": [[_round(x, _COORD_DIGITS), _round(y, _COORD_DIGITS)] for x, y in track.trail],
        "speed": round(math.hypot(vx, vy), _COORD_DIGITS),
        "heading": round(math.atan2(vy, vx), 4),
    }


def frame_message(
    snapshot: SnapshotLike,
    jpeg: bytes | None,
    identities: Mapping[str, tuple[str, str | None]],
) -> dict[str, Any]:
    """``frame``: the (downscaled) JPEG plus confirmed tracks in *frame* pixel coordinates.

    ``width``/``height`` are the source frame's size, the coordinate space
    of ``bbox`` and ``trail``; the dashboard stretches the JPEG to it, so
    the stream can be smaller than the frame. Tentative tracks are left out:
    they are detector noise until the tracker confirms them.
    """
    return {
        "index": int(snapshot.frame_index),
        "t": _finite(snapshot.t),
        "width": int(snapshot.width),
        "height": int(snapshot.height),
        "jpeg": base64.b64encode(jpeg).decode("ascii") if jpeg else None,
        "tracks": [_track(t, identities) for t in snapshot.tracks if t.confirmed],
    }


def _channel_entropy(stats: ChannelStatsLike | None) -> dict[str, Any]:
    if stats is None:
        return {"bits_per_s": 0.0, "h_per_sample": None, "credited_total": 0.0, "health": "failed"}
    return {
        "bits_per_s": _round(stats.bits_per_s, _RATE_DIGITS),
        "h_per_sample": _optional_finite(stats.h_per_sample),
        "credited_total": _round(stats.credited_total, _RATE_DIGITS),
        "health": _health(stats.health),
    }


def entropy_message(snapshot: SnapshotLike, capacity_bits: float) -> dict[str, Any]:
    """``entropy``: per-channel credit rates and the undrawn pool.

    A channel the engine did not report is sent as failed with zero credit
    rather than omitted: the dashboard requires all three channels, and "not
    reported" must never look healthy.
    """
    channels = {name: _channel_entropy(snapshot.channels.get(name)) for name in PROTOCOL_CHANNELS}
    pool: dict[str, float] = {
        name: _round(snapshot.pool.get(name, 0.0), _RATE_DIGITS) for name in CAMERA_CHANNELS
    }
    return {
        "t": _finite(snapshot.t),
        "mode": _mode(snapshot.mode),
        "channels": channels,
        "pool": {**pool, "capacity_bits": _finite(capacity_bits)},
    }


def bits_message(t: float, bits: str) -> dict[str, Any]:
    """``bits``: public display bits (never key material)."""
    if not _BITS.fullmatch(bits):
        raise ValueError("display bits must be a string of 0/1")
    return {"t": _finite(t), "bits": bits}


def health_message(snapshot: SnapshotLike) -> dict[str, Any]:
    """``health``: overall state and every channel's report."""
    return {
        "overall": _health(snapshot.overall),
        "channels": {
            str(name): health_report(report, with_channel=False)
            for name, report in snapshot.health.items()
        },
    }


# --------------------------------------------------------------------------
# REST payloads (DESIGN.md 7.1)
# --------------------------------------------------------------------------


def status_message(snapshot: SnapshotLike, *, source: str, uptime_s: float) -> dict[str, Any]:
    """``GET /api/status``. ``frames`` counts frames assuming indices start at 0."""
    return {
        "source": source,
        "mode": _mode(snapshot.mode),
        "fps": _round(snapshot.fps, 2),
        "uptime_s": _round(uptime_s, 1),
        "frames": max(0, int(snapshot.frame_index) + 1),
        "pool": {str(k): _round(v, _RATE_DIGITS) for k, v in snapshot.pool.items()},
        "liveness": liveness_status(snapshot.liveness),
        "health": {
            str(name): health_report(report, with_channel=True)
            for name, report in snapshot.health.items()
        },
    }


def identities_from_labels(labels: object) -> dict[str, tuple[str, str | None]]:
    """``fish_id -> (name, color)`` from ``EngineSnapshot.fish`` (``{id: FishLabel}``).

    The engine attaches the labels known at frame time, so a newly named
    fish is labelled from its very first frame. Anything that does not look
    like ``{str: object with str name/color}`` is ignored.
    """
    out: dict[str, tuple[str, str | None]] = {}
    if not isinstance(labels, Mapping):
        return out
    for fish_id, label in labels.items():
        name = getattr(label, "name", None)
        if isinstance(fish_id, str) and isinstance(name, str):
            out[fish_id] = (name, _color(getattr(label, "color", None)))
    return out


def identities_from_cards(cards: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, str | None]]:
    """``fish_id -> (name, color)`` for labelling frame tracks."""
    out: dict[str, tuple[str, str | None]] = {}
    for card in cards:
        fish_id = card.get("fish_id")
        if isinstance(fish_id, str) and fish_id:
            out[fish_id] = (str(card.get("name") or fish_id), _color(card.get("color")))
    return out
