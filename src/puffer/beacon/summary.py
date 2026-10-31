"""Turn an entropy receipt into the public beacon summary (DESIGN.md 3.4 fields).

The beacon's entropy source returns ``(bytes, summary)``. In production the
bytes come from ``PufferRNG.generate(64, "beacon")`` together with an
``EntropyReceipt`` whose ``to_dict()`` (DESIGN.md 4.5) carries float bit
counts and shares. The signed summary may contain integers only, so this
module maps the receipt to the documented summary fields and converts every
bit amount with ``floor``: the public record must never claim more entropy
than was credited ("credit is earned, not assumed").

Only facts that are safe to publish are copied: aggregate bit counts, health
states, liveness state, frame range and per-fish credit - never pool
material or anything derived from the secret output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from puffer.types import CHANNEL_FISH, CHANNEL_SENSOR, CHANNEL_SYSTEM

DEFAULT_TOP_FISH: Final = 3
"""How many fish the summary lists, by credited bits."""


def summary_from_receipt(
    receipt: Mapping[str, Any], *, top_n: int = DEFAULT_TOP_FISH
) -> dict[str, Any]:
    """Build the beacon summary dict from an ``EntropyReceipt.to_dict()`` mapping.

    Fields: ``fish_count`` (fish credited in this draw), ``fish_bits``,
    ``sensor_bits``, ``system_bits``, ``mode``, ``liveness``,
    ``liveness_confidence_bits``, ``health`` ({channel: state}), ``top_fish``
    ([{fish_id, name, bits}] by bits descending, ties by fish_id) and
    ``frames`` ([first, last], empty when no camera frame contributed).
    The result is ready for :func:`~puffer.beacon.pulse.canonical_summary`.

    Raises:
        KeyError: a required receipt field is missing.
        ValueError: a bit amount is negative or not finite, or ``top_n < 0``.
    """
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    channels: Mapping[str, Mapping[str, Any]] = receipt["channels"]
    liveness: Mapping[str, Any] = receipt["liveness"]
    fish: Sequence[Mapping[str, Any]] = receipt.get("fish") or ()
    frames = receipt.get("frames") or ()

    ranked = sorted(fish, key=lambda f: (-float(f["bits"]), str(f["fish_id"])))
    return {
        "fish_count": len(fish),
        "fish_bits": _channel_bits(channels, CHANNEL_FISH),
        "sensor_bits": _channel_bits(channels, CHANNEL_SENSOR),
        "system_bits": _channel_bits(channels, CHANNEL_SYSTEM),
        "mode": str(receipt["mode"]),
        "liveness": str(liveness["state"]),
        "liveness_confidence_bits": floor_bits(liveness.get("confidence_bits", 0)),
        "health": {str(name): str(channel["health"]) for name, channel in channels.items()},
        "top_fish": [
            {"fish_id": str(f["fish_id"]), "name": str(f["name"]), "bits": floor_bits(f["bits"])}
            for f in ranked[:top_n]
        ],
        "frames": [int(frame) for frame in frames],
    }


def floor_bits(value: object) -> int:
    """Round a credited bit amount *down* to an integer.

    Raises:
        ValueError: the value is negative, NaN or infinite.
        TypeError: the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"bit amount must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"bit amount must be finite and non-negative, got {value!r}")
    return math.floor(value)


def _channel_bits(channels: Mapping[str, Mapping[str, Any]], name: str) -> int:
    channel = channels.get(name)
    return 0 if channel is None else floor_bits(channel["credited_bits"])
