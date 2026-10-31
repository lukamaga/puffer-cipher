"""Entropy receipts: an honest, human-readable account of one RNG output.

Every :meth:`PufferRNG.generate <puffer.entropy.rng.PufferRNG.generate>`
call returns a receipt next to its bytes. It records which channels were
credited how many bits, how healthy they were, whether the LED liveness
check passed, and which fish contributed - the data behind the dashboard's
"this key was 31% Nemo" story. Receipts are never secret: they contain no
pool material, only accounting.

What a receipt does *not* claim: ``credited_bits`` is conservatively
estimated entropy *absorbed into the pool* since the previous draw. The
output itself comes from a 256-bit seed keyed with 256 bits of fresh OS
randomness, so no output ever holds more than 256 bits of entropy however
large the camera credit, and never less than what ``os.urandom`` provides
however small it is.

``to_dict`` follows DESIGN.md section 4.5 exactly; ``summary_for_beacon``
produces the integer-only summary embedded in beacon pulses (section 3.4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from puffer.entropy.pool import PoolDraw
from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    HealthState,
    LivenessState,
    LivenessStatus,
)

__all__ = [
    "RECEIPT_VERSION",
    "SYSTEM_BITS",
    "ChannelCredit",
    "EntropyReceipt",
    "FishCredit",
    "ReceiptContext",
]

RECEIPT_VERSION = 1
SYSTEM_BITS = 256.0
"""Every output is keyed with 32 fresh bytes from ``os.urandom``."""

TOP_FISH = 3
"""Number of fish listed in a beacon summary."""

_BITS_DIGITS = 3
_SHARE_DIGITS = 4


def _bits(value: float) -> float:
    """Sanitise an entropy amount: anything non-finite or negative claims nothing."""
    return value if math.isfinite(value) and value > 0 else 0.0


def _floor(value: float) -> int:
    return math.floor(_bits(value))


@dataclass(frozen=True, slots=True)
class ReceiptContext:
    """Engine state at the moment of a draw, supplied via a callback.

    ``mode`` is the light mode (``"day"``/``"night"``) or ``"unknown"``;
    ``health`` maps channel names to their state; ``fish_names`` maps fish
    ids to display names.
    """

    mode: str = "unknown"
    liveness: LivenessStatus = field(default_factory=lambda: LivenessStatus(LivenessState.DISABLED))
    health: dict[str, HealthState] = field(default_factory=dict)
    fish_names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChannelCredit:
    """One channel's line in a receipt.

    ``share`` is the fraction of the camera-channel total (``None`` for the
    system channel, and for every channel when no camera credit exists);
    ``health`` is ``None`` when the engine did not report it.
    """

    credited_bits: float
    share: float | None
    health: HealthState | None


@dataclass(frozen=True, slots=True)
class FishCredit:
    """One fish's contribution; ``share`` is its fraction of the fish channel's credit."""

    fish_id: str
    name: str
    bits: float
    share: float


@dataclass(frozen=True, slots=True)
class EntropyReceipt:
    """Accounting for one RNG output (see the module docstring for its meaning)."""

    purpose: str
    n_bytes: int
    timestamp_ms: int
    frames: tuple[int, int] | None
    mode: str
    liveness_state: LivenessState
    liveness_confidence_bits: float
    channels: dict[str, ChannelCredit]
    fish: tuple[FishCredit, ...] = ()

    @classmethod
    def from_draw(
        cls,
        draw: PoolDraw,
        *,
        purpose: str,
        n_bytes: int,
        timestamp_ms: int,
        context: ReceiptContext,
    ) -> EntropyReceipt:
        """Build the receipt for an output produced from ``draw``."""
        names = list(CAMERA_CHANNELS) + sorted(set(draw.credits) - set(CAMERA_CHANNELS))
        bits = {name: _bits(draw.credits.get(name, 0.0)) for name in names}
        camera_total = sum(bits.values())
        channels = {
            name: ChannelCredit(
                credited_bits=bits[name],
                share=bits[name] / camera_total if camera_total > 0 else None,
                health=context.health.get(name),
            )
            for name in names
        }
        channels[CHANNEL_SYSTEM] = ChannelCredit(SYSTEM_BITS, None, HealthState.OK)
        fish_total = bits.get(CHANNEL_FISH, 0.0)
        fish = tuple(
            sorted(
                (
                    FishCredit(
                        fish_id=fish_id,
                        name=context.fish_names.get(fish_id, fish_id),
                        bits=_bits(value),
                        share=min(1.0, _bits(value) / fish_total) if fish_total > 0 else 0.0,
                    )
                    for fish_id, value in draw.fish_credits.items()
                    if _bits(value) > 0
                ),
                key=lambda f: (-f.bits, f.fish_id),
            )
        )
        return cls(
            purpose=purpose,
            n_bytes=n_bytes,
            timestamp_ms=timestamp_ms,
            frames=draw.frames,
            mode=str(context.mode),
            liveness_state=context.liveness.state,
            liveness_confidence_bits=_bits(context.liveness.confidence_bits),
            channels=channels,
            fish=fish,
        )

    @property
    def camera_bits(self) -> float:
        """Credited bits from the camera channels."""
        return sum(c.credited_bits for name, c in self.channels.items() if name != CHANNEL_SYSTEM)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict in the DESIGN.md section 4.5 shape (``frames`` is ``[]`` if none)."""
        return {
            "receipt_version": RECEIPT_VERSION,
            "purpose": self.purpose,
            "bytes": self.n_bytes,
            "timestamp_ms": self.timestamp_ms,
            "frames": list(self.frames) if self.frames is not None else [],
            "mode": self.mode,
            "liveness": {
                "state": self.liveness_state.value,
                "confidence_bits": round(self.liveness_confidence_bits, _BITS_DIGITS),
            },
            "channels": {
                name: {
                    "credited_bits": round(c.credited_bits, _BITS_DIGITS),
                    "share": round(c.share, _SHARE_DIGITS) if c.share is not None else None,
                    "health": c.health.value if c.health is not None else None,
                }
                for name, c in self.channels.items()
            },
            "fish": [
                {
                    "fish_id": f.fish_id,
                    "name": f.name,
                    "bits": round(f.bits, _BITS_DIGITS),
                    "share": round(f.share, _SHARE_DIGITS),
                }
                for f in self.fish
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EntropyReceipt:
        """Parse :meth:`to_dict` output (e.g. the ``receipt`` of a key file)."""
        version = data.get("receipt_version")
        if version != RECEIPT_VERSION:
            raise ValueError(f"unsupported receipt_version {version!r}")
        frames = data["frames"]
        if frames and len(frames) != 2:
            raise ValueError("frames must be [] or [first, last]")
        liveness = data["liveness"]
        return cls(
            purpose=str(data["purpose"]),
            n_bytes=int(data["bytes"]),
            timestamp_ms=int(data["timestamp_ms"]),
            frames=(int(frames[0]), int(frames[1])) if frames else None,
            mode=str(data["mode"]),
            liveness_state=LivenessState(liveness["state"]),
            liveness_confidence_bits=float(liveness["confidence_bits"]),
            channels={
                str(name): ChannelCredit(
                    credited_bits=float(c["credited_bits"]),
                    share=float(c["share"]) if c["share"] is not None else None,
                    health=HealthState(c["health"]) if c["health"] is not None else None,
                )
                for name, c in data["channels"].items()
            },
            fish=tuple(
                FishCredit(
                    fish_id=str(f["fish_id"]),
                    name=str(f["name"]),
                    bits=float(f["bits"]),
                    share=float(f["share"]),
                )
                for f in data["fish"]
            ),
        )

    def summary_for_beacon(self) -> dict[str, Any]:
        """Integer-only summary for beacon pulses (DESIGN.md section 3.4).

        Only ints, strings, bools, lists and dicts: the summary is hashed as
        canonical JSON and verified in the browser, where float formatting
        differs between languages. Bit amounts are rounded *down*, so a
        summary never overstates entropy.
        """

        def channel_bits(name: str) -> int:
            credit = self.channels.get(name)
            return _floor(credit.credited_bits) if credit is not None else 0

        return {
            "fish_count": len(self.fish),
            "fish_bits": channel_bits(CHANNEL_FISH),
            "sensor_bits": channel_bits(CHANNEL_SENSOR),
            "system_bits": channel_bits(CHANNEL_SYSTEM),
            "mode": self.mode,
            "liveness": self.liveness_state.value,
            "liveness_confidence_bits": _floor(self.liveness_confidence_bits),
            "health": {
                name: c.health.value for name, c in self.channels.items() if c.health is not None
            },
            "top_fish": [
                {"fish_id": f.fish_id, "name": f.name, "bits": _floor(f.bits)}
                for f in self.fish[:TOP_FISH]
            ],
            "frames": list(self.frames) if self.frames is not None else [],
        }
