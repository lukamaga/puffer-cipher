"""Tunables of the HTTP/WebSocket server, with the DESIGN.md 7 defaults.

Everything that bounds work or memory per client lives here, so the abuse
limits are visible in one place and tests can shrink or speed them up (a
test that needs a ``fish`` message should not wait five seconds for it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

from puffer.api.ports import ConfigLike

__all__ = ["ApiSettings"]

# Dashboard dev servers (Vite `npm run dev` / `npm run preview`, see
# dashboard/README.md). Their proxy rewrites Host but forwards Origin.
_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Server settings; :meth:`from_config` maps ``PufferConfig`` onto them.

    Rates are in messages per second (DESIGN.md 7.2). ``frame_hz`` is an
    upper bound: a frame is only sent when the engine produced a new one.
    """

    # -- live stream (DESIGN 7.2 rates) ------------------------------------
    frame_hz: float = 10.0
    entropy_hz: float = 2.0
    bits_hz: float = 5.0
    health_hz: float = 1.0
    led_hz: float = 2.0
    fish_hz: float = 0.2
    stream_width: int = 640
    jpeg_quality: int = 70
    display_bits: int = 256
    pool_capacity_bits: float = 4096.0

    # -- per-client WebSocket limits ----------------------------------------
    max_clients: int = 64
    client_queue: int = 128
    """Ordered (non-frame) messages buffered per client before it is dropped as too slow."""
    hello_wait_s: float = 2.0
    """How long a new client waits for the engine's first frame before ``hello`` is sent anyway."""
    handshake_timeout_s: float = 15.0
    """Time a new client gets to read ``hello`` and the replayed history (~300 KB)."""
    replay_entropy_s: float = 300.0
    """``entropy`` history replayed to each new client so charts start populated (0 = off)."""
    allowed_origins: tuple[str, ...] = _DEV_ORIGINS
    """Cross-origin WebSocket origins accepted in addition to same-origin requests."""

    # -- REST limits ---------------------------------------------------------
    history_max_s: float = 600.0
    max_random_bytes: int = 1024
    max_plaintext_bytes: int = 16 * 1024
    max_pulses: int = 1000
    max_body_bytes: int = 128 * 1024
    """Request body cap: a 16 KiB plaintext of control characters is ~96 KiB of JSON escapes."""
    rate_per_s: float = 2.0
    """Sustained requests per second per client on ``/api/random`` and ``/api/demo/*``."""
    rate_burst: float = 20.0
    """Token-bucket size: requests a client may make back to back."""
    rate_clients: int = 4096
    """Distinct clients tracked by the rate limiter (least recently seen are forgotten)."""

    # -- static dashboard ----------------------------------------------------
    csp_inline_styles: bool = False
    """Add ``'unsafe-inline'`` to ``style-src`` (the built dashboard does not need it)."""

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("allowed_origins", "csp_inline_styles"):
                continue
            if f.name == "replay_entropy_s":
                ok = math.isfinite(value) and value >= 0
            else:
                ok = math.isfinite(value) and value > 0
            if not ok:
                raise ValueError(f"ApiSettings.{f.name} must be a positive finite number")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in 1..100")
        if self.replay_entropy_s > self.history_max_s:
            raise ValueError("replay_entropy_s cannot exceed history_max_s")

    @classmethod
    def from_config(cls, config: ConfigLike, **overrides: Any) -> ApiSettings:
        """Settings for a ``PufferConfig`` (stream rate, size, quality, pool capacity)."""
        base = cls(
            frame_hz=float(config.stream_fps),
            stream_width=int(config.stream_width),
            jpeg_quality=int(config.jpeg_quality),
            pool_capacity_bits=float(config.pool_capacity_bits),
        )
        return replace(base, **overrides)
