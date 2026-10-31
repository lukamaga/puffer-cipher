"""The FastAPI application: REST (DESIGN.md 7.1), ``/ws/live`` (7.2) and the dashboard.

:func:`create_app` wires an engine (anything satisfying
:class:`~puffer.api.ports.EngineLike`) and, optionally, the beacon into one
ASGI app. It starts nothing by itself except the live hub, whose producer
tasks live and die with the app's lifespan; running the engine and the beacon
thread is :func:`puffer.api.server.serve`'s job, which keeps this module
importable and testable without a camera.

Where secret bytes come from
    Only from ``engine.rng``: ``/random`` uses ``rng.generate`` (it consumes
    pool credit, so it is rate limited), the demo key ``rng.generate_key``
    and the demo nonce ``rng.nonce`` (public, so it spends no camera credit
    where PufferRNG supports that). Secrets are never logged; responses
    that carry them are ``Cache-Control: no-store``.

Errors
    Every error body is ``{"detail": ...}`` (FastAPI's convention, which the
    dashboard's REST client parses): 400 for undecryptable demo input, 404
    for unknown pulses or a disabled beacon, 413 for oversized bodies, 422
    for invalid parameters, 429 (with ``Retry-After``) when rate limited and
    503 while the engine has not produced its first frame.
"""

# No ``from __future__ import annotations`` here: FastAPI reads the route
# signatures at runtime, and the limits inside their ``Annotated[...]`` refer
# to ``settings``, a local of ``create_app`` that string annotations could not
# resolve.

import base64
import logging
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import puffer
from puffer.api import wire
from puffer.api.hub import LiveHub
from puffer.api.ports import EngineLike, RngLike
from puffer.api.ratelimit import TokenBucketLimiter
from puffer.api.security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    content_security_policy,
    index_script_hashes,
)
from puffer.api.settings import ApiSettings
from puffer.api.static import DashboardFiles, dashboard_available
from puffer.beacon.pulse import MAX_JSON_INT
from puffer.beacon.service import BeaconService
from puffer.beacon.store import BeaconStore
from puffer.crypto import (
    KEY_BYTES,
    NONCE_BYTES,
    AuthenticationError,
    FormatError,
    PufferCryptoError,
    b64url_decode,
    b64url_encode,
    decrypt_message,
    encrypt_message_b64,
    key_id,
)
from puffer.crypto.aead import TAG_BYTES

logger = logging.getLogger(__name__)

__all__ = ["create_app"]

_RANDOM_PURPOSE = "random"
_DASHBOARD_HINT = "cd dashboard && npm ci && npm run build"


class EncryptRequest(BaseModel):
    """``POST /api/demo/encrypt`` body."""

    plaintext: str


class DecryptRequest(BaseModel):
    """``POST /api/demo/decrypt`` body (both fields base64url, as returned by encrypt)."""

    ciphertext: str
    key: str = Field(max_length=256)


@dataclass
class _Beacon:
    """The beacon as the API sees it: an optional producer and an optional store.

    With a service the store defaults to the service's; a store alone is a
    read-only mirror whose key is the one its chain was signed with.
    """

    service: BeaconService | None
    store: BeaconStore | None
    _store_key: str | None = field(default=None, init=False)

    def public_key_hex(self) -> str | None:
        if self.service is not None:
            return self.service.public_key.hex()
        if self._store_key is None and self.store is not None:
            latest = self.store.latest()  # a store's chain key never changes, so cache it
            self._store_key = latest.public_key.hex() if latest is not None else None
        return self._store_key

    def require_store(self) -> BeaconStore:
        if self.store is None:
            raise HTTPException(status_code=404, detail="the beacon is disabled")
        return self.store


def _client_key(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _rate_limit(limiter: TokenBucketLimiter) -> Callable[[Request], Awaitable[None]]:
    """A dependency charging one token per request to the caller's bucket."""

    async def dependency(request: Request) -> None:
        verdict = limiter.acquire(_client_key(request))
        if verdict is not None:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded: please slow down",
                headers={"Retry-After": verdict.retry_after_header},
            )

    return dependency


def _public_nonce(rng: RngLike) -> Callable[[int], bytes]:
    """``random_bytes`` hook for the AEAD nonce, backed by ``rng.nonce()``.

    A nonce is public and only has to be unique, so it should not spend the
    camera credit that the key next to it is supposed to carry (PufferRNG's
    ``nonce`` is ``consume=False`` where available).
    """

    def nonce(n: int) -> bytes:
        if n != NONCE_BYTES:
            raise ValueError(f"expected a {NONCE_BYTES}-byte nonce request, got {n}")
        return rng.nonce()

    return nonce


def _decrypt_demo(ciphertext: str, key_text: str) -> str:
    """Decrypt a demo message, raising ``HTTPException(400)`` with a safe explanation.

    Messages say *which input* is malformed but, for authentication
    failures, never more than libsodium does: tampering and a wrong key are
    indistinguishable by design.
    """
    try:
        key = b64url_decode(key_text)
    except FormatError:
        raise HTTPException(400, "key is not valid base64url") from None
    if len(key) != KEY_BYTES:
        raise HTTPException(400, f"key must be {KEY_BYTES} bytes")
    try:
        blob = b64url_decode(ciphertext)
    except FormatError:
        raise HTTPException(400, "ciphertext is not valid base64url") from None
    try:
        plaintext = decrypt_message(blob, key)
    except AuthenticationError:
        raise HTTPException(
            400, "authentication failed: the ciphertext was modified or the key is wrong"
        ) from None
    except (PufferCryptoError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"ciphertext rejected: {exc}") from None
    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "the decrypted message is not UTF-8 text") from None


def _json_text(text: str, status_code: int = 200) -> Response:
    """Pre-encoded JSON (beacon pulses: byte-identical to the WebSocket push)."""
    return Response(content=text, status_code=status_code, media_type="application/json")


def create_app(
    engine: EngineLike,
    *,
    beacon_service: BeaconService | None = None,
    beacon_store: BeaconStore | None = None,
    static_dir: Path | None = None,
    settings: ApiSettings | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Build the puffer API around ``engine`` (DESIGN.md 7).

    Args:
        engine: the entropy engine (or a fake satisfying ``EngineLike``).
        beacon_service: the running beacon; its pulses are pushed to
            ``/ws/live`` and its key is pinned in ``hello``.
        beacon_store: where pulses are read from; defaults to the service's.
            ``None`` for both disables the beacon endpoints (404).
        static_dir: the built dashboard (``dashboard/dist``), served at ``/``
            after the API routes; when missing, ``/`` returns a JSON hint.
        settings: rates and limits; default: derived from ``engine.config``.
        clock: monotonic clock for uptime, history and rate limits (tests).
    """
    settings = settings if settings is not None else ApiSettings.from_config(engine.config)
    static_dir = Path(static_dir) if static_dir is not None else None
    beacon = _Beacon(
        beacon_service,
        beacon_store if beacon_store is not None else getattr(beacon_service, "store", None),
    )
    hub = LiveHub(
        engine,
        settings,
        version=puffer.__version__,
        beacon_public_key=beacon.public_key_hex,
        clock=clock,
    )
    started = clock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await hub.start()
        unsubscribe = beacon_service.on_pulse(hub.publish_pulse) if beacon_service else None
        try:
            yield
        finally:
            if unsubscribe is not None:
                unsubscribe()
            await hub.stop()

    app = FastAPI(
        title="puffer-cipher",
        version=puffer.__version__,
        summary="Cryptographic randomness from live fish: REST + live WebSocket feed.",
        docs_url=None,  # Swagger UI loads scripts from a CDN, which the CSP forbids
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.hub = hub
    app.state.settings = settings

    def limiter() -> TokenBucketLimiter:
        return TokenBucketLimiter(
            settings.rate_per_s, settings.rate_burst, max_clients=settings.rate_clients, clock=clock
        )

    limit_random = Depends(_rate_limit(limiter()))
    limit_demo = Depends(_rate_limit(limiter()))
    max_ciphertext_chars = 4 * math.ceil(
        (NONCE_BYTES + settings.max_plaintext_bytes + TAG_BYTES) / 3
    )

    api = APIRouter(prefix="/api")

    # -- status, randomness, demo ----------------------------------------------------

    @api.get("/status", response_model=None)
    def get_status() -> dict[str, Any]:
        """Engine status: source, frames, pool credit, liveness and per-channel health."""
        snapshot = engine.snapshot()
        if snapshot is None:
            raise HTTPException(
                503, "the engine has not processed a frame yet", headers={"Retry-After": "1"}
            )
        return wire.status_message(snapshot, source=hub.source_label(), uptime_s=clock() - started)

    @api.get("/random", response_model=None, dependencies=[limit_random])
    def get_random(
        n: Annotated[int, Query(ge=1, le=settings.max_random_bytes)] = 32,
        fmt: Annotated[Literal["hex", "base64"], Query(alias="format")] = "hex",
    ) -> dict[str, Any]:
        """``n`` fresh random bytes from PufferRNG with their entropy receipt.

        ``base64`` is standard RFC 4648 base64 with padding.
        """
        data, receipt = engine.rng.generate(n, _RANDOM_PURPOSE)
        text = data.hex() if fmt == "hex" else base64.b64encode(data).decode("ascii")
        return {"data": text, "receipt": wire.receipt(receipt)}

    @api.post("/demo/encrypt", response_model=None, dependencies=[limit_demo])
    def demo_encrypt(body: EncryptRequest) -> dict[str, Any]:
        """DEMO ONLY: encrypt with a fresh fish key and hand the key to the caller."""
        try:
            plaintext = body.plaintext.encode("utf-8")
        except UnicodeEncodeError:
            raise HTTPException(422, "plaintext must be valid Unicode text") from None
        if len(plaintext) > settings.max_plaintext_bytes:
            raise HTTPException(
                413, f"plaintext is limited to {settings.max_plaintext_bytes} bytes (UTF-8)"
            )
        key, receipt = engine.rng.generate_key()
        ciphertext = encrypt_message_b64(plaintext, key, random_bytes=_public_nonce(engine.rng))
        return {
            "ciphertext": ciphertext,
            "key": b64url_encode(key),
            "key_id": key_id(key),
            "receipt": wire.receipt(receipt),
        }

    @api.post("/demo/decrypt", response_model=None, dependencies=[limit_demo])
    def demo_decrypt(body: DecryptRequest) -> dict[str, str]:
        """Decrypt a demo message; 400 with an explanation if it does not authenticate."""
        if len(body.ciphertext) > max_ciphertext_chars:
            raise HTTPException(413, "ciphertext is larger than any demo message")
        return {"plaintext": _decrypt_demo(body.ciphertext, body.key)}

    # -- fish ---------------------------------------------------------------------------

    @api.get("/fish", response_model=None)
    def get_fish() -> list[dict[str, Any]]:
        """Every known fish with its live stats (``FishCard``)."""
        return [wire.fish_card(card) for card in engine.fish_cards()]

    @api.get("/leaderboard", response_model=None)
    def get_leaderboard(
        days: Annotated[int, Query(ge=1, le=365)] = 7,
    ) -> list[dict[str, Any]]:
        """Fish ranked by mean chaos score over the last ``days`` days."""
        return [
            wire.leaderboard_entry(entry, rank)
            for rank, entry in enumerate(engine.leaderboard(days), start=1)
        ]

    # -- entropy history ------------------------------------------------------------------

    @api.get("/entropy/history", response_model=None)
    async def get_entropy_history(
        seconds: Annotated[int, Query(ge=1, le=int(settings.history_max_s))] = 300,
    ) -> list[dict[str, Any]]:
        """``entropy`` message data of the last ``seconds``, ascending (server ring buffer)."""
        return hub.history(seconds)

    # -- beacon -------------------------------------------------------------------------

    @api.get("/beacon/public-key", response_model=None)
    def get_beacon_public_key() -> dict[str, str]:
        """The Ed25519 key that signs the pulse chain."""
        key = beacon.public_key_hex()
        if key is None:
            raise HTTPException(404, "the beacon is disabled or has no pulses yet")
        return {"algorithm": "Ed25519", "public_key": key}

    @api.get("/beacon/latest", response_model=None)
    def get_beacon_latest() -> Response:
        """The newest pulse (404 before the first one)."""
        pulse = beacon.require_store().latest()
        if pulse is None:
            raise HTTPException(404, "no pulses yet")
        return _json_text(wire.pulse_text(pulse))

    @api.get("/beacon/pulses", response_model=None)
    def get_beacon_pulses(
        start: Annotated[int, Query(ge=0, le=MAX_JSON_INT)] = 0,
        limit: Annotated[int, Query(ge=1, le=settings.max_pulses)] = 50,
    ) -> Response:
        """Up to ``limit`` pulses with ``index >= start``, ascending."""
        return _json_text(wire.pulses_text(beacon.require_store().range(start, limit)))

    @api.get("/beacon/pulse/{index}", response_model=None)
    def get_beacon_pulse(index: Annotated[int, PathParam(ge=0, le=MAX_JSON_INT)]) -> Response:
        """One pulse by index."""
        pulse = beacon.require_store().get(index)
        if pulse is None:
            raise HTTPException(404, f"no pulse with index {index}")
        return _json_text(wire.pulse_text(pulse))

    app.include_router(api)

    # -- live feed ------------------------------------------------------------------------

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        await hub.serve(websocket)

    # -- dashboard (mounted last so it never shadows a route) ------------------------------

    if static_dir is not None and dashboard_available(static_dir):
        app.mount("/", DashboardFiles(static_dir), name="dashboard")
    else:
        if static_dir is not None:
            logger.warning("no built dashboard in %s; serving the API only", static_dir)

        @app.get("/", include_in_schema=False)
        def dashboard_missing() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": "the dashboard is not built",
                    "hint": _DASHBOARD_HINT,
                    "api": "/api/status",
                    "live": "/ws/live",
                }
            )

    csp = content_security_policy(
        index_script_hashes(static_dir if dashboard_available(static_dir) else None),
        inline_styles=settings.csp_inline_styles,
    )
    # Middleware added last runs first: headers go on everything, including 413s.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware, csp=csp)
    return app
