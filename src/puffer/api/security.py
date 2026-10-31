"""Response hardening: security headers, a hash-pinned CSP, and a request-body cap.

Both middlewares are plain ASGI (not ``BaseHTTPMiddleware``) so they add no
task hop per request and never buffer streamed responses.

Content-Security-Policy
    The dashboard is a static bundle from the same origin: scripts and styles
    from ``'self'``, frames decoded from base64 into ``blob:``/``data:``
    images, and one WebSocket. The only inline code is ``index.html``'s tiny
    pre-paint theme script; instead of ``'unsafe-inline'`` the policy pins
    the SHA-256 of each inline script found in the served ``index.html``
    (CSP Level 2 hash sources), so an injected script cannot run even if
    some future bug reflected markup into the page. ``style-src`` needs no
    ``'unsafe-inline'``: React applies ``style`` props through the CSSOM,
    which CSP does not restrict (verified against the built dashboard).

Cache-Control
    Every ``/api`` response defaults to ``no-store``: ``/random`` and the
    encrypt demo return secrets, and nothing there is worth a stale copy.
    Static assets set their own policy (see :mod:`puffer.api.static`).
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = [
    "BodySizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "content_security_policy",
    "index_script_hashes",
    "inline_script_hashes",
]

_INLINE_SCRIPT = re.compile(
    r"<script(?![^>]*\bsrc\s*=)[^>]*>(?P<body>.*?)</script\s*>", re.IGNORECASE | re.DOTALL
)


def inline_script_hashes(html: str) -> list[str]:
    """CSP hash sources (``'sha256-...'``) for every inline ``<script>`` in ``html``.

    The browser hashes the exact text between the tags, so nothing is
    stripped or normalised here.
    """
    hashes = []
    for match in _INLINE_SCRIPT.finditer(html):
        digest = hashlib.sha256(match.group("body").encode("utf-8")).digest()
        source = f"'sha256-{base64.b64encode(digest).decode('ascii')}'"
        if source not in hashes:
            hashes.append(source)
    return hashes


def index_script_hashes(static_dir: Path | None) -> list[str]:
    """Hashes for ``static_dir/index.html``, or none if there is no built dashboard."""
    if static_dir is None:
        return []
    index = static_dir / "index.html"
    try:
        return inline_script_hashes(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return []


def content_security_policy(
    script_hashes: Sequence[str] = (), *, inline_styles: bool = False
) -> str:
    """The policy for every response; see the module docstring for each source."""
    directives = {
        "default-src": ["'self'"],
        "script-src": ["'self'", *script_hashes],
        "style-src": ["'self'", *(["'unsafe-inline'"] if inline_styles else [])],
        "img-src": ["'self'", "data:", "blob:"],
        "font-src": ["'self'"],
        "connect-src": ["'self'", "ws:", "wss:"],
        "object-src": ["'none'"],
        "base-uri": ["'none'"],
        "form-action": ["'self'"],
        "frame-ancestors": ["'none'"],
    }
    return "; ".join(f"{name} {' '.join(sources)}" for name, sources in directives.items())


def _security_headers(csp: str) -> list[tuple[bytes, bytes]]:
    pairs = {
        "content-security-policy": csp,
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "x-frame-options": "DENY",
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-resource-policy": "same-origin",
        "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    }
    return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in pairs.items()]


class SecurityHeadersMiddleware:
    """Adds the security headers to every HTTP response (never overriding a route's own)."""

    def __init__(self, app: ASGIApp, *, csp: str, no_store_prefixes: Iterable[str] = ("/api",)):
        self.app = app
        self._headers = _security_headers(csp)
        self._no_store = tuple(no_store_prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        no_store = path.startswith(self._no_store)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers.extend(pair for pair in self._headers if pair[0] not in present)
                if no_store and b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Rejects request bodies over ``max_bytes`` with 413 before they are parsed.

    A declared ``Content-Length`` is checked up front; chunked bodies are
    counted as they stream, so a client cannot make the JSON parser buffer
    an unbounded body by omitting the header. The streaming case raises an
    ``HTTPException``, which FastAPI's body reader re-raises as is and the
    exception middleware renders as ``{"detail": ...}``.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    @property
    def detail(self) -> str:
        return f"request body is limited to {self.max_bytes} bytes"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            response = JSONResponse({"detail": self.detail}, status_code=413)
            await response(scope, receive, send)
            return
        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail=self.detail)
            return message

        await self.app(scope, counting_receive, send)
