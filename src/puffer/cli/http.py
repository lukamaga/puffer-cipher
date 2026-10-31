"""A small, strict JSON-over-HTTP client for ``puffer beacon`` (stdlib only).

Beacon pulses come from servers the user may not control, so fetching them
is treated like parsing any other untrusted input:

* only ``http`` and ``https`` URLs are accepted (``urllib`` would also open
  ``file://`` and ``ftp://``);
* responses are capped at :data:`MAX_RESPONSE_BYTES`, so a hostile server
  cannot exhaust memory;
* JSON objects that repeat a key are refused, because parsers disagree on
  which copy wins and the verdict must be about the same bytes every
  verifier sees (the same rule as :func:`puffer.beacon.chain.verify_file`).

Proxies from the environment (``https_proxy``/``no_proxy``) are honoured,
as users of ``curl`` expect.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from puffer import __version__
from puffer.cli.errors import CliError, ExitCode

__all__ = ["MAX_RESPONSE_BYTES", "HttpStatusError", "check_url", "get_json", "join_url"]

MAX_RESPONSE_BYTES = 32 << 20
DEFAULT_TIMEOUT_S = 10.0


class HttpStatusError(CliError):
    """The server answered with a non-2xx status; ``status`` lets callers special-case 404."""

    def __init__(self, url: str, status: int, reason: str) -> None:
        super().__init__(f"{url}: HTTP {status} {reason}".rstrip(), ExitCode.IO)
        self.status = status


def check_url(url: str) -> urllib.parse.SplitResult:
    """Parse ``url`` and accept only absolute ``http(s)://host`` URLs."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise CliError(f"not an http(s) URL: {url!r}", ExitCode.USAGE)
    return parts


def join_url(base: str, path: str, **query: int | str) -> str:
    """``base`` (scheme://host[:port][/prefix]) + ``path`` + ``?query``."""
    parts = check_url(base)
    prefix = parts.path.rstrip("/")
    encoded = urllib.parse.urlencode(query) if query else ""
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, prefix + path, encoded, ""))


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def parse_json(body: bytes, origin: str) -> Any:
    """Strictly parse a JSON document (UTF-8, no duplicate keys, no NaN)."""

    def reject(name: str) -> Any:
        raise ValueError(f"non-standard JSON constant {name}")

    try:
        return json.loads(
            body.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=reject
        )
    except (ValueError, RecursionError) as exc:
        raise CliError(f"{origin}: not a valid JSON document: {exc}", ExitCode.FAILURE) from exc


def get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Any:
    """GET ``url`` and return its parsed JSON body.

    Raises:
        HttpStatusError: non-2xx response (exit code IO).
        CliError: bad URL (USAGE), network failure or oversized body (IO),
            invalid JSON (FAILURE).
    """
    check_url(url)
    request = urllib.request.Request(  # noqa: S310 - scheme restricted to http(s) above
        url,
        headers={"Accept": "application/json", "User-Agent": f"puffer-cipher/{__version__}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise HttpStatusError(url, exc.code, str(exc.reason or "")) from None
    except urllib.error.URLError as exc:
        raise CliError(f"cannot reach {url}: {exc.reason}", ExitCode.IO) from None
    except (TimeoutError, OSError) as exc:
        raise CliError(f"cannot reach {url}: {exc}", ExitCode.IO) from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise CliError(f"{url}: response larger than {MAX_RESPONSE_BYTES >> 20} MiB", ExitCode.IO)
    return parse_json(body, url)
