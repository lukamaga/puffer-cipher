"""Strict, canonical base64url (RFC 4648 section 5) without padding.

Keys, ciphertexts and signing seeds travel through JSON, URLs and the
browser, so every textual binary value in puffer uses the URL-safe
alphabet and omits ``=`` padding.

The decoder is deliberately stricter than :func:`base64.urlsafe_b64decode`:

* only ``A-Z a-z 0-9 - _`` are accepted (no whitespace, no ``+`` or ``/``
  that the standard library would silently translate or skip);
* padding is optional, but if present it must be exactly right;
* the encoding must be *canonical*: the unused low bits of the last
  character must be zero. Without this check several distinct strings
  decode to the same bytes, a malleability that has caused signature and
  cache-key confusion bugs elsewhere. Here ``encode(decode(s)) == s``
  (modulo padding) always holds.
"""

from __future__ import annotations

import base64
import binascii
import re

from puffer.crypto.errors import FormatError

_B64URL = re.compile(r"[A-Za-z0-9_-]*(={0,2})")


def b64url_encode(data: bytes) -> str:
    """Encode ``data`` as unpadded base64url text."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    """Decode canonical base64url text, with or without padding.

    Raises:
        FormatError: ``text`` is not a string, contains characters outside
            the URL-safe alphabet, has an impossible length or wrong
            padding, or is a non-canonical encoding.
    """
    if not isinstance(text, str):
        raise FormatError(f"base64url input must be str, not {type(text).__name__}")
    match = _B64URL.fullmatch(text)
    if match is None:
        raise FormatError("invalid base64url: characters outside the URL-safe alphabet")
    padding = match.group(1)
    if padding and len(text) % 4:
        raise FormatError("invalid base64url: incorrect padding")
    core = text[: len(text) - len(padding)]
    try:
        data = base64.urlsafe_b64decode(core + "=" * (-len(core) % 4))
    except binascii.Error as exc:
        raise FormatError(f"invalid base64url: {exc}") from exc
    if b64url_encode(data) != core:
        raise FormatError("invalid base64url: non-canonical encoding (nonzero trailing bits)")
    return data
