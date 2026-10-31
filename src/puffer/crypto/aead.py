"""Short authenticated messages: XChaCha20-Poly1305-IETF (DESIGN.md section 3.3).

Wire format: ``nonce (24) || ciphertext || tag (16)``, optionally as
unpadded base64url text. The associated data is the constant
:data:`MESSAGE_AAD`, which binds every message to this protocol and version:
a ciphertext produced for some other purpose with the same key will not
authenticate here.

Why XChaCha20 rather than ChaCha20 or AES-GCM: the 192-bit nonce makes
*random* nonces safe for an effectively unlimited number of messages per
key, so the caller never has to keep a counter. The nonce comes from the
injectable :data:`RandomBytes` source - in production the fish-seeded
``PufferRNG``, which is never weaker than ``os.urandom`` - and defaults to
libsodium's CSPRNG. The primitive itself is libsodium's; nothing here is
home-made.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import nacl.bindings
import nacl.exceptions
import nacl.utils

from puffer.crypto.encoding import b64url_decode, b64url_encode
from puffer.crypto.errors import AuthenticationError, FormatError
from puffer.crypto.keys import check_key

RandomBytes: TypeAlias = Callable[[int], bytes]
"""A source of ``n`` uniformly random bytes, e.g. ``nacl.utils.random`` or a PufferRNG."""

MESSAGE_AAD = b"puffer-msg-v1"
NONCE_BYTES: int = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
TAG_BYTES: int = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_ABYTES


def encrypt_message(
    plaintext: bytes,
    key: bytes,
    *,
    nonce: bytes | None = None,
    random_bytes: RandomBytes = nacl.utils.random,
) -> bytes:
    """Encrypt and authenticate ``plaintext``; return ``nonce || ciphertext``.

    ``nonce`` exists only for reproducible test vectors. Supplying the same
    nonce twice with one key destroys confidentiality and authenticity, so
    normal callers leave it ``None`` and a fresh one is drawn from
    ``random_bytes``.

    Raises:
        TypeError, ValueError: bad key, or a nonce of the wrong length
            (including one returned by a misbehaving ``random_bytes``).
    """
    check_key(key)
    if nonce is None:
        nonce = random_bytes(NONCE_BYTES)
    if len(nonce) != NONCE_BYTES:
        raise ValueError(f"nonce must be {NONCE_BYTES} bytes, got {len(nonce)}")
    ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
        plaintext, MESSAGE_AAD, nonce, key
    )
    return nonce + ciphertext


def decrypt_message(blob: bytes, key: bytes) -> bytes:
    """Verify and decrypt a ``nonce || ciphertext`` blob from :func:`encrypt_message`.

    Raises:
        FormatError: ``blob`` is too short to hold a nonce and a tag.
        AuthenticationError: wrong key, or the blob was modified.
        TypeError, ValueError: ``key`` is not a valid 32-byte key.
    """
    check_key(key)
    if len(blob) < NONCE_BYTES + TAG_BYTES:
        raise FormatError(
            f"message too short: {len(blob)} bytes, need at least {NONCE_BYTES + TAG_BYTES}"
        )
    nonce, ciphertext = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    try:
        return nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
            ciphertext, MESSAGE_AAD, nonce, key
        )
    except nacl.exceptions.CryptoError:
        # libsodium intentionally gives no detail; neither do we.
        raise AuthenticationError("message authentication failed: wrong key or tampered") from None


def encrypt_message_b64(
    plaintext: bytes,
    key: bytes,
    *,
    nonce: bytes | None = None,
    random_bytes: RandomBytes = nacl.utils.random,
) -> str:
    """Like :func:`encrypt_message`, returning unpadded base64url text."""
    return b64url_encode(encrypt_message(plaintext, key, nonce=nonce, random_bytes=random_bytes))


def decrypt_message_b64(token: str, key: bytes) -> bytes:
    """Like :func:`decrypt_message`, taking base64url text (FormatError on bad encoding)."""
    return decrypt_message(b64url_decode(token), key)
