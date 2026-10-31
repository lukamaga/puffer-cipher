"""Persistent Ed25519 signing key for the public randomness beacon.

The beacon signs every pulse (DESIGN.md section 3.4), and verifiers pin its
public key, so the key must survive restarts: losing it silently would
break every client's chain verification. It is stored as::

    {"format": "puffer-signing-key", "version": 1,
     "seed": "<base64url, 32 bytes>", "public_key": "<64 hex>"}

* Only the 32-byte RFC 8032 seed is secret; the public key is stored too so
  operators can read it without loading the secret, and it is re-derived
  and compared on load as a tamper/corruption check.
* The file is created atomically with mode ``0600`` and never clobbered.
  If two processes race to create it, the loser loads the winner's key
  instead of overwriting it, so both end up signing with the same identity.
* Signing itself is libsodium's Ed25519 via :mod:`nacl.signing`.
"""

from __future__ import annotations

import re
from typing import Any

import nacl.utils
from nacl.signing import SigningKey, VerifyKey

from puffer.crypto._files import (
    StrPath,
    dump_json,
    read_json_object,
    warn_if_exposed,
    write_private_file,
)
from puffer.crypto.aead import RandomBytes
from puffer.crypto.encoding import b64url_decode, b64url_encode
from puffer.crypto.errors import FormatError

SIGNING_KEY_FORMAT = "puffer-signing-key"
SIGNING_KEY_VERSION = 1
SEED_BYTES = 32

_FIELDS = frozenset({"format", "version", "seed", "public_key"})
_PUBLIC_KEY_RE = re.compile(r"[0-9a-f]{64}")


def public_key_hex(key: SigningKey | VerifyKey) -> str:
    """Return the 32-byte Ed25519 public key as 64 lowercase hex characters."""
    verify_key = key.verify_key if isinstance(key, SigningKey) else key
    return bytes(verify_key).hex()


def _parse(doc: dict[str, Any]) -> SigningKey:
    if doc.keys() != _FIELDS:
        raise FormatError(f"signing key file must have exactly the fields {sorted(_FIELDS)}")
    if doc["format"] != SIGNING_KEY_FORMAT:
        raise FormatError(f"not a puffer signing key file (format {doc['format']!r})")
    version = doc["version"]
    # ``type(...) is int`` because ``True == 1`` would otherwise pass.
    if type(version) is not int or version != SIGNING_KEY_VERSION:
        raise FormatError(f"unsupported signing key file version {version!r}")
    encoded_seed, stored_public = doc["seed"], doc["public_key"]
    if not isinstance(encoded_seed, str):
        raise FormatError("signing key 'seed' must be a base64url string")
    seed = b64url_decode(encoded_seed)
    if len(seed) != SEED_BYTES:
        raise FormatError(f"signing key seed must be {SEED_BYTES} bytes, got {len(seed)}")
    if not isinstance(stored_public, str) or _PUBLIC_KEY_RE.fullmatch(stored_public) is None:
        raise FormatError("signing key 'public_key' must be 64 lowercase hex characters")
    signing_key = SigningKey(seed)
    if public_key_hex(signing_key) != stored_public:
        raise FormatError("signing key 'public_key' does not match its seed: corrupted or tampered")
    return signing_key


def _load(path: StrPath) -> SigningKey:
    doc = read_json_object(path, what="signing key file")
    warn_if_exposed(path, stacklevel=3)
    return _parse(doc)


def load_or_create_signing_key(
    path: StrPath, *, random_bytes: RandomBytes = nacl.utils.random
) -> SigningKey:
    """Load the signing key at ``path``, creating it first if it does not exist.

    A new key's seed is drawn from ``random_bytes`` (inject the fish-seeded
    PufferRNG to make the beacon identity itself fish-born).

    Raises:
        FormatError: an existing file is malformed, or its public key does
            not match its seed.
        ValueError: ``random_bytes`` returned the wrong number of bytes.
        OSError: the file cannot be read or created.
    """
    try:
        return _load(path)
    except FileNotFoundError:
        pass
    seed = random_bytes(SEED_BYTES)
    if len(seed) != SEED_BYTES:
        raise ValueError(f"random_bytes returned {len(seed)} bytes, expected {SEED_BYTES}")
    signing_key = SigningKey(seed)
    doc = {
        "format": SIGNING_KEY_FORMAT,
        "version": SIGNING_KEY_VERSION,
        "seed": b64url_encode(seed),
        "public_key": public_key_hex(signing_key),
    }
    try:
        write_private_file(path, dump_json(doc))
    except FileExistsError:
        # Another process created the key between our load and our write:
        # adopt its identity rather than replacing it.
        return _load(path)
    return signing_key
