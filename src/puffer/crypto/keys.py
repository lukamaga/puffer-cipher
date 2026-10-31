"""Symmetric key identity and the ``*.key`` file format (DESIGN.md section 3.1).

A key file is a small JSON document::

    {"format": "puffer-key", "version": 1, "key_id": "a1b2c3d4e5f60718",
     "created": "2026-10-31T15:00:00Z", "key": "<base64url, 32 bytes>",
     "receipt": {...}}

Design decisions:

* ``key_id = hex(BLAKE2b-64(key, person=b"puffer-key-id"))``. The key is
  256 uniformly random bits, so a 64-bit personalised hash reveals nothing
  useful about it while letting ``.puf`` headers name their key and fail
  fast with :class:`WrongKeyError` instead of an opaque tag failure. The
  personalisation string separates this hash from every other BLAKE2b use
  in the project.
* The stored ``key_id`` is re-derived on load. A mismatch means the file was
  corrupted or edited, and it is rejected rather than "repaired": a key file
  whose two halves disagree cannot be trusted to be either.
* The optional ``receipt`` is the :class:`EntropyReceipt` of the draw that
  produced the key - which fish and camera channels contributed. It is
  public metadata and never contains secret material.
* Files are written atomically with mode ``0600`` from creation and are
  never overwritten unless asked; :class:`KeyFile` redacts the key in its
  ``repr`` so it cannot leak into logs or tracebacks. (Python offers no
  reliable way to wipe secrets from memory, so we do not pretend to.)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from puffer.crypto._files import (
    StrPath,
    dump_json,
    read_json_object,
    warn_if_exposed,
    write_private_file,
)
from puffer.crypto.encoding import b64url_decode, b64url_encode
from puffer.crypto.errors import FormatError

KEY_BYTES = 32
"""Length of every symmetric key (XChaCha20-Poly1305, secretstream)."""

KEY_ID_BYTES = 8
"""BLAKE2b-64 key id length; the full id is stored in ``.puf`` headers."""

KEY_FILE_FORMAT = "puffer-key"
KEY_FILE_VERSION = 1

KEY_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
"""A well-formed textual key id (match with ``fullmatch``)."""

_KEY_ID_PERSON = b"puffer-key-id"
_FIELDS = frozenset({"format", "version", "key_id", "created", "key", "receipt"})
_REQUIRED = _FIELDS - {"receipt"}
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def check_key(key: bytes) -> None:
    """Raise unless ``key`` is a :data:`KEY_BYTES`-long ``bytes`` object.

    Raises:
        TypeError: ``key`` is not ``bytes`` (libsodium bindings need bytes).
        ValueError: ``key`` has the wrong length.
    """
    if not isinstance(key, bytes):
        raise TypeError(f"key must be bytes, not {type(key).__name__}")
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes, got {len(key)}")


def key_id_bytes(key: bytes) -> bytes:
    """Return the 8-byte public identifier of ``key``."""
    check_key(key)
    return hashlib.blake2b(key, digest_size=KEY_ID_BYTES, person=_KEY_ID_PERSON).digest()


def key_id(key: bytes) -> str:
    """Return the key id of ``key`` as 16 lowercase hex characters."""
    return key_id_bytes(key).hex()


@dataclass(frozen=True, slots=True)
class KeyFile:
    """A loaded or freshly saved key file.

    Construction validates that ``key_id`` belongs to ``key`` and that
    ``created`` is timezone-aware, so a :class:`KeyFile` instance is always
    internally consistent.
    """

    key: bytes
    key_id: str
    created: datetime
    receipt: dict[str, Any] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.key_id != key_id(self.key):
            raise ValueError("key_id does not match key")
        if self.created.utcoffset() is None:
            raise ValueError("created must be a timezone-aware datetime")

    def __repr__(self) -> str:
        return (
            f"KeyFile(key=<redacted>, key_id={self.key_id!r}, "
            f"created={self.created.isoformat()!r}, receipt={self.receipt!r})"
        )

    def to_json_bytes(self) -> bytes:
        """Serialize to the on-disk JSON representation (contains the secret key)."""
        doc: dict[str, Any] = {
            "format": KEY_FILE_FORMAT,
            "version": KEY_FILE_VERSION,
            "key_id": self.key_id,
            "created": self.created.astimezone(UTC).strftime(_TIMESTAMP_FORMAT),
            "key": b64url_encode(self.key),
        }
        if self.receipt is not None:
            doc["receipt"] = self.receipt
        return dump_json(doc)


def save_key(
    path: StrPath,
    key: bytes,
    *,
    receipt: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> KeyFile:
    """Write ``key`` (and its optional entropy receipt) to a new key file.

    The file is created atomically with mode ``0600``. ``created`` is the
    current UTC time truncated to whole seconds, exactly as it is stored, so
    the returned :class:`KeyFile` equals what :func:`load_key` will return.

    Raises:
        FileExistsError: ``path`` exists and ``overwrite`` is false.
        TypeError, ValueError: invalid key, or a receipt that is not
            strict-JSON serializable (e.g. contains ``NaN``).
    """
    check_key(key)
    key_file = KeyFile(
        key=key,
        key_id=key_id(key),
        created=datetime.now(UTC).replace(microsecond=0),
        receipt=dict(receipt) if receipt is not None else None,
    )
    write_private_file(path, key_file.to_json_bytes(), overwrite=overwrite)
    return key_file


def _parse_created(value: object) -> datetime:
    if not isinstance(value, str):
        raise FormatError("key file 'created' must be a string")
    try:
        created = datetime.fromisoformat(value)
    except ValueError as exc:
        raise FormatError(f"key file 'created' is not an ISO 8601 timestamp: {value!r}") from exc
    if created.utcoffset() is None:
        raise FormatError("key file 'created' must include a UTC offset")
    return created.astimezone(UTC)


def _parse_key_document(doc: dict[str, Any]) -> KeyFile:
    unknown = doc.keys() - _FIELDS
    if unknown:
        raise FormatError(f"key file has unknown fields: {sorted(unknown)}")
    missing = _REQUIRED - doc.keys()
    if missing:
        raise FormatError(f"key file is missing fields: {sorted(missing)}")
    if doc["format"] != KEY_FILE_FORMAT:
        raise FormatError(f"not a puffer key file (format {doc['format']!r})")
    version = doc["version"]
    # ``type(...) is int`` because ``True == 1`` would otherwise pass.
    if type(version) is not int or version != KEY_FILE_VERSION:
        raise FormatError(f"unsupported key file version {version!r}")
    encoded = doc["key"]
    if not isinstance(encoded, str):
        raise FormatError("key file 'key' must be a base64url string")
    key = b64url_decode(encoded)
    if len(key) != KEY_BYTES:
        raise FormatError(f"key file key must be {KEY_BYTES} bytes, got {len(key)}")
    stored_id = doc["key_id"]
    if not isinstance(stored_id, str) or KEY_ID_PATTERN.fullmatch(stored_id) is None:
        raise FormatError("key file 'key_id' must be 16 lowercase hex characters")
    if stored_id != key_id(key):
        raise FormatError("key file 'key_id' does not match its key: file is corrupted or tampered")
    receipt = doc.get("receipt")
    if receipt is not None and not isinstance(receipt, dict):
        raise FormatError("key file 'receipt' must be a JSON object")
    return KeyFile(
        key=key, key_id=stored_id, created=_parse_created(doc["created"]), receipt=receipt
    )


def load_key(path: StrPath) -> KeyFile:
    """Load and fully validate a key file.

    Emits :class:`InsecurePermissionsWarning` if the file is accessible to
    other users.

    Raises:
        FormatError: wrong format/version, missing or unknown fields, bad
            encoding, wrong key length, or a ``key_id`` that does not match
            the key.
        OSError: the file cannot be read.
    """
    doc = read_json_object(path, what="key file")
    warn_if_exposed(path, stacklevel=2)
    return _parse_key_document(doc)
