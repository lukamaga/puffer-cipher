"""The ``.puf`` encrypted file format (DESIGN.md section 3.2).

Layout (all integers big-endian)::

    offset size  field
    0      4     magic b"PUFR"
    4      1     version = 1
    5      1     kdf: 0 = raw key, 1 = Argon2id password
    6      2     flags = 0 (reserved)
    8      4     chunk_size (1 KiB .. 16 MiB)
    12     8     key_id (kdf 0) or zeros (kdf 1)
    -- kdf 1 only --
    20     8     opslimit
    28     8     memlimit
    36     16    salt
    -- always --
    ..     24    secretstream header
    ..     ...   chunks of chunk_size + 17 bytes; only the last may be shorter

The payload is libsodium's ``crypto_secretstream_xchacha20poly1305``, which
already solves the hard problems of chunked encryption: every chunk is
authenticated, chunks are bound to their position (reordering, dropping or
duplicating fails), and the last chunk carries ``TAG_FINAL`` so truncation
at a chunk boundary is detectable. On top of it this module adds:

* **Header authentication.** The whole preamble (magic .. secretstream
  header) is the associated data of chunk 0, so flipping any header bit -
  chunk size, KDF parameters, salt, key id - makes decryption fail.
* **Early, explicit failures.** Structure is validated before any
  cryptography runs: bad magic/version/KDF/flags raise :class:`FormatError`,
  a key id that does not match the supplied key raises
  :class:`WrongKeyError` before decrypting, and a stream that ends without
  ``TAG_FINAL`` raises :class:`TruncatedError`.
* **DoS-resistant parsing.** A crafted header could otherwise demand a
  4 TiB Argon2 run or a 4 GiB read buffer. ``chunk_size`` must lie in
  [1 KiB, 16 MiB]; Argon2id ``opslimit`` in [``OPSLIMIT_MIN``,
  :data:`MAX_OPSLIMIT`] and ``memlimit`` in [``MEMLIMIT_MIN``,
  :data:`MAX_MEMLIMIT`]. The encryptor enforces the same bounds, so we never
  write a file we would refuse to read.
* **All-or-nothing files.** :func:`decrypt_file` writes to a private
  temporary file next to the destination and renames it only after the
  final tag verified; on any failure the temporary file is deleted.

Known limit of the format: a ``.puf`` has no length field, so the final
chunk's size is only known from EOF. Bytes appended after a *full-size*
final chunk are reported as trailing data (:class:`FormatError`); bytes
appended after a *short* final chunk are indistinguishable from a modified
final chunk and fail its tag (:class:`AuthenticationError`). Likewise a cut
inside a chunk fails authentication unless fewer than 17 bytes of that
chunk remain. Every such case is rejected; only the error class differs.
"""

from __future__ import annotations

import enum
import struct
import unicodedata
from dataclasses import dataclass
from typing import BinaryIO

import nacl.bindings
import nacl.exceptions
import nacl.utils
from nacl.pwhash import argon2id

from puffer.crypto._files import StrPath, atomic_output
from puffer.crypto.aead import RandomBytes
from puffer.crypto.errors import AuthenticationError, FormatError, TruncatedError, WrongKeyError
from puffer.crypto.keys import KEY_BYTES, KEY_ID_BYTES, KEY_ID_PATTERN, check_key, key_id

MAGIC = b"PUFR"
FORMAT_VERSION = 1

DEFAULT_CHUNK_SIZE = 64 * 1024
MIN_CHUNK_SIZE = 1024
MAX_CHUNK_SIZE = 16 * 1024 * 1024

MIN_OPSLIMIT: int = argon2id.OPSLIMIT_MIN
MAX_OPSLIMIT = 16
"""Four times ``OPSLIMIT_SENSITIVE``; libsodium's own maximum (2**32 - 1) is a DoS."""
MIN_MEMLIMIT: int = argon2id.MEMLIMIT_MIN
MAX_MEMLIMIT = 4 * 1024**3
"""4 GiB; libsodium allows 4 TiB, which no legitimate file needs."""

SALT_BYTES: int = argon2id.SALTBYTES
STREAM_HEADER_BYTES: int = nacl.bindings.crypto_secretstream_xchacha20poly1305_HEADERBYTES
CHUNK_OVERHEAD: int = nacl.bindings.crypto_secretstream_xchacha20poly1305_ABYTES

_TAG_MESSAGE: int = nacl.bindings.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE
_TAG_FINAL: int = nacl.bindings.crypto_secretstream_xchacha20poly1305_TAG_FINAL

_FIXED = struct.Struct(f">4sBBHI{KEY_ID_BYTES}s")  # magic, version, kdf, flags, chunk_size, key_id
_ARGON2 = struct.Struct(f">QQ{SALT_BYTES}s")  # opslimit, memlimit, salt
_NO_KEY_ID = bytes(KEY_ID_BYTES)
_ONE_CREDENTIAL = "exactly one of key= or password= must be given"


class Kdf(enum.IntEnum):
    """How the stream key is obtained."""

    RAW_KEY = 0
    """A 32-byte key from a key file, used directly."""
    ARGON2ID = 1
    """``argon2id.kdf(32, password, salt, opslimit, memlimit)``."""


def _check_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        raise FormatError(f"{name} {value} outside the accepted range [{low}, {high}]")


@dataclass(frozen=True, slots=True)
class Argon2Params:
    """Argon2id cost parameters and salt of a password-protected file.

    Validated on construction, so out-of-range values from a crafted header
    are rejected before any key derivation runs.
    """

    opslimit: int
    memlimit: int
    salt: bytes

    def __post_init__(self) -> None:
        _check_range("Argon2id opslimit", self.opslimit, MIN_OPSLIMIT, MAX_OPSLIMIT)
        _check_range("Argon2id memlimit", self.memlimit, MIN_MEMLIMIT, MAX_MEMLIMIT)
        if len(self.salt) != SALT_BYTES:
            raise FormatError(f"Argon2id salt must be {SALT_BYTES} bytes, got {len(self.salt)}")


@dataclass(frozen=True, slots=True)
class FileHeader:
    """Parsed (and validated) preamble of a ``.puf`` file.

    ``key_id`` is the 16-hex-character id of the raw key (``Kdf.RAW_KEY``)
    or ``None`` for password files; ``argon2`` is set exactly for
    ``Kdf.ARGON2ID``. Every field has a single valid encoding, so
    :meth:`to_bytes` reproduces the on-disk preamble byte for byte - which
    is what chunk 0 authenticates.
    """

    kdf: Kdf
    chunk_size: int
    stream_header: bytes
    key_id: str | None = None
    argon2: Argon2Params | None = None

    def __post_init__(self) -> None:
        _check_range("chunk_size", self.chunk_size, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
        if len(self.stream_header) != STREAM_HEADER_BYTES:
            raise FormatError(f"secretstream header must be {STREAM_HEADER_BYTES} bytes")
        if self.kdf == Kdf.RAW_KEY:
            if self.key_id is None or KEY_ID_PATTERN.fullmatch(self.key_id) is None:
                raise FormatError("a raw-key file needs a 16-hex-character key_id")
            if self.argon2 is not None:
                raise FormatError("a raw-key file has no Argon2id parameters")
        elif self.key_id is not None or self.argon2 is None:
            raise FormatError("a password file has Argon2id parameters and no key_id")

    def to_bytes(self) -> bytes:
        """Encode the preamble: every byte before the first chunk."""
        kid = bytes.fromhex(self.key_id) if self.key_id is not None else _NO_KEY_ID
        parts = [_FIXED.pack(MAGIC, FORMAT_VERSION, self.kdf, 0, self.chunk_size, kid)]
        if self.argon2 is not None:
            parts.append(_ARGON2.pack(self.argon2.opslimit, self.argon2.memlimit, self.argon2.salt))
        parts.append(self.stream_header)
        return b"".join(parts)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read ``size`` bytes, looping over short reads; return fewer only at EOF."""
    data = stream.read(size)
    if len(data) == size or not data:
        return data
    parts = [data]
    remaining = size - len(data)
    while remaining:
        part = stream.read(remaining)
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _read_header_field(stream: BinaryIO, size: int, what: str) -> bytes:
    data = _read_exact(stream, size)
    if len(data) < size:
        raise TruncatedError(f"file ends inside the header ({what})")
    return data


def read_header(stream: BinaryIO) -> FileHeader:
    """Read and validate the preamble, leaving ``stream`` at the first chunk.

    Raises:
        FormatError: not a ``.puf`` file, unsupported version, unknown KDF,
            nonzero flags, or a field outside its accepted range.
        TruncatedError: the stream ends inside the header.
    """
    fixed = _read_exact(stream, _FIXED.size)
    if fixed[: len(MAGIC)] != MAGIC:
        raise FormatError("not a puffer-cipher file (bad magic)")
    if len(fixed) < _FIXED.size:
        raise TruncatedError("file ends inside the header")
    _, version, kdf_value, flags, chunk_size, raw_key_id = _FIXED.unpack(fixed)
    if version != FORMAT_VERSION:
        raise FormatError(f"unsupported .puf version {version}")
    try:
        kdf = Kdf(kdf_value)
    except ValueError:
        raise FormatError(f"unknown kdf {kdf_value}") from None
    if flags != 0:
        raise FormatError(f"reserved flags must be zero, got {flags:#06x}")
    key_id_hex: str | None = raw_key_id.hex()
    argon2: Argon2Params | None = None
    if kdf == Kdf.ARGON2ID:
        if raw_key_id != _NO_KEY_ID:
            raise FormatError("password files must have an all-zero key_id field")
        key_id_hex = None
        opslimit, memlimit, salt = _ARGON2.unpack(
            _read_header_field(stream, _ARGON2.size, "Argon2id parameters")
        )
        argon2 = Argon2Params(opslimit=opslimit, memlimit=memlimit, salt=salt)
    stream_header = _read_header_field(stream, STREAM_HEADER_BYTES, "secretstream header")
    return FileHeader(
        kdf=kdf,
        chunk_size=chunk_size,
        stream_header=stream_header,
        key_id=key_id_hex,
        argon2=argon2,
    )


def _password_bytes(password: str | bytes) -> bytes:
    """UTF-8 encode a text password after NFC normalisation.

    NFC makes the same visible password derive the same key regardless of
    how the keyboard/OS composed accented characters (RFC 8265 makes the
    same choice). ``bytes`` passwords are used verbatim.
    """
    raw = (
        unicodedata.normalize("NFC", password).encode("utf-8")
        if isinstance(password, str)
        else password
    )
    if not raw:
        raise ValueError("password must not be empty")
    return raw


def _derive_key(password: str | bytes, params: Argon2Params) -> bytes:
    return argon2id.kdf(
        KEY_BYTES,
        _password_bytes(password),
        params.salt,
        opslimit=params.opslimit,
        memlimit=params.memlimit,
    )


def encrypt_stream(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    key: bytes | None = None,
    password: str | bytes | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    opslimit: int = argon2id.OPSLIMIT_MODERATE,
    memlimit: int = argon2id.MEMLIMIT_MODERATE,
    random_bytes: RandomBytes = nacl.utils.random,
) -> FileHeader:
    """Encrypt everything readable from ``src`` into ``dst`` as a ``.puf`` stream.

    Exactly one of ``key`` (32 bytes, typically from a key file) or
    ``password`` must be given. ``opslimit``/``memlimit`` apply to password
    mode only; ``random_bytes`` supplies the Argon2id salt (inject the
    fish-seeded PufferRNG here). The secretstream header nonce is generated
    inside libsodium, whose CSPRNG is already the right tool for a value
    that only has to be unique.

    Plaintext is read one chunk ahead so the last chunk - even a full-size
    one - can be tagged ``TAG_FINAL``; empty input produces a single empty
    final chunk.

    Returns:
        The header that was written.

    Raises:
        ValueError: both or neither credential, an empty password, a bad
            key, or ``chunk_size``/Argon2id parameters outside the readable
            range (:class:`FormatError` is a ``ValueError``).
    """
    _check_range("chunk_size", chunk_size, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
    argon2: Argon2Params | None = None
    if key is not None and password is None:
        check_key(key)
        stream_key, kdf, kid = key, Kdf.RAW_KEY, key_id(key)
    elif password is not None and key is None:
        argon2 = Argon2Params(opslimit=opslimit, memlimit=memlimit, salt=random_bytes(SALT_BYTES))
        stream_key, kdf, kid = _derive_key(password, argon2), Kdf.ARGON2ID, None
    else:
        raise ValueError(_ONE_CREDENTIAL)

    state = nacl.bindings.crypto_secretstream_xchacha20poly1305_state()
    stream_header = nacl.bindings.crypto_secretstream_xchacha20poly1305_init_push(state, stream_key)
    header = FileHeader(
        kdf=kdf, chunk_size=chunk_size, stream_header=stream_header, key_id=kid, argon2=argon2
    )
    ad = header.to_bytes()
    dst.write(ad)

    chunk = _read_exact(src, chunk_size)
    while True:
        following = _read_exact(src, chunk_size) if len(chunk) == chunk_size else b""
        tag = _TAG_MESSAGE if following else _TAG_FINAL
        dst.write(nacl.bindings.crypto_secretstream_xchacha20poly1305_push(state, chunk, ad, tag))
        if tag == _TAG_FINAL:
            return header
        chunk, ad = following, b""


def _stream_key_for(header: FileHeader, key: bytes | None, password: str | bytes | None) -> bytes:
    """Check the credential against the header (no crypto yet), then return the stream key."""
    if (key is None) == (password is None):
        raise ValueError(_ONE_CREDENTIAL)
    if header.kdf == Kdf.RAW_KEY:
        if key is None:
            raise WrongKeyError(
                f"file is encrypted with key {header.key_id}, not a password; pass key="
            )
        check_key(key)
        supplied = key_id(key)
        if supplied != header.key_id:
            raise WrongKeyError(f"file is encrypted with key {header.key_id}, not {supplied}")
        return key
    if password is None:
        raise WrongKeyError("file is password-protected; pass password=")
    assert header.argon2 is not None
    return _derive_key(password, header.argon2)


def decrypt_stream(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    key: bytes | None = None,
    password: str | bytes | None = None,
) -> FileHeader:
    """Verify and decrypt a ``.puf`` stream from ``src`` into ``dst``.

    Each chunk is written to ``dst`` as soon as its tag verifies, so memory
    use is bounded by ``chunk_size``. Released chunks are authentic, but if
    this function raises, ``dst`` holds an *incomplete* plaintext (the final
    chunk is only written once the stream is known to end there): callers
    that need all-or-nothing semantics should use :func:`decrypt_file` (or
    discard ``dst`` on error).

    Returns:
        The validated header.

    Raises:
        FormatError: malformed header, an unexpected secretstream tag, or
            trailing bytes after the final chunk.
        WrongKeyError: the key id differs from the supplied key's, or the
            credential type does not match the file's KDF.
        AuthenticationError: a chunk (or the header, via chunk 0) was
            modified, chunks were reordered, or the key/password is wrong.
        TruncatedError: the stream ends before the ``TAG_FINAL`` chunk.
        ValueError: both or neither credential, or an invalid key/password.
    """
    header = read_header(src)
    stream_key = _stream_key_for(header, key, password)
    state = nacl.bindings.crypto_secretstream_xchacha20poly1305_state()
    nacl.bindings.crypto_secretstream_xchacha20poly1305_init_pull(
        state, header.stream_header, stream_key
    )
    ad = header.to_bytes()
    index = 0
    while True:
        chunk = _read_exact(src, header.chunk_size + CHUNK_OVERHEAD)
        if len(chunk) < CHUNK_OVERHEAD:
            raise TruncatedError(f"stream ends at chunk {index} without the final chunk")
        try:
            plaintext, tag = nacl.bindings.crypto_secretstream_xchacha20poly1305_pull(
                state, chunk, ad
            )
        except nacl.exceptions.CryptoError:
            # libsodium intentionally gives no detail; neither do we.
            hint = "wrong key/password, or header or" if index == 0 else "chunk"
            raise AuthenticationError(
                f"authentication failed at chunk {index}: {hint} data modified"
            ) from None
        if tag not in (_TAG_MESSAGE, _TAG_FINAL):
            raise FormatError(f"unexpected secretstream tag {tag} at chunk {index}")
        if tag == _TAG_FINAL and src.read(1):
            raise FormatError("unexpected data after the final chunk")
        dst.write(plaintext)
        if tag == _TAG_FINAL:
            return header
        ad = b""
        index += 1


def encrypt_file(
    src_path: StrPath,
    dst_path: StrPath,
    *,
    overwrite: bool = False,
    key: bytes | None = None,
    password: str | bytes | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    opslimit: int = argon2id.OPSLIMIT_MODERATE,
    memlimit: int = argon2id.MEMLIMIT_MODERATE,
    random_bytes: RandomBytes = nacl.utils.random,
) -> FileHeader:
    """Encrypt the file at ``src_path`` to ``dst_path`` atomically.

    The output is written to a private temporary file in the destination
    directory and renamed into place only when complete. Without
    ``overwrite`` an existing ``dst_path`` raises :class:`FileExistsError`.
    Other arguments are as for :func:`encrypt_stream`.
    """
    with open(src_path, "rb") as src, atomic_output(dst_path, overwrite=overwrite) as dst:
        return encrypt_stream(
            src,
            dst,
            key=key,
            password=password,
            chunk_size=chunk_size,
            opslimit=opslimit,
            memlimit=memlimit,
            random_bytes=random_bytes,
        )


def decrypt_file(
    src_path: StrPath,
    dst_path: StrPath,
    *,
    overwrite: bool = False,
    key: bytes | None = None,
    password: str | bytes | None = None,
) -> FileHeader:
    """Decrypt ``src_path`` to ``dst_path``, publishing plaintext only on full success.

    Plaintext goes to a ``0600`` temporary file in the destination
    directory, which is renamed to ``dst_path`` only after the final tag has
    verified and no trailing data was found. On any failure the temporary
    file is deleted (unlinked - Python cannot securely erase disk blocks)
    and ``dst_path`` is untouched, so a truncated or tampered file never
    yields a plaintext that could be mistaken for the complete one.
    """
    with open(src_path, "rb") as src, atomic_output(dst_path, overwrite=overwrite) as dst:
        return decrypt_stream(src, dst, key=key, password=password)
