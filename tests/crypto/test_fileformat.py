"""The .puf stream format: byte layout, round trips, and every way to break a file.

Error-class expectations are spelled out per header byte so that a change
in validation order or semantics shows up as a precise failure.
"""

from __future__ import annotations

import hashlib
import io
import struct
import unicodedata
from typing import Any

import nacl.bindings as sodium
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from nacl.pwhash import argon2id

from puffer.crypto import (
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    Argon2Params,
    AuthenticationError,
    FileHeader,
    FormatError,
    Kdf,
    PufferCryptoError,
    TruncatedError,
    WrongKeyError,
    decrypt_stream,
    encrypt_stream,
    key_id,
    read_header,
)

PROPERTY = settings(derandomize=True, database=None, deadline=None, max_examples=60)

C = 1024  # chunk size used by most tests
ABYTES = 17
KEY_PREAMBLE = 44  # 20 fixed + 24 secretstream header
PASSWORD_PREAMBLE = 76  # 20 fixed + 32 Argon2id + 24 secretstream header
PASSWORD = "correct horse battery staple"


def seal(data: bytes, **kw: Any) -> bytes:
    out = io.BytesIO()
    encrypt_stream(io.BytesIO(data), out, **kw)
    return out.getvalue()


def unseal(blob: bytes, **kw: Any) -> bytes:
    out = io.BytesIO()
    decrypt_stream(io.BytesIO(blob), out, **kw)
    return out.getvalue()


def flip(blob: bytes, offset: int, mask: int = 0x01) -> bytes:
    tampered = bytearray(blob)
    tampered[offset] ^= mask
    return bytes(tampered)


def chunks_of(blob: bytes, preamble: int, chunk_size: int = C) -> list[bytes]:
    body = blob[preamble:]
    step = chunk_size + ABYTES
    return [body[i : i + step] for i in range(0, len(body), step)]


def reference_decrypt(blob: bytes, stream_key: bytes, preamble: int) -> list[tuple[bytes, int]]:
    """Independent decoder written straight from DESIGN.md section 3.2."""
    state = sodium.crypto_secretstream_xchacha20poly1305_state()
    sodium.crypto_secretstream_xchacha20poly1305_init_pull(
        state, blob[preamble - 24 : preamble], stream_key
    )
    out = []
    for i, chunk in enumerate(chunks_of(blob, preamble, struct.unpack(">I", blob[8:12])[0])):
        ad = blob[:preamble] if i == 0 else b""
        out.append(sodium.crypto_secretstream_xchacha20poly1305_pull(state, chunk, ad))
    return out


@pytest.fixture
def plaintext(rng: np.random.Generator) -> bytes:
    """2.5 chunks: two full chunks and a short final one."""
    return rng.bytes(2 * C + C // 2)


@pytest.fixture
def key_blob(key: bytes, plaintext: bytes) -> bytes:
    return seal(plaintext, key=key, chunk_size=C)


@pytest.fixture
def password_blob(plaintext: bytes, fast_kdf: dict[str, Any]) -> bytes:
    return seal(plaintext, password=PASSWORD, chunk_size=C, **fast_kdf)


# --- round trips -------------------------------------------------------------

SIZES = [0, 1, C - 1, C, C + 1, 3 * C + C // 2]


@pytest.mark.parametrize("size", SIZES)
def test_round_trip_raw_key(size: int, key: bytes, rng: np.random.Generator) -> None:
    data = rng.bytes(size)
    blob = seal(data, key=key, chunk_size=C)
    n_chunks = max(1, -(-size // C))
    assert len(blob) == KEY_PREAMBLE + size + n_chunks * ABYTES
    assert unseal(blob, key=key) == data


@pytest.mark.parametrize("size", SIZES)
def test_round_trip_password(size: int, rng: np.random.Generator, fast_kdf: dict[str, Any]) -> None:
    data = rng.bytes(size)
    blob = seal(data, password=PASSWORD, chunk_size=C, **fast_kdf)
    n_chunks = max(1, -(-size // C))
    assert len(blob) == PASSWORD_PREAMBLE + size + n_chunks * ABYTES
    assert unseal(blob, password=PASSWORD) == data


@pytest.mark.parametrize("size", [0, DEFAULT_CHUNK_SIZE - 1, DEFAULT_CHUNK_SIZE, 150_000])
def test_round_trip_default_chunk_size(size: int, key: bytes, rng: np.random.Generator) -> None:
    data = rng.bytes(size)
    blob = seal(data, key=key)
    assert struct.unpack(">I", blob[8:12])[0] == DEFAULT_CHUNK_SIZE == 65536
    assert unseal(blob, key=key) == data


def test_round_trip_several_megabytes_small_chunks(key: bytes, rng: np.random.Generator) -> None:
    data = rng.bytes(3 * 1024 * 1024 + 123)
    blob = seal(data, key=key, chunk_size=MIN_CHUNK_SIZE)
    assert len(blob) == KEY_PREAMBLE + len(data) + 3073 * ABYTES
    assert unseal(blob, key=key) == data


@pytest.mark.parametrize("chunk_size", [MIN_CHUNK_SIZE, MAX_CHUNK_SIZE])
def test_chunk_size_bounds_are_inclusive(chunk_size: int, key: bytes) -> None:
    assert unseal(seal(b"edge", key=key, chunk_size=chunk_size), key=key) == b"edge"


class Trickle(io.RawIOBase):
    """A raw stream that returns at most 7 bytes per read, like a slow pipe."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        piece = self._buf.read(min(7, len(b)))
        b[: len(piece)] = piece
        return len(piece)


def test_short_reads_are_handled(key: bytes, plaintext: bytes) -> None:
    out = io.BytesIO()
    encrypt_stream(Trickle(plaintext), out, key=key, chunk_size=C)  # type: ignore[arg-type]
    result = io.BytesIO()
    decrypt_stream(Trickle(out.getvalue()), result, key=key)  # type: ignore[arg-type]
    assert result.getvalue() == plaintext


# --- byte layout -------------------------------------------------------------


def test_raw_key_layout(key: bytes, key_blob: bytes, plaintext: bytes) -> None:
    magic, version, kdf, flags, chunk_size, kid = struct.unpack(">4sBBHI8s", key_blob[:20])
    assert (magic, version, kdf, flags, chunk_size) == (b"PUFR", 1, 0, 0, C)
    assert kid == hashlib.blake2b(key, digest_size=8, person=b"puffer-key-id").digest()
    assert [len(c) for c in chunks_of(key_blob, KEY_PREAMBLE)] == [C + 17, C + 17, C // 2 + 17]
    pieces = reference_decrypt(key_blob, key, KEY_PREAMBLE)
    assert b"".join(m for m, _ in pieces) == plaintext
    assert [tag for _, tag in pieces] == [
        sodium.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE,
        sodium.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE,
        sodium.crypto_secretstream_xchacha20poly1305_TAG_FINAL,
    ]


def test_password_layout(plaintext: bytes) -> None:
    salt = bytes(range(16))
    blob = seal(
        plaintext,
        password=PASSWORD,
        chunk_size=C,
        opslimit=2,
        memlimit=argon2id.MEMLIMIT_MIN,
        random_bytes=lambda n: salt[:n],
    )
    magic, version, kdf, flags, chunk_size, kid = struct.unpack(">4sBBHI8s", blob[:20])
    assert (magic, version, kdf, flags, chunk_size, kid) == (b"PUFR", 1, 1, 0, C, bytes(8))
    assert struct.unpack(">QQ16s", blob[20:52]) == (2, argon2id.MEMLIMIT_MIN, salt)
    stream_key = argon2id.kdf(32, PASSWORD.encode(), salt, 2, argon2id.MEMLIMIT_MIN)
    pieces = reference_decrypt(blob, stream_key, PASSWORD_PREAMBLE)
    assert b"".join(m for m, _ in pieces) == plaintext


def test_empty_input_is_one_empty_final_chunk(key: bytes) -> None:
    blob = seal(b"", key=key, chunk_size=C)
    assert len(blob) == KEY_PREAMBLE + ABYTES
    assert reference_decrypt(blob, key, KEY_PREAMBLE) == [
        (b"", sodium.crypto_secretstream_xchacha20poly1305_TAG_FINAL)
    ]


def test_exact_multiple_has_no_extra_chunk(key: bytes, rng: np.random.Generator) -> None:
    blob = seal(rng.bytes(2 * C), key=key, chunk_size=C)
    tags = [tag for _, tag in reference_decrypt(blob, key, KEY_PREAMBLE)]
    assert tags == [
        sodium.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE,
        sodium.crypto_secretstream_xchacha20poly1305_TAG_FINAL,
    ]


def test_read_header_round_trips_preamble(key: bytes, plaintext: bytes) -> None:
    out = io.BytesIO()
    written = encrypt_stream(io.BytesIO(plaintext), out, key=key, chunk_size=C)
    src = io.BytesIO(out.getvalue())
    header = read_header(src)
    assert header == written
    assert src.tell() == KEY_PREAMBLE
    assert header.to_bytes() == out.getvalue()[:KEY_PREAMBLE]
    assert header.kdf is Kdf.RAW_KEY
    assert header.key_id == key_id(key)
    assert header.argon2 is None


def test_password_header_fields(password_blob: bytes) -> None:
    header = read_header(io.BytesIO(password_blob))
    assert header.kdf is Kdf.ARGON2ID
    assert header.key_id is None
    assert header.argon2 is not None
    assert header.argon2.opslimit == argon2id.OPSLIMIT_MIN
    assert header.argon2.memlimit == argon2id.MEMLIMIT_MIN
    assert header.to_bytes() == password_blob[:PASSWORD_PREAMBLE]


def test_salt_comes_from_injected_rng(plaintext: bytes) -> None:
    calls: list[int] = []

    def rng(n: int) -> bytes:
        calls.append(n)
        return b"\xaa" * n

    out = io.BytesIO()
    header = encrypt_stream(
        io.BytesIO(plaintext),
        out,
        password=PASSWORD,
        opslimit=argon2id.OPSLIMIT_MIN,
        memlimit=argon2id.MEMLIMIT_MIN,
        random_bytes=rng,
    )
    assert calls == [16]
    assert header.argon2 is not None
    assert header.argon2.salt == b"\xaa" * 16


# --- header tampering --------------------------------------------------------

FE, WK, AE = FormatError, WrongKeyError, AuthenticationError

# Expected error after XOR-ing 0x01 into each byte of a raw-key preamble (chunk_size 1024).
KEY_HEADER_TAMPER: list[tuple[range, type[PufferCryptoError], str]] = [
    (range(0, 4), FE, "magic"),
    (range(4, 5), FE, "version 1 -> 0"),
    (range(5, 6), FE, "kdf 0 -> 1: a password file must have a zero key_id"),
    (range(6, 8), FE, "reserved flags"),
    (range(8, 9), FE, "chunk_size 1024 -> 16 MiB + 1024: out of range"),
    (range(9, 12), AE, "chunk_size still in range: preamble is chunk 0's AD"),
    (range(12, 20), WK, "key_id no longer matches the key"),
    (range(20, 44), AE, "secretstream header"),
]

# Same for a password preamble (opslimit 1, memlimit 8192 = 0x2000, chunk_size 1024).
PASSWORD_HEADER_TAMPER: list[tuple[range, type[PufferCryptoError], str]] = [
    (range(0, 4), FE, "magic"),
    (range(4, 5), FE, "version"),
    (range(5, 6), WK, "kdf 1 -> 0: file now claims to need a key file"),
    (range(6, 8), FE, "reserved flags"),
    (range(8, 9), FE, "chunk_size out of range"),
    (range(9, 12), AE, "chunk_size in range"),
    (range(12, 20), FE, "password files need a zero key_id"),
    (range(20, 28), FE, "opslimit 1 -> 0 or >= 2**8: outside [1, 16]"),
    (range(28, 32), FE, "memlimit >= 2**32: above the 4 GiB cap"),
    (range(32, 36), AE, "memlimit still in range (<= 16 MiB): different key"),
    (range(36, 52), AE, "salt: different key"),
    (range(52, 76), AE, "secretstream header"),
]


def _expand(table: list[tuple[range, type[PufferCryptoError], str]]) -> list[Any]:
    return [
        pytest.param(offset, error, id=f"byte{offset}-{error.__name__}")
        for span, error, _ in table
        for offset in span
    ]


def test_tamper_tables_cover_every_header_byte() -> None:
    assert [o for span, _, _ in KEY_HEADER_TAMPER for o in span] == list(range(KEY_PREAMBLE))
    assert [o for span, _, _ in PASSWORD_HEADER_TAMPER for o in span] == list(
        range(PASSWORD_PREAMBLE)
    )


@pytest.mark.parametrize(("offset", "error"), _expand(KEY_HEADER_TAMPER))
def test_raw_key_header_tampering(
    key_blob: bytes, key: bytes, offset: int, error: type[PufferCryptoError]
) -> None:
    out = io.BytesIO()
    with pytest.raises(error) as excinfo:
        decrypt_stream(io.BytesIO(flip(key_blob, offset)), out, key=key)
    assert type(excinfo.value) is error
    assert out.getvalue() == b""


@pytest.mark.parametrize(("offset", "error"), _expand(PASSWORD_HEADER_TAMPER))
def test_password_header_tampering(
    password_blob: bytes, offset: int, error: type[PufferCryptoError]
) -> None:
    out = io.BytesIO()
    with pytest.raises(error) as excinfo:
        decrypt_stream(io.BytesIO(flip(password_blob, offset)), out, password=PASSWORD)
    assert type(excinfo.value) is error
    assert out.getvalue() == b""


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (4, 2, "unsupported .puf version 2"),
        (5, 7, "unknown kdf 7"),
        (6, 0x80, "reserved flags"),
    ],
)
def test_header_error_messages(
    key_blob: bytes, key: bytes, offset: int, value: int, message: str
) -> None:
    crafted = bytearray(key_blob)
    crafted[offset] = value
    with pytest.raises(FormatError, match=message):
        unseal(bytes(crafted), key=key)


def test_bad_magic_message(key_blob: bytes, key: bytes) -> None:
    with pytest.raises(FormatError, match="not a puffer-cipher file"):
        unseal(b"PK\x03\x04" + key_blob[4:], key=key)


@pytest.mark.parametrize(
    ("opslimit", "memlimit"),
    [
        (0, argon2id.MEMLIMIT_MIN),
        (17, argon2id.MEMLIMIT_MIN),
        (2**32 - 1, argon2id.MEMLIMIT_MIN),
        (1, argon2id.MEMLIMIT_MIN - 1),
        (1, 4 * 1024**3 + 1),
        (1, argon2id.MEMLIMIT_MAX),
    ],
)
def test_crafted_kdf_cost_is_rejected_before_argon2(
    password_blob: bytes, opslimit: int, memlimit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header demanding absurd Argon2 cost must be refused without running Argon2."""
    crafted = password_blob[:20] + struct.pack(">QQ", opslimit, memlimit) + password_blob[36:]

    def forbidden(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("Argon2 must not run for a rejected header")

    monkeypatch.setattr(argon2id, "kdf", forbidden)
    with pytest.raises(FormatError, match="outside the accepted range"):
        unseal(crafted, password=PASSWORD)


@pytest.mark.parametrize("chunk_size", [0, 1, MIN_CHUNK_SIZE - 1, MAX_CHUNK_SIZE + 1, 2**32 - 1])
def test_crafted_chunk_size_is_rejected(key_blob: bytes, key: bytes, chunk_size: int) -> None:
    crafted = key_blob[:8] + struct.pack(">I", chunk_size) + key_blob[12:]
    with pytest.raises(FormatError, match="chunk_size"):
        read_header(io.BytesIO(crafted))


# --- body tampering ----------------------------------------------------------


def test_every_ciphertext_byte_is_authenticated(key_blob: bytes, key: bytes) -> None:
    for offset in range(KEY_PREAMBLE, len(key_blob)):
        with pytest.raises(AuthenticationError):
            unseal(flip(key_blob, offset, 0x80), key=key)


def test_password_ciphertext_is_authenticated(password_blob: bytes) -> None:
    for offset in range(PASSWORD_PREAMBLE, len(password_blob), 37):
        with pytest.raises(AuthenticationError):
            unseal(flip(password_blob, offset), password=PASSWORD)


def _join(blob: bytes, chunks: list[bytes], preamble: int = KEY_PREAMBLE) -> bytes:
    return blob[:preamble] + b"".join(chunks)


@pytest.mark.parametrize(
    "order",
    [[1, 0, 2], [0, 2, 1], [2, 1, 0], [0, 1, 1, 2], [0, 0, 1, 2], [0, 2], [1, 2]],
    ids=["swap01", "swap12", "reverse", "duplicate1", "duplicate0", "drop1", "drop0"],
)
def test_reordered_chunks_fail(key_blob: bytes, key: bytes, order: list[int]) -> None:
    chunks = chunks_of(key_blob, KEY_PREAMBLE)
    with pytest.raises(AuthenticationError):
        unseal(_join(key_blob, [chunks[i] for i in order]), key=key)


def test_splicing_header_and_body_of_two_files_fails(key: bytes, plaintext: bytes) -> None:
    a = seal(plaintext, key=key, chunk_size=C)
    b = seal(plaintext, key=key, chunk_size=C)
    with pytest.raises(AuthenticationError):
        unseal(a[:KEY_PREAMBLE] + b[KEY_PREAMBLE:], key=key)


def test_unexpected_secretstream_tag_is_rejected(key: bytes) -> None:
    """A correctly keyed stream using TAG_REKEY is still not a valid .puf file."""
    state = sodium.crypto_secretstream_xchacha20poly1305_state()
    stream_header = sodium.crypto_secretstream_xchacha20poly1305_init_push(state, key)
    header = FileHeader(
        kdf=Kdf.RAW_KEY, chunk_size=C, stream_header=stream_header, key_id=key_id(key)
    )
    preamble = header.to_bytes()
    chunk = sodium.crypto_secretstream_xchacha20poly1305_push(
        state, b"x", preamble, sodium.crypto_secretstream_xchacha20poly1305_TAG_REKEY
    )
    with pytest.raises(FormatError, match="unexpected secretstream tag"):
        unseal(preamble + chunk, key=key)


# --- truncation / extension --------------------------------------------------


def test_dropping_the_final_chunk_is_truncation(key_blob: bytes, key: bytes) -> None:
    chunks = chunks_of(key_blob, KEY_PREAMBLE)
    out = io.BytesIO()
    with pytest.raises(TruncatedError, match="without the final chunk"):
        decrypt_stream(io.BytesIO(_join(key_blob, chunks[:-1])), out, key=key)
    # Released chunks are authentic: a strict prefix of the real plaintext.
    assert out.getvalue() == unseal(key_blob, key=key)[: 2 * C]


def test_dropping_a_full_final_chunk_is_truncation(key: bytes, rng: np.random.Generator) -> None:
    blob = seal(rng.bytes(3 * C), key=key, chunk_size=C)
    with pytest.raises(TruncatedError):
        unseal(blob[: -(C + ABYTES)], key=key)


def test_header_only_is_truncation(key_blob: bytes, password_blob: bytes, key: bytes) -> None:
    with pytest.raises(TruncatedError):
        unseal(key_blob[:KEY_PREAMBLE], key=key)
    with pytest.raises(TruncatedError):
        unseal(password_blob[:PASSWORD_PREAMBLE], password=PASSWORD)


def _expected_truncation_error(cut: int, preamble: int) -> type[PufferCryptoError]:
    if cut < 4:
        return FormatError  # not even the magic: not a .puf file
    if cut < preamble:
        return TruncatedError  # ends inside the header
    leftover = (cut - preamble) % (C + ABYTES)
    if leftover < ABYTES:
        return TruncatedError  # at a chunk boundary, or too short to hold a tag
    return AuthenticationError  # a partial chunk cannot authenticate


def test_every_truncation_is_rejected(key_blob: bytes, key: bytes) -> None:
    for cut in range(len(key_blob)):
        expected = _expected_truncation_error(cut, KEY_PREAMBLE)
        with pytest.raises(PufferCryptoError) as excinfo:
            unseal(key_blob[:cut], key=key)
        assert type(excinfo.value) is expected, cut


def test_password_header_truncation(password_blob: bytes) -> None:
    for cut in range(PASSWORD_PREAMBLE + 1):
        expected = _expected_truncation_error(cut, PASSWORD_PREAMBLE)
        with pytest.raises(PufferCryptoError) as excinfo:
            unseal(password_blob[:cut], password=PASSWORD)
        assert type(excinfo.value) is expected, cut


@pytest.mark.parametrize("extra", [b"\x00", b"trailing", bytes(C + ABYTES)])
def test_data_after_full_size_final_chunk_is_format_error(
    key: bytes, rng: np.random.Generator, extra: bytes
) -> None:
    blob = seal(rng.bytes(2 * C), key=key, chunk_size=C)
    out = io.BytesIO()
    with pytest.raises(FormatError, match="after the final chunk"):
        decrypt_stream(io.BytesIO(blob + extra), out, key=key)
    assert len(out.getvalue()) == C  # the final chunk is withheld


@pytest.mark.parametrize("extra", [b"\x00", b"trailing", bytes(C)])
def test_data_after_short_final_chunk_fails_authentication(
    key_blob: bytes, key: bytes, extra: bytes
) -> None:
    """No length field: appended bytes merge into a short final chunk and break its tag."""
    with pytest.raises(AuthenticationError):
        unseal(key_blob + extra, key=key)


def test_data_after_empty_stream(key: bytes) -> None:
    with pytest.raises(AuthenticationError):
        unseal(seal(b"", key=key, chunk_size=C) + b"x", key=key)


# --- credentials -------------------------------------------------------------


def test_wrong_key_is_reported_before_decrypting(
    key_blob: bytes, key: bytes, other_key: bytes
) -> None:
    out = io.BytesIO()
    with pytest.raises(WrongKeyError, match=key_id(key)) as excinfo:
        decrypt_stream(io.BytesIO(key_blob), out, key=other_key)
    assert key_id(other_key) in str(excinfo.value)
    assert out.getvalue() == b""


def test_wrong_key_with_forged_key_id_fails_authentication(
    key_blob: bytes, other_key: bytes
) -> None:
    forged = key_blob[:12] + bytes.fromhex(key_id(other_key)) + key_blob[20:]
    with pytest.raises(AuthenticationError, match="chunk 0"):
        unseal(forged, key=other_key)


def test_wrong_password(password_blob: bytes) -> None:
    out = io.BytesIO()
    with pytest.raises(AuthenticationError, match="wrong key/password"):
        decrypt_stream(io.BytesIO(password_blob), out, password=PASSWORD + "!")
    assert out.getvalue() == b""


def test_credential_kind_mismatch(key_blob: bytes, password_blob: bytes, key: bytes) -> None:
    with pytest.raises(WrongKeyError, match="pass key="):
        unseal(key_blob, password=PASSWORD)
    with pytest.raises(WrongKeyError, match="password-protected"):
        unseal(password_blob, key=key)


@pytest.mark.parametrize(
    "credentials",
    [{}, {"key": bytes(32), "password": PASSWORD}],
    ids=["neither", "both"],
)
def test_exactly_one_credential(credentials: dict[str, Any], key_blob: bytes) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        seal(b"data", **credentials)
    with pytest.raises(ValueError, match="exactly one"):
        unseal(key_blob, **credentials)


def test_bad_key_length(key_blob: bytes) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        seal(b"data", key=bytes(16))
    with pytest.raises(ValueError, match="32 bytes"):
        unseal(key_blob, key=bytes(16))


def test_empty_password_is_refused(fast_kdf: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        seal(b"data", password="", **fast_kdf)
    with pytest.raises(ValueError, match="must not be empty"):
        seal(b"data", password=b"", **fast_kdf)


def test_passwords_are_nfc_normalised(fast_kdf: dict[str, Any], plaintext: bytes) -> None:
    composed = unicodedata.normalize("NFC", "pässwörd-рыба")
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    blob = seal(plaintext, password=composed, **fast_kdf)
    assert unseal(blob, password=decomposed) == plaintext
    assert unseal(blob, password=composed.encode("utf-8")) == plaintext
    with pytest.raises(AuthenticationError):
        unseal(blob, password=decomposed.encode("utf-8"))  # bytes are used verbatim


# --- encrypt-side validation -------------------------------------------------


@pytest.mark.parametrize("chunk_size", [0, MIN_CHUNK_SIZE - 1, MAX_CHUNK_SIZE + 1])
def test_encrypt_rejects_unreadable_chunk_size(key: bytes, chunk_size: int) -> None:
    out = io.BytesIO()
    with pytest.raises(ValueError, match="chunk_size"):
        encrypt_stream(io.BytesIO(b"x"), out, key=key, chunk_size=chunk_size)
    assert out.getvalue() == b""


@pytest.mark.parametrize(
    ("opslimit", "memlimit"),
    [(0, argon2id.MEMLIMIT_MIN), (17, argon2id.MEMLIMIT_MIN), (1, 1024), (1, 4 * 1024**3 + 1)],
)
def test_encrypt_rejects_unreadable_kdf_cost(opslimit: int, memlimit: int) -> None:
    out = io.BytesIO()
    with pytest.raises(ValueError, match="outside the accepted range"):
        encrypt_stream(
            io.BytesIO(b"x"), out, password=PASSWORD, opslimit=opslimit, memlimit=memlimit
        )
    assert out.getvalue() == b""


def test_encrypt_rejects_misbehaving_rng() -> None:
    with pytest.raises(ValueError, match="salt must be 16 bytes"):
        seal(
            b"x",
            password=PASSWORD,
            opslimit=argon2id.OPSLIMIT_MIN,
            memlimit=argon2id.MEMLIMIT_MIN,
            random_bytes=lambda n: bytes(n // 2),
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"kdf": Kdf.RAW_KEY, "key_id": None},
        {"kdf": Kdf.RAW_KEY, "key_id": "XYZ0000000000000"},
        {"kdf": Kdf.RAW_KEY, "key_id": "00" * 8, "argon2": Argon2Params(1, 8192, bytes(16))},
        {"kdf": Kdf.ARGON2ID, "key_id": None},
        {"kdf": Kdf.ARGON2ID, "key_id": "00" * 8, "argon2": Argon2Params(1, 8192, bytes(16))},
        {"kdf": Kdf.RAW_KEY, "key_id": "00" * 8, "stream_header": bytes(23)},
    ],
)
def test_file_header_invariants(fields: dict[str, Any]) -> None:
    with pytest.raises(FormatError):
        FileHeader(**{"chunk_size": C, "stream_header": bytes(24), **fields})


# --- properties --------------------------------------------------------------


@PROPERTY
@given(
    data=st.binary(max_size=5 * 1024),
    chunk_size=st.sampled_from([1024, 1500, 2048, 4096]),
    key=st.binary(min_size=32, max_size=32),
)
def test_property_stream_round_trip(data: bytes, chunk_size: int, key: bytes) -> None:
    blob = seal(data, key=key, chunk_size=chunk_size)
    n_chunks = max(1, -(-len(data) // chunk_size))
    assert len(blob) == KEY_PREAMBLE + len(data) + n_chunks * ABYTES
    assert unseal(blob, key=key) == data


@PROPERTY
@given(
    plaintext=st.binary(max_size=3 * 1024),
    key=st.binary(min_size=32, max_size=32),
    data=st.data(),
)
def test_property_no_modification_is_ever_accepted(
    plaintext: bytes, key: bytes, data: st.DataObject
) -> None:
    blob = bytearray(seal(plaintext, key=key, chunk_size=1024))
    offset = data.draw(st.integers(0, len(blob) - 1), label="offset")
    blob[offset] ^= data.draw(st.integers(1, 255), label="mask")
    with pytest.raises(PufferCryptoError):
        unseal(bytes(blob), key=key)


@PROPERTY
@given(plaintext=st.binary(max_size=3 * 1024), data=st.data())
def test_property_no_truncation_is_ever_accepted(plaintext: bytes, data: st.DataObject) -> None:
    key = bytes(range(32))
    blob = seal(plaintext, key=key, chunk_size=1024)
    cut = data.draw(st.integers(0, len(blob) - 1), label="cut")
    with pytest.raises(PufferCryptoError):
        unseal(blob[:cut], key=key)
