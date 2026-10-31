"""Short AEAD messages: wire format, libsodium conformance, tampering."""

from __future__ import annotations

import nacl.bindings
import nacl.exceptions
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from puffer.crypto import (
    MESSAGE_AAD,
    NONCE_BYTES,
    AuthenticationError,
    FormatError,
    RandomBytes,
    b64url_decode,
    b64url_encode,
    decrypt_message,
    decrypt_message_b64,
    encrypt_message,
    encrypt_message_b64,
)

PROPERTY = settings(derandomize=True, database=None, deadline=None, max_examples=100)
TAG = 16


def test_round_trip(key: bytes) -> None:
    blob = encrypt_message(b"fish say hi", key)
    assert decrypt_message(blob, key) == b"fish say hi"


def test_empty_plaintext(key: bytes) -> None:
    blob = encrypt_message(b"", key)
    assert len(blob) == NONCE_BYTES + TAG
    assert decrypt_message(blob, key) == b""


def test_wire_format_matches_libsodium(key: bytes, det_random: RandomBytes) -> None:
    """nonce || crypto_aead_xchacha20poly1305_ietf_encrypt(pt, b"puffer-msg-v1", nonce, key)."""
    nonce = det_random(NONCE_BYTES)
    blob = encrypt_message(b"payload", key, nonce=nonce)
    expected = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
        b"payload", b"puffer-msg-v1", nonce, key
    )
    assert MESSAGE_AAD == b"puffer-msg-v1"
    assert blob == nonce + expected
    assert len(blob) == NONCE_BYTES + len(b"payload") + TAG


def test_aad_is_bound(key: bytes) -> None:
    blob = encrypt_message(b"payload", key)
    with pytest.raises(nacl.exceptions.CryptoError):
        nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
            blob[NONCE_BYTES:], b"other-protocol", blob[:NONCE_BYTES], key
        )


def test_nonce_comes_from_injected_rng(key: bytes) -> None:
    calls: list[int] = []

    def fixed(n: int) -> bytes:
        calls.append(n)
        return bytes(range(n))

    blob = encrypt_message(b"x", key, random_bytes=fixed)
    assert calls == [NONCE_BYTES]
    assert blob[:NONCE_BYTES] == bytes(range(NONCE_BYTES))


def test_fresh_nonce_per_message(key: bytes) -> None:
    assert encrypt_message(b"same", key) != encrypt_message(b"same", key)


def test_wrong_key(key: bytes, other_key: bytes) -> None:
    blob = encrypt_message(b"secret", key)
    with pytest.raises(AuthenticationError):
        decrypt_message(blob, other_key)


def test_every_single_byte_flip_is_rejected(key: bytes) -> None:
    blob = encrypt_message(b"twelve bytes", key)
    for offset in range(len(blob)):
        for mask in (0x01, 0x80):
            tampered = bytearray(blob)
            tampered[offset] ^= mask
            with pytest.raises(AuthenticationError):
                decrypt_message(bytes(tampered), key)


def test_too_short_is_format_error(key: bytes) -> None:
    blob = encrypt_message(b"", key)
    for cut in range(len(blob)):
        with pytest.raises(FormatError, match="too short"):
            decrypt_message(blob[:cut], key)


def test_truncated_but_long_enough_fails_authentication(key: bytes) -> None:
    blob = encrypt_message(b"some plaintext", key)
    with pytest.raises(AuthenticationError):
        decrypt_message(blob[:-1], key)
    with pytest.raises(AuthenticationError):
        decrypt_message(blob + b"\x00", key)


def test_b64_round_trip(key: bytes) -> None:
    token = encrypt_message_b64("рыба 🐡".encode(), key)
    assert "=" not in token
    assert len(b64url_decode(token)) == NONCE_BYTES + len("рыба 🐡".encode()) + TAG
    assert decrypt_message_b64(token, key).decode() == "рыба 🐡"
    padded = token + "=" * (-len(token) % 4)
    assert decrypt_message_b64(padded, key).decode() == "рыба 🐡"


def test_b64_explicit_nonce_is_deterministic(key: bytes) -> None:
    nonce = bytes(NONCE_BYTES)
    token = encrypt_message_b64(b"m", key, nonce=nonce)
    assert token == b64url_encode(encrypt_message(b"m", key, nonce=nonce))


@pytest.mark.parametrize("token", ["", "not base64!", "AAAA AAAA", "A"])
def test_b64_garbage_is_format_error(token: str, key: bytes) -> None:
    with pytest.raises(FormatError):
        decrypt_message_b64(token, key)


def test_bad_key_arguments() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        encrypt_message(b"m", b"short")
    with pytest.raises(ValueError, match="32 bytes"):
        decrypt_message(bytes(64), bytes(31))
    with pytest.raises(TypeError, match="bytes"):
        encrypt_message(b"m", "k" * 32)  # type: ignore[arg-type]


def test_bad_nonce_length(key: bytes) -> None:
    with pytest.raises(ValueError, match="nonce must be 24 bytes"):
        encrypt_message(b"m", key, nonce=bytes(12))
    with pytest.raises(ValueError, match="nonce must be 24 bytes"):
        encrypt_message(b"m", key, random_bytes=lambda n: bytes(n - 1))


@PROPERTY
@given(plaintext=st.binary(max_size=2048), key=st.binary(min_size=32, max_size=32))
def test_property_round_trip(plaintext: bytes, key: bytes) -> None:
    blob = encrypt_message(plaintext, key)
    assert len(blob) == NONCE_BYTES + len(plaintext) + TAG
    assert decrypt_message(blob, key) == plaintext
    assert decrypt_message_b64(b64url_encode(blob), key) == plaintext


@PROPERTY
@given(
    plaintext=st.binary(max_size=256),
    key=st.binary(min_size=32, max_size=32),
    data=st.data(),
)
def test_property_any_modification_is_rejected(
    plaintext: bytes, key: bytes, data: st.DataObject
) -> None:
    blob = bytearray(encrypt_message(plaintext, key))
    offset = data.draw(st.integers(0, len(blob) - 1), label="offset")
    blob[offset] ^= data.draw(st.integers(1, 255), label="mask")
    with pytest.raises(AuthenticationError):
        decrypt_message(bytes(blob), key)
